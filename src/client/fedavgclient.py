import copy
import torch
import inspect
import itertools
from src.criterions.segmentation_loss import SegLoss

from .baseclient import BaseClient
from src import MetricManager, TqdmToLogger
from torch import nn

import logging
logger = logging.getLogger(__name__)


class FedavgClient(BaseClient):
    def __init__(self, args, training_set, test_set, task='cls', eval_metrics=['acc1'], modality='ct', writer=None, criterion='CrossEntropyLoss'):
        super(FedavgClient, self).__init__()
        self.args = args
        self.training_set = training_set
        self.test_set = test_set
        
        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = nn.__dict__[criterion]

        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False, test=True)

        self.task = task
        self.modality = modality
        self.eval_metrics = eval_metrics

        self.writer = writer

    def _refine_optim_args(self, args):
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        return refined_args

    def _create_dataloader(self, dataset, shuffle, test=True):
        if self.args.B == 0 :
            self.args.B = len(self.training_set)
        if not test:
            if self.modality == 'img+txt':
                return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle, persistent_workers=True, num_workers=8, pin_memory=True)
            else:
                return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle)
        else:
            return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle)
        
    def update(self):
        mm = MetricManager(self.eval_metrics) if self.modality!= 'img+txt' else MetricManager([])
        self.model.train()
        self.model.to(self.device)

        if self.args.distributed or (self.args.mm_distributed and self.modality == 'img+txt'):
            self.model = nn.DataParallel(self.model).cuda()
        
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))
        logger.info(f'[{self.task.upper()}] [{self.modality.upper()}] ...working on client {self.id}... ')
        for e in TqdmToLogger(
                    range(self.args.E), 
                    logger=logger, 
                    desc=f'[{self.task.upper()}] [{self.modality.upper()}] ...update client {self.id}... ',
                    total=self.args.E
                    ):
            num = 0
            for batch in self.train_loader:
                if num >= 2 and self.args.debug:
                    mm.aggregate(num * self.args.B, e + 1)
                    break



                optimizer.zero_grad()

                if self.modality == 'img':
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model([inputs, None])[0]
                    loss = self.criterion()(outputs.to(targets.device), targets)
                elif self.modality == 'txt':
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model([None, inputs])[1]
                    loss = self.criterion()(outputs.to(targets.device), targets)
                elif self.modality == 'img+txt':
                    inputs, targets, _, _, _ = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model([inputs, targets], feat_out=True)
                    loss = self.criterion()(*outputs)
                    
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                mm.track(loss.item(), outputs.to(targets.device), targets) if self.modality!= 'img+txt' else mm.track(loss.item(), outputs[0].to(targets.device))
                num += 1
            else:
                mm.aggregate(len(self.training_set), e + 1)
                res = mm.results[e+1]
                # self.writer.log({f"Debug/client_{self.id}/loss": res["loss"], 
                #                  f"Debug/client_{self.id}/acc1": res["metrics"]["acc1"],
                #                  }, e+1)
                logger.info(f'[Client {self.id}] loss: {res["loss"]}, acc1: {res["metrics"]["acc1"]}') if self.modality!= 'img+txt' else logger.info(f'[Client {self.id}] loss: {res["loss"]}')
        else:
            if self.args.distributed or (self.args.mm_distributed and self.modality == 'img+txt'):
                self.model = self.model.module
            self.model.to('cpu')
            
        return mm.results

    @torch.inference_mode()
    def evaluate(self): # Not used
        if self.args.train_only: # `args.test_size` == 0
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.eval_metrics)
        self.model.eval()
        self.model.to(self.device)
        # self.model = nn.DataParallel(self.model).to('cuda:0')

        logger.info(f'[{self.task.upper()}] [{self.modality.upper()}] ...evaluating on client {self.id}... ')

        num = 0
        for inputs, targets in self.test_loader:    
            if num >= 2 and self.args.debug:
                # self.model = self.model.module
                self.model.to('cpu')
                mm.aggregate(num * self.args.B)
                break
            # logger.info(f"{self.id}: dataloaded")
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs, task=self.task)
            # logger.info(f"{self.id}: output got")
            loss = self.criterion()(outputs, targets)

            # logger.info(f"{self.id}: loss got")

            mm.track(loss.item(), outputs, targets)
            # logger.info(f"{self.id}: tracked")
            num += 1
        else:
            # self.model = self.model.module
            self.model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results

    def download(self, models):
        self.model = copy.deepcopy(models[self.dataset])

    def upload(self):
        # return itertools.chain.from_iterable([self.model.named_parameters(), self.model.named_buffers()])
        sd = self.model.cpu().state_dict()
        if self.args.with_aux and self.modality != 'img+txt':
            new_sd = copy.deepcopy(sd)

            if self.args.aux_attn_only:
                if self.args.aux_mlp_only:
                    raise ValueError('Both aux_attn_only and aux_mlp_only cannot be True.')
                layer_name = ('attn.qkv', 'attn.proj')
            elif self.args.aux_mlp_only:
                layer_name = ('mlp.fc1','mlp.fc2')
            else:
                layer_name = ('attn.qkv', 'attn.proj', 'mlp.fc1','mlp.fc2')
            
            with torch.no_grad():
                for k,v in sd.items():
                    if any([name in k for name in layer_name]):
                        if 'aux' not in k and 'weight' in k:
                            new_sd[k] = v + new_sd[k.replace('weight', 'aux_weight')] * new_sd[k.replace('weight', 'cross_modal_scale')]
                for k,v in sd.items():
                    if 'aux' in k or 'cross_modal_scale' in k:
                        new_sd.pop(k)
            return new_sd


        return sd
    
    def __len__(self):
        return len(self.training_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'
