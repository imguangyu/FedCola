import copy
import torch

from .fedavgclient import FedavgClient
from src import MetricManager, TqdmToLogger
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)



class FedproxClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedproxClient, self).__init__(**kwargs)

    def update(self):
        mm = MetricManager(self.eval_metrics) if self.modality!= 'img+txt' else MetricManager([])
        self.model.train()
        self.model.to(self.device)

        global_model = copy.deepcopy(self.model)
        for param in global_model.parameters(): 
            param.requires_grad = False
        

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
                
                prox = 0.
                for name, param in self.model.named_parameters():
                    prox += (param - global_model.get_parameter(name)).norm(2)
                loss += self.args.mu * (0.5 * prox)

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