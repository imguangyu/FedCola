import inspect
import os
import gc
import json
import torch
import random
import logging
import numpy as np
import concurrent.futures
import copy
from copy import deepcopy

from importlib import import_module
from collections import ChainMap, defaultdict

from src import init_weights, TqdmToLogger, MetricManager
from .baseserver import BaseServer

import timm
import wandb
import src.models.mome

import torch.nn as nn
from src.metrics.eval_coco import COCOEvaluator

from multiprocessing import Pool
import re



logger = logging.getLogger(__name__)

DATASET_2_TASK = {
    'BraTS': 'seg',
    'MedMNIST': 'cls',
    'CIFAR100': 'cls',
    'AG_NEWS': 'cls',
    'MTSamples': 'cls',
    'MedicalAbstracts': 'cls',
    'Flickr30k': 'rtv',
    'Coco': 'rtv',
}

DATASET_2_MODALITY = {
    'BraTS': 't1',
    'MedMNIST': 'img',
    'CIFAR100': 'img',
    'AG_NEWS': 'txt',
    'MTSamples': 'txt',
    'MedicalAbstracts': 'txt',
    'Flickr30k': 'img+txt',
    'Coco': 'img+txt',
}

# dataset version
NUM_CLASS = {
    'CIFAR100': 100,
    'AG_NEWS': 4,
    'MedMNIST': 11,
    'MTSamples': 40,
    'MedicalAbstracts': 5,
    'Flickr30k': None,
    'Coco': None,
}

MODALITY_2_DATASET = {
    'ct': 'MedMNIST',
    'mic': 'MedMNIST',
    "pat": 'MedMNIST',
    'der': 'MedMNIST',
    't1': 'BraTS',
    'flair': 'BraTS',
    't2': 'BraTS',
}

TASK_2_CRITERION = {
    'cls': 'CrossEntropyLoss',
    'seg': 'SegLoss',
    'img+txt':  'ContrastiveLoss' #'MCSoftContrastiveLoss', #'
}

MM_METRICS = {
    'recall_1',
    'recall_5',
    'recall_10',
    'rsum'
}

VOCAB_SIZES = {
    'Flickr30k': 7732,
    'MedicalAbstracts':20264
}

def get_name_type(name):
    if 'embeddings' in name:
        return 'embedding'
    elif 'attention' in name or 'attn' in name:
        return 'attn'
    elif 'blocks' in name:
        return 'blocks'
    elif 'mlp' in name:
        return 'mlp'
    else:
        return 'task'

def get_first_number(string):
    match = re.search(r'\d+', string)
    if match:
        return int(match.group())
    else:
        return None
            
def get_name_modality(name, modalities):
    idx = get_first_number(name)
    return modalities[idx] if idx is not None else None

class FedavgServer(BaseServer):
    def __init__(self, args, writer, server_dataset, client_datasets, model_str):
        super(FedavgServer, self).__init__()
        self.args = args
        self.writer = writer

        self.round = 0 # round indicator
        if self.args.eval_type != 'local': # global holdout set for central evaluation
            self._set_loaders(server_dataset)
        # self.global_model = self._init_model(model_str) # global model
        self.global_models = self._init_model(model_str) # global model
        self._init_param_scope(args.shared_param, args.share_scope)
        # self.sync_shared_params()
        self._set_evaluator()
        self.opt_kwargs = dict(lr=self.args.lr, momentum=self.args.beta1) # federation algorithm arguments
        self.curr_lr = self.args.lr # learning rate
        self.clients = self._create_clients(client_datasets) # clients container
        self.results = defaultdict(dict) # logging results container
        self.server_device = self.args.server_device

        if type(args.Cs) != list or len(args.Cs) == 1:
            if len(args.Cs) == 1:
                self.args.Cs = self.args.Cs * len(self.args.datasets)
            else:
                self.args.Cs = [self.args.Cs] * len(self.args.datasets)
        self.Cs = {dataset: C for dataset, C in zip(self.args.datasets, self.args.Cs)}

    def _init_model(self, model_str):
        self.args.datasets = self.args.datasets[:-1]
        models = {}
        for i, dataset in enumerate(self.args.datasets): # CIFAR, AGNEWS, Flickr
            # try:
                self.args.vocab_size = VOCAB_SIZES[dataset] if dataset in VOCAB_SIZES else 30522
                if DATASET_2_MODALITY[dataset] == 'img':
                    models[dataset] = timm.create_model(model_str, pretrained=self.args.pretrained, num_classes=[NUM_CLASS[dataset], None], modalities=[self.args.modalities[i], None], args=self.args, tasks=[DATASET_2_TASK[dataset], None], with_aux=self.args.with_aux, aux_trained=self.args.aux_trained, aux_attn_only=self.args.aux_attn_only, aux_mlp_only=self.args.aux_mlp_only)
                elif DATASET_2_MODALITY[dataset] == 'txt':
                    models[dataset] = timm.create_model(model_str, pretrained=self.args.pretrained, num_classes=[None, NUM_CLASS[dataset]], modalities=[None, self.args.modalities[i]], args=self.args, tasks=[None, DATASET_2_TASK[dataset]], with_aux=self.args.with_aux, aux_trained=self.args.aux_trained, aux_attn_only=self.args.aux_attn_only, aux_mlp_only=self.args.aux_mlp_only)
                elif DATASET_2_MODALITY[dataset] == 'img+txt':
                    models[dataset] = timm.create_model(model_str, pretrained=self.args.pretrained, num_classes=[None, None], modalities=['img', 'txt'], args=self.args, tasks=[DATASET_2_TASK[dataset], DATASET_2_TASK[dataset]], with_aux=self.args.with_aux, aux_trained=self.args.aux_trained, aux_attn_only=self.args.aux_attn_only, aux_mlp_only=self.args.aux_mlp_only)
            # except:
                # models[dataset] = timm.create_model(model_str, pretrained=self.args.pretrained, num_classes=NUM_CLASS[dataset])
        return models
    
    def sync_shared_params(self):
        sd = self.global_models[self.args.datasets[-1]].state_dict()
        
        for model in self.global_models.values():
            new_sd = model.required_params()
            for k,v in sd.items():
                if k in new_sd.keys() and self.param_scope[k] != 'dataset':
                    new_sd[k] = v
            model.load_state_dict(new_sd, strict=False)
    
    def _set_loaders(self, datasets):
        self.server_dataset = datasets[1] #uni-modal datasets
        # self.server_dataset_name = datasets[0][0].name
        # self.server_modality = datasets[0][0].modality
        # self.train_loader = torch.utils.data.DataLoader(dataset=datasets[0][0], batch_size=self.args.B, shuffle=True)
        # self.test_loader = torch.utils.data.DataLoader(dataset=datasets[0][1], batch_size=self.args.B, shuffle=False)

    def _set_evaluator(self):

        evaluator = COCOEvaluator('matmul', n_crossfolds=5, extract_device=self.args.server_device, eval_device=self.args.server_device, verbose=False)
        evaluator.set_logger(logger)
        self.evaluator = evaluator
    
    def _init_param_scope(self, shared_param, share_scope):
        # args.shared_param = 'attn', 'none', 'blocks'
        # args.share_scope = 'modality','dataset','all'
        # params_stragegy : ['all', 'modality', 'dataset']


        self.param_scope = {}
        pram_names = []
        for model in self.global_models.values():
            for key in model.state_dict().keys():
                if key not in pram_names:
                    pram_names.append(key)
        # pram_names = self.global_model.state_dict().keys()
        if shared_param == 'none':
            for name in pram_names:
                self.param_scope[name] = 'dataset' 

        # elif strategy == 'task':
        #     for name in pram_names:
        #         type = get_name_type(name)
        #         self.param_strategy[name] = 'task' if 'head' not in name else 'modality'
        elif shared_param == 'attn':
            for name in pram_names:
                type = get_name_type(name)
                if type == 'embedding':
                    self.param_scope[name] = 'dataset'
                elif type == 'attn':
                    self.param_scope[name] = share_scope
                elif type == 'blocks':
                    self.param_scope[name] = 'dataset'
                else:
                    self.param_scope[name] = 'dataset'
        elif shared_param == 'blocks':
            for name in pram_names:
                type = get_name_type(name)
                if type == 'embedding':
                    self.param_scope[name] = 'dataset'
                elif type == 'attn':
                    self.param_scope[name] = 'dataset'
                elif type == 'blocks':
                    self.param_scope[name] = share_scope
                else:
                    self.param_scope[name] = 'dataset'
        elif shared_param == 'mlp':
            for name in pram_names:
                type = get_name_type(name)
                if type == 'embedding':
                    self.param_scope[name] = 'dataset'
                elif type == 'attn':
                    self.param_scope[name] = 'dataset'
                elif type == 'blocks':
                    self.param_scope[name] = 'dataset'
                elif type == 'mlp':
                    self.param_scope[name] = share_scope
                else:
                    self.param_scope[name] = 'dataset'
        
    
    def _get_algorithm(self, model, **kwargs):
        ALGORITHM_CLASS = import_module(f'..algorithm.{self.args.algorithm}', package=__package__).__dict__[f'{self.args.algorithm.title()}Optimizer']
        optimizer = ALGORITHM_CLASS(params=model.state_dict(), **kwargs)
        # if self.args.algorithm != 'fedsgd': 
            # optimizer.add_param_group(dict(params=list(self.global_model.buffers()))) # add buffered tensors (i.e., gamma and beta of batchnorm layers)
        return optimizer

    def _create_clients(self, client_datasets):
        CLINET_CLASS = import_module(f'..client.{self.args.algorithm}client', package=__package__).__dict__[f'{self.args.algorithm.title()}Client']

        def __create_client(identifier, datasets):
            client = CLINET_CLASS(args=self.args, training_set=datasets[0], test_set=datasets[1],task=datasets[2],modality=datasets[3], eval_metrics=['acc1'] if datasets[2]== 'cls' else ['f1'], 
                                  criterion=TASK_2_CRITERION[datasets[2]],
                                  writer=self.writer)
            client.id = identifier
            client.dataset = datasets[4]
            client.device = 'cuda:%d' % (identifier % torch.cuda.device_count()) if torch.cuda.is_available() else 'cpu'
            # client.device = 'cuda:0' 
            return client

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Create clients!')
        clients = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(int(self.args.K), os.cpu_count() - 1)) as workhorse:
            futures = []
            for identifier, datasets in TqdmToLogger(
                enumerate(client_datasets), 
                logger=logger, 
                desc=f'[{self.args.algorithm.upper()}][Round: {str(self.round).zfill(4)}] ...creating clients... ',
                total=len(client_datasets)
                ):
                futures.append(workhorse.submit(__create_client, identifier, datasets))

            for future in concurrent.futures.as_completed(futures):
                client = future.result()
                clients.append(client)

        clients.sort(key=lambda client: client.id)

        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully created {self.args.K} clients!')
        return clients

    def _sample_clients(self, exclude=[]):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Sample clients!')
        if self.args.equal_sampled:

            sampled_client_ids = []
            for i, dataset in enumerate(self.args.datasets):
                ids = [client.id for client in self.clients if client.dataset == dataset]
                num_sampled_clients = max(int(self.Cs[dataset] * len(ids)), 1)
                sampled_client_ids += sorted(random.sample(ids, num_sampled_clients))

            sampled_client_ids = sorted(sampled_client_ids)
        else:
            if exclude == []: # Update - randomly select max(floor(C * K), 1) clients
                num_sampled_clients = max(int(self.args.C * self.args.K), 1)
                sampled_client_ids = sorted(random.sample([i for i in range(self.args.K)], num_sampled_clients))
            else: # Evaluation - randomly select unparticipated clients in amount of `eval_fraction` multiplied
                num_unparticipated_clients = self.args.K - len(exclude)
                if num_unparticipated_clients == 0: # when C = 1, i.e., need to evaluate on all clients
                    num_sampled_clients = self.args.K
                    sampled_client_ids = sorted([i for i in range(self.args.K)])
                else:
                    num_sampled_clients = max(int(self.args.eval_fraction * num_unparticipated_clients), 1)
                    sampled_client_ids = sorted(random.sample([identifier for identifier in [i for i in range(self.args.K)] if identifier not in exclude], num_sampled_clients))
        
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...{num_sampled_clients} clients are selected!')
        if self.args.warmup_modality != 'none' and self.round <= self.args.warmup_rounds:
            sampled_client_ids = [id for id in sampled_client_ids if self.clients[id].modality == self.args.warmup_modality]

        for i, id in enumerate(sampled_client_ids):
            self.clients[id].device = 'cuda:%d' % (i % torch.cuda.device_count()) if torch.cuda.is_available() else 'cpu'
        return sampled_client_ids
    

    def _log_results(self, resulting_sizes, results, eval, participated, save_raw):
        losses, metrics, num_samples = list(), defaultdict(list), list()
        log_dict = defaultdict(dict)
        averaged = 0.
        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [{self.clients[identifier].dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [CLIENT] < {str(identifier).zfill(6)} > '
            if eval: # get loss and metrics
                # loss
                loss = result['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # metrics
                for metric, value in result['metrics'].items():
                    client_log_string += f'| {metric}: {value:.4f} '
                    metrics[metric].append(value)

                    log_dict['Train/'+ self.clients[identifier].modality+'_'+metric] = value
                    averaged += value
            else: # same, but retireve results of last epoch's
                # loss
                loss = result[self.args.E]['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # metrics
                for name, value in result[self.args.E]['metrics'].items():
                    client_log_string += f'| {name}: {value:.4f} '
                    metrics[name].append(value)                
            # get sample size
            num_samples.append(resulting_sizes[identifier])

            # log per client
            logger.info(client_log_string)
        else:
            num_samples = np.array(num_samples).astype(float)
            
        for metric, value in metrics.items():
            log_dict["Test" if eval else "Training" + f'/{metric}_Avg.'] = np.mean(value)

        log_dict["Test" if eval else "Training"+'/All_Avg.'] = averaged / self.args.K
        self.writer.log(log_dict, self.round)

        # aggregate into total logs
        result_dict = defaultdict(dict)
        total_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [SUMMARY] ({len(resulting_sizes)} clients):'

        # loss
        losses_array = np.array(losses).astype(float)
        weighted = losses_array.dot(num_samples) / sum(num_samples); std = losses_array.std()
        
        top10_indices = np.argpartition(losses_array, -int(0.1 * len(losses_array)))[-int(0.1 * len(losses_array)):] if len(losses_array) > 1 else 0
        top10 = np.atleast_1d(losses_array[top10_indices])
        top10_mean, top10_std = top10.dot(np.atleast_1d(num_samples[top10_indices])) / num_samples[top10_indices].sum(), top10.std()

        bot10_indices = np.argpartition(losses_array, max(1, int(0.1 * len(losses_array)) - 1))[:max(1, int(0.1 * len(losses_array)))] if len(losses_array) > 1 else 0
        bot10 = np.atleast_1d(losses_array[bot10_indices])
        bot10_mean, bot10_std = bot10.dot(np.atleast_1d(num_samples[bot10_indices])) / num_samples[bot10_indices].sum(), bot10.std()

        total_log_string += f'\n    - Loss: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Top 10% ({top10_mean:.4f}) Std. ({top10_std:.4f}) | Bottom 10% ({bot10_mean:.4f}) Std. ({bot10_std:.4f})'
        result_dict['loss'] = {
            'avg': weighted.astype(float), 'std': std.astype(float), 
            'top10p_avg': top10_mean.astype(float), 'top10p_std': top10_std.astype(float), 
            'bottom10p_avg': bot10_mean.astype(float), 'bottom10p_std': bot10_std.astype(float)
        }

        if save_raw:
            result_dict['loss']['raw'] = losses

        self.writer.log(
            {f'Local {"Test" if eval else "Training"} Loss ' + eval * f'({"In" if participated else "Out"})/'+'Avg.': weighted, f'Local {"Test" if eval else "Training"} Loss ' + eval * f'({"In" if participated else "Out"})/'+'Std.': std},
            self.round
        )

        # metrics #NOTE: Unnecessary for current 1 client setting.
        # for name, val in metrics.items():
        #     val_array = np.array(val).astype(float)
        #     weighted = val_array.dot(num_samples) / sum(num_samples); std = val_array.std()
            
        #     top10_indices = np.argpartition(val_array, -int(0.1 * len(val_array)))[-int(0.1 * len(val_array)):] if len(val_array) > 1 else 0
        #     top10 = np.atleast_1d(val_array[top10_indices])
        #     top10_mean, top10_std = top10.dot(np.atleast_1d(num_samples[top10_indices])) / num_samples[top10_indices].sum(), top10.std()

        #     bot10_indices = np.argpartition(val_array, max(1, int(0.1 * len(val_array)) - 1))[:max(1, int(0.1 * len(val_array)))] if len(val_array) > 1 else 0
        #     bot10 = np.atleast_1d(val_array[bot10_indices])
        #     bot10_mean, bot10_std = bot10.dot(np.atleast_1d(num_samples[bot10_indices])) / num_samples[bot10_indices].sum(), bot10.std()

        #     total_log_string += f'\n    - {name.title()}: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Top 10% ({top10_mean:.4f}) Std. ({top10_std:.4f}) | Bottom 10% ({bot10_mean:.4f}) Std. ({bot10_std:.4f})'
        #     result_dict[name] = {
        #         'avg': weighted.astype(float), 'std': std.astype(float), 
        #         'top10p_avg': top10_mean.astype(float), 'top10p_std': top10_std.astype(float), 
        #         'bottom10p_avg': bot10_mean.astype(float), 'bottom10p_std': bot10_std.astype(float)
        #     }
                
        #     if save_raw:
        #         result_dict[name]['raw'] = val

        #     self.writer.log(
        #         {f'Local {"Test" if eval else "Training"} {name.title()}' + eval * f' ({"In" if participated else "Out"})/'+'Avg.': weighted, f'Local {"Test" if eval else "Training"} {name.title()}' + eval * f' ({"In" if participated else "Out"})/'+'Std.': std},
        #         self.round
        #     )
            # self.writer.flush()
        
        # log total message
        logger.info(total_log_string)
        return result_dict
    
    def _freeze_shared_params(self, client):
        for name, param in client.model.named_parameters():
            if self.param_scope[name] == 'all':
                param.requires_grad = False
    
    def _unfreeze_params(self, client):
        for name, param in client.model.named_parameters():
                param.requires_grad = True


    def _refine_optim_args(self, args):
        required_args = inspect.getfullargspec(torch.optim.__dict__[self.args.optimizer])[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        return refined_args
    
    # def _train(self):
    #     self.dataset = self.train_loader.dataset.name
    #     logger.info(f'[{self.args.algorithm.upper()}] [{self.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Server training!')

    #     self.criterion = torch.nn.__dict__[TASK_2_CRITERION['img+txt']]
    #     self.global_model = self.global_models[self.dataset]

    #     self.global_model.train()
    #     self.global_model.to(self.server_device)

    #     if self.args.distributed:
    #         self.global_model = nn.DataParallel(self.global_model).cuda()

    #     optimizer = torch.optim.__dict__[self.args.optimizer](self.global_model.parameters(), **self._refine_optim_args(self.args))

    #     mm = MetricManager([])

    #     num = 0
    #     for inputs, targets, _, _, _ in self.train_loader:
 
    #         optimizer.zero_grad()

    #         if num >= 2 and self.args.debug:
    #             mm.aggregate(num * self.args.B)
    #             break

    #         inputs, targets = inputs.to(self.server_device), targets.to(self.server_device)

    #         outputs = self.global_model([inputs, targets], feat_out=True)
    #         loss = self.criterion()(*outputs)
                
    #         loss.backward()
    #         if self.args.max_grad_norm > 0:
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
    #         optimizer.step()

    #         mm.track(loss.item(), outputs[0].to(targets.device))
    #         num += 1
    #     else:
    #         mm.aggregate(len(self.training_set))
    #         res = mm.results
    #         # self.writer.log({f"Debug/client_{self.id}/loss": res["loss"], 
    #         #                  f"Debug/client_{self.id}/acc1": res["metrics"]["acc1"],
    #         #                  }, e+1)
    #         logger.info(f'[Client {self.id}] loss: {res["loss"]}')

    def __update_clients(self,client, retain_model=True):
            if client.model is None:
                client.download(self.global_models)
            client.args.lr = self.curr_lr
            
            if self.args.freeze_modality != 'none':
                if client.modality == self.args.freeze_modality:
                    if self.round <= (self.args.freeze_rounds + self.args.warmup_rounds) and self.round > self.args.warmup_rounds:
                        self._freeze_shared_params(client)
                    elif self.round > (self.args.freeze_rounds + self.args.warmup_rounds):
                        self._unfreeze_params(client)
            update_result = client.update()
            if not retain_model:
                client.model = None
            return {client.id: len(client.training_set)}, {client.id: update_result}


    def _request(self, ids, eval, participated, retain_model, save_raw):
        def __update_clients(client):
            if client.model is None:
                client.download(self.global_models)
            client.args.lr = self.curr_lr
            
            if self.args.freeze_modality != 'none':
                if client.modality == self.args.freeze_modality:
                    if self.round <= (self.args.freeze_rounds + self.args.warmup_rounds) and self.round > self.args.warmup_rounds:
                        self._freeze_shared_params(client)
                    elif self.round > (self.args.freeze_rounds + self.args.warmup_rounds):
                        self._unfreeze_params(client)
            update_result = client.update()
            if not retain_model:
                client.model = None
            return {client.id: len(client.training_set)}, {client.id: update_result}

        def __evaluate_clients(client, require_model=True):
            if client.model is None or require_model:
                client.download(self.global_models)
            eval_result = client.evaluate() 
            if not retain_model:
                client.model = None
            return {client.id: len(client.test_set)}, {client.id: eval_result}

        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Request {"updates" if not eval else "evaluation"} to {"all" if ids is None else len(ids)} clients!')
        if eval:
            if self.args.train_only:
                return None
            results = []
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers= self.args.num_thread) as workhorse: # min(len(ids), os.cpu_count() - 1)
                for idx in TqdmToLogger(
                    ids, 
                    logger=logger, 
                    desc=f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...evaluate clients... ',
                    total=len(ids)
                    ):
                    futures.append(workhorse.submit(__evaluate_clients, self.clients[idx])) 

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
            _eval_sizes, _eval_results = list(map(list, zip(*results)))
            _eval_sizes, _eval_results = dict(ChainMap(*_eval_sizes)), dict(ChainMap(*_eval_results))
            self.results[self.round][f'clients_evaluated_{"in" if participated else "out"}'] = self._log_results(
                _eval_sizes, 
                _eval_results, 
                eval=True, 
                participated=participated,
                save_raw=save_raw
            )
            logger.info(f'[{self.args.algorithm.upper()}][Round: {str(self.round).zfill(4)}] ...completed evaluation of {"all" if ids is None else len(ids)} clients!')
            return None
        else:
            if self.args.mp:
                with concurrent.futures.ProcessPoolExecutor(max_workers= self.args.num_thread) as workhorse:
                    results = workhorse.map(self.__update_clients, [self.clients[idx] for idx in ids])
            else:
                results = []
                futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.num_thread) as workhorse: # min(len(ids), os.cpu_count() - 1)
                    for idx in TqdmToLogger(
                        ids, 
                        logger=logger, 
                        desc=f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...update clients... ',
                        total=len(ids)
                        ):
                        futures.append(workhorse.submit(__update_clients, self.clients[idx]))

                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        results.append(result)

            update_sizes, _update_results = list(map(list, zip(*results)))
            update_sizes, _update_results = dict(ChainMap(*update_sizes)), dict(ChainMap(*_update_results))
            self.results[self.round]['clients_updated'] = self._log_results(
                update_sizes, 
                _update_results, 
                eval=False, 
                participated=True,
                save_raw=False
            )
            logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...completed updates of {"all" if ids is None else len(ids)} clients!')
            return update_sizes
    
    def _aggregate(self, ids, updated_sizes, fedavg=False):
        assert set(updated_sizes.keys()) == set(ids)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')
        
        # calculate mixing coefficients according to sample sizes
        # coefficients = {identifier: float(nuemrator / sum(updated_sizes.values())) for identifier, nuemrator in updated_sizes.items()}
        final_sd = self.global_model.cpu().required_params()
        # for k,v in final_sd.items():
        #     final_sd[k] = torch.zeros_like(v).float()

        if fedavg:
            coefficients = {}
            for param_name, _ in self.global_model.cpu().required_params().items():
                new_nuemrator = {}
                old_sum = sum(updated_sizes.values())
                for identifier, nuemrator in updated_sizes.items():
                        if self.param_scope[param_name] == 'all':
                            new_nuemrator[identifier] = nuemrator 
                        elif self.param_scope[param_name] == 'dataset':
                            new_nuemrator[identifier] = nuemrator if self.clients[identifier].dataset == self.dataset else 0
                        elif self.param_scope[param_name] == 'task':
                            new_nuemrator[identifier] = nuemrator if self.clients[identifier].task == self.task else 0
                        elif self.param_scope[param_name] == 'modality':
                            new_nuemrator[identifier] = nuemrator if self.clients[identifier].modality == self.modality else 0

                coefficients[param_name] = {identifier: float(nuemrator / sum(new_nuemrator.values())) if sum(new_nuemrator.values())!= 0 else 0 for identifier, nuemrator in new_nuemrator.items()}
        else:
            coefficients = {}
            for param_name, _ in self.global_model.cpu().required_params().items():
                new_nuemrator = {}
                old_sum = sum(updated_sizes.values())
                param_modality = get_name_modality(param_name, self.args.modalities)
                for identifier, nuemrator in updated_sizes.items():
                        if self.param_scope[param_name] == 'all':
                            new_nuemrator[identifier] = nuemrator 
                        elif self.param_scope[param_name] == 'dataset':
                            new_nuemrator[identifier] = nuemrator if self.clients[identifier].dataset == self.dataset else 0
                        elif self.param_scope[param_name] == 'task':
                            new_nuemrator[identifier] = nuemrator if self.clients[identifier].task == self.task else 0
                        elif self.param_scope[param_name] == 'modality':
                            new_nuemrator[identifier] = nuemrator if (self.clients[identifier].modality in self.modality or self.modality in self.clients[identifier].modality) else 0
                        elif self.param_scope[param_name] == 'modality_exact':
                            new_nuemrator[identifier] = nuemrator if (self.clients[identifier].modality == param_modality or param_modality in self.clients[identifier].modality) else 0
                        # Code for modality scaling
                        if self.clients[identifier].modality != self.modality and self.out_modality_scale != 1:
                            old_sum -= new_nuemrator[identifier]
                            new_nuemrator[identifier] *= self.out_modality_scale
                            old_sum += new_nuemrator[identifier]

                if self.args.compensation:
                    if self.args.share_scope == 'all':
                        coefficients[param_name] = {identifier: float(nuemrator / old_sum) for identifier, nuemrator in new_nuemrator.items()}
                    elif self.args.share_scope == 'modality':
                        compened_size = sum([size for id, size in updated_sizes.items() if self.clients[id].modality in self.modality or self.modality in self.clients[id].modality])
                        coefficients[param_name] = {identifier: float(nuemrator / compened_size) if compened_size!= 0 else 0 for identifier, nuemrator in new_nuemrator.items()}
                    elif self.args.share_scope == 'modality_exact':
                        if param_modality:
                            compened_size = sum([size for id, size in updated_sizes.items() if self.clients[id].modality == param_modality or param_modality in self.clients[identifier].modality])
                        else:
                            compened_size = sum([size for id, size in updated_sizes.items() if self.clients[id].modality in self.modality or self.modality in self.clients[id].modality])
                        coefficients[param_name] = {identifier: float(nuemrator / compened_size) if compened_size!= 0 else 0 for identifier, nuemrator in new_nuemrator.items()}
                else:
                    coefficients[param_name] = {identifier: float(nuemrator / sum(new_nuemrator.values())) if sum(new_nuemrator.values())!= 0 else 0 for identifier, nuemrator in new_nuemrator.items()}

        # accumulate weights
        for identifier in ids:
            local_layers_iterator = dict(self.clients[identifier].upload())
            coefficient = {param: coefficients[param][identifier] for param in coefficients.keys()}
            with torch.no_grad():
                for param in coefficients.keys():
                    if param not in local_layers_iterator.keys() or coefficient[param] == 0:
                        # iterator[param] = None
                        continue
                    final_sd[param] += ((local_layers_iterator[param] - final_sd[param]) * coefficient[param]).type(final_sd[param].dtype)
        
        self.global_model.load_state_dict(final_sd, strict=False)

        logger.info(f'[{self.args.algorithm.upper()}] [{self.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')
    
    def _empty_client_models(self):
        for client in self.clients:
            # del client.model
            client.model = None
            gc.collect()
            # client.model = None

    @torch.no_grad()
    def _central_evaluate(self, fedavg=False):
        # Unimodal eval
        for dataset in self.server_dataset.keys():
            # logger.info(f'[{self.args.algorithm.upper()}] [{dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...start to evaluate {dataset.upper()}!')

            if DATASET_2_MODALITY[dataset]== 'img+txt':
                self.global_model = self.global_models[dataset]
                self.evaluator.set_model(self.global_model)
                server_dataset = self.server_dataset[dataset]
                result = self.evaluator.evaluate(torch.utils.data.DataLoader(dataset=server_dataset, batch_size=self.args.eval_batch_size, shuffle=True, num_workers=4, persistent_workers=True), eval_batch_size=self.args.eval_batch_size)
                server_log_string = f'[{self.args.algorithm.upper()}] [{dataset.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SERVER] '

                res_dict = {}
                for metric in MM_METRICS:
                    server_log_string += f"| {dataset +'1k_i2t_'+metric}: {result['n_fold']['i2t'][metric]:.4f} "
                    res_dict.update({f"Result/Server {dataset} 1k_i2t_{metric.title()}": result['n_fold']['i2t'][metric]})
                for metric in MM_METRICS:
                    server_log_string += f"| {dataset +'1k_t2i_'+metric}: {result['n_fold']['t2i'][metric]:.4f} "
                    res_dict.update({f"Result/Server {dataset} 1k_t2i_{metric.title()}": result['n_fold']['t2i'][metric]})

                server_log_string += f"| {dataset} 1k_rsum: {result['n_fold']['t2i']['recall_1']+result['n_fold']['i2t']['recall_1']:.4f} "
                res_dict.update({f"Test/Server {dataset} 1k_r@1sum": result['n_fold']['t2i']['recall_1']+result['n_fold']['i2t']['recall_1']})
                
                for metric in MM_METRICS:
                    server_log_string += f"| {dataset +'5k_i2t_'+metric}: {result['i2t'][metric]:.4f} "
                    res_dict.update({f"Result/Server {dataset} 5k_i2t_{metric.title()}": result['i2t'][metric]})
                for metric in MM_METRICS:
                    server_log_string += f"| {dataset +'5k_t2i_'+metric}: {result['t2i'][metric]:.4f} "
                    res_dict.update({f"Result/Server {dataset} 5k_t2i_{metric.title()}": result['t2i'][metric]})

                server_log_string += f"| {dataset} 5k_rsum: {result['t2i']['recall_1'] + result['i2t']['recall_1']:.4f} "
                res_dict.update({f"Test/Server {dataset} 5k_r@1sum": result['t2i']['recall_1']+result['i2t']['recall_1']})

                server_log_string += f"| {dataset} rsum: {result['n_fold']['t2i']['recall_1']+result['n_fold']['i2t']['recall_1'] + result['t2i']['recall_1']+result['i2t']['recall_1']:.4f}"
                res_dict.update({f"Test/Server {dataset} r@1sum": result['n_fold']['t2i']['recall_1']+result['n_fold']['i2t']['recall_1'] + result['t2i']['recall_1']+result['i2t']['recall_1']})

                logger.info(server_log_string)
                self.writer.log(res_dict, self.round)

            else:
                self.global_model = self.global_models[dataset]
                mm = MetricManager(self.args.eval_metrics)
                self.global_model.eval()
                self.global_model.to(self.args.server_device)
                server_dataset = self.server_dataset[dataset]

                for inputs, targets in torch.utils.data.DataLoader(dataset=server_dataset, batch_size=self.args.B, shuffle=False, num_workers=4, persistent_workers=True):
                    inputs, targets = inputs.to(self.args.server_device), targets.to(self.args.server_device)

                    if DATASET_2_MODALITY[dataset] == 'img':
                        outputs = self.global_model([inputs, None])[0]
                    elif DATASET_2_MODALITY[dataset] == 'txt':
                        outputs = self.global_model([None, inputs])[1]
                    loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)

                    mm.track(loss.item(), outputs, targets)
                else:
                    self.global_model.to('cpu')
                    mm.aggregate(len(server_dataset))

                # log result
                result = mm.results
                server_log_string = f'[{self.args.algorithm.upper()}] [{dataset.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SERVER] '

                ## loss
                loss = result['loss']
                server_log_string += f'| loss: {loss:.4f} '
                
                ## metrics
                for metric, value in result['metrics'].items():
                    server_log_string += f'| {metric}: {value:.4f} '
                logger.info(server_log_string)

                # log TensorBoard
                self.writer.log({f'Loss/Server {dataset + "after" if not fedavg else ""} Loss': loss}, self.round)
                for name, value in result['metrics'].items():
                    self.writer.log({f'Test/Server {dataset + ("after" if not fedavg else "")} {name.title()}': value}, self.round)
                # else:
                #     self.writer.flush()
                self.results[self.round][f'server_evaluated_{dataset + ("after" if not fedavg else "")}'] = result
            
        # Multimodal eval




    # def copy_weights(self, source, target):
    #     sd = source.state_dict()
    #     td = target.state_dict()
    #     for k in td.keys():
    #         if self.param_strategy[k] != 'dataset' and k in sd.keys():
    #             td[k] = sd[k]
        
    #     target.load_state_dict(td)

    # def sync_server_model(self):
    #     for dataset in self.global_models.keys():
    #         if DATASET_2_MODALITY[dataset] == self.server_modality and dataset != self.server_dataset_name:
    #             self.copy_weights(self.global_models[dataset], self.global_models[self.server_dataset_name])

    # def sync_client_models(self):
    #     for dataset in self.global_models.keys():
    #         if dataset == self.server_dataset_name:
    #             continue
    #         self.copy_weights(self.global_models[self.server_dataset_name], self.global_models[dataset])

    def update(self):
        """Update the global model through federated learning.
        """
        #################
        # Client Update #
        #################
        selected_ids = self._sample_clients() # randomly select clients
        updated_sizes = self._request(selected_ids, eval=False, participated=True, retain_model=True, save_raw=False) # request update to selected clients
        # _ = self._request(selected_ids, eval=True, participated=True, retain_model=True, save_raw=False) # request evaluation to selected clients 
        
        if self.args.fedavg_eval:
            old_models = deepcopy(self.global_models)
            for i, dataset in enumerate(self.global_models.keys()):
                self.global_model = self.global_models[dataset]
                self.task = DATASET_2_TASK[dataset]
                self.modality = DATASET_2_MODALITY[dataset]
                self.dataset = dataset
                self.out_modality_scale = self.args.out_modality_scales[i]
                self._aggregate(selected_ids, updated_sizes, fedavg=True) # aggregate local updates
                self.global_models[dataset] = self.global_model

            self._central_evaluate(fedavg=True)
            self.global_models = old_models
        #################
        # Server Update #
        #################
        # for task in self.global_models.keys():
        #     for modality in self.global_models[task].keys():
        for i, dataset in enumerate(self.global_models.keys()):
            self.global_model = self.global_models[dataset]
            self.task = DATASET_2_TASK[dataset]
            self.modality = DATASET_2_MODALITY[dataset]
            self.dataset = dataset
            self.out_modality_scale = self.args.out_modality_scales[i]
            self._aggregate(selected_ids, updated_sizes) # aggregate local updates
            self.global_models[dataset] = self.global_model
        
        if self.args.with_aux:
            for dataset in self.global_models.keys():
                self.global_model = self.global_models[dataset]
                self.task = DATASET_2_TASK[dataset]
                self.modality = DATASET_2_MODALITY[dataset]
                self.dataset = dataset

                if self.modality == 'img+txt':
                    continue
                elif self.modality == 'img':
                    txt_dataset = [d for d in self.global_models.keys() if DATASET_2_MODALITY[d] == 'txt'][0]
                    aux_model = self.global_models[txt_dataset]
                    sd = aux_model.state_dict()
                    auxes = {}
                    for k in self.global_model.aux_params().keys():
                        auxes.update({k:sd[k.replace('aux_','').replace('blockses.0', 'blockses.1')]})
                    self.global_model.load_state_dict(auxes, strict=False)
                elif self.modality == 'txt':
                    img_dataset = [d for d in self.global_models.keys() if DATASET_2_MODALITY[d] == 'img'][0]
                    aux_model = self.global_models[img_dataset]
                    sd = aux_model.state_dict()
                    auxes = {}
                    for k in self.global_model.aux_params().keys():
                        auxes.update({k:sd[k.replace('aux_','').replace('blockses.1', 'blockses.0')]})
                    self.global_model.load_state_dict(auxes, strict=False)

        # self.sync_server_model()
        # self._train()
        # self.sync_client_models()

        if self.round % self.args.lr_decay_step == 0: # update learning rate
            self.curr_lr *= self.args.lr_decay

        # Empty client models
        self._empty_client_models()
        return selected_ids

    def evaluate(self, excluded_ids):
        """Evaluate the global model located at the server.
        """
        ##############
        # Evaluation #
        ##############
        if self.args.eval_type != 'global': # `local` or `both`: evaluate on selected clients' holdout set
            selected_ids = range(self.args.K)
            _ = self._request(selected_ids, eval=True, participated=False, retain_model=False, save_raw=self.round == self.args.R)
        if self.args.eval_type != 'local': # `global` or `both`: evaluate on the server's global holdout set 
            self._central_evaluate()

        # calculate generalization gap
        # if (not self.args.train_only) and (not self.args.eval_type == 'global'):
        #     gen_gap = dict()
        #     curr_res = self.results[self.round]
        #     for key in curr_res['clients_evaluated_out'].keys():
        #         for name in curr_res['clients_evaluated_out'][key].keys():
        #             if 'avg' in name:
        #                 gap = curr_res['clients_evaluated_out'][key][name] - curr_res['clients_evaluated_in'][key][name]
        #                 gen_gap[f'gen_gap_{key}'] = {name: gap}
        #                 self.writer.add_scalars(f'Generalization Gap ({key.title()})', gen_gap[f'gen_gap_{key}'], self.round)
        #                 self.writer.flush()
        #     else:
        #         self.results[self.round]['generalization_gap'] = dict(gen_gap)

    def finalize(self):
        """Save results.
        """
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Save results and the global model checkpoint!')
        with open(os.path.join(self.args.result_path, f'{self.args.exp_name}.json'), 'w', encoding='utf8') as result_file: # save results
            results = {key: value for key, value in self.results.items()}
            json.dump(results, result_file, indent=4)
        for dataset in self.global_models.keys():
            if not os.path.exists(os.path.join(self.args.result_path, f'{self.args.exp_name}')):
                os.makedirs(os.path.join(self.args.result_path, f'{self.args.exp_name}'))
            torch.save(self.global_models[dataset].state_dict(), os.path.join(self.args.result_path, f'{self.args.exp_name}', f'{dataset}.pt')) # save model checkpoint
        self.writer.finish()
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...finished federated learning!')
        if self.args.use_tb:
            input('[FINISH] ...press <Enter> to exit after tidying up your TensorBoard logging!')
