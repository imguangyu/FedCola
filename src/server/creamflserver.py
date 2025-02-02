import gc
from importlib import import_module
import operator
import os
from .fedavgserver import FedavgServer
from .fedavgserver import DATASET_2_MODALITY, NUM_CLASS, DATASET_2_TASK, TASK_2_CRITERION, VOCAB_SIZES
from src.datasets.coco import CocoCaptionsCap, img_transform, txt_transform

import torch
import torch.nn as nn

from torch.utils import data
import logging
import concurrent.futures

from collections import ChainMap, defaultdict
from copy import deepcopy

from src import init_weights, TqdmToLogger, MetricManager

import timm
import numpy as np

logger = logging.getLogger(__name__)


class CreamflServer(FedavgServer):

    def __init__(self, args, writer, server_dataset, client_datasets, model_str):
        
        self.pub_loader = self.get_pub_loader(root=args.pub_data_dir, anno_path=args.pub_anno_path, num_pub_samples=args.num_pub_samples, img_size=args.resize, max_length=args.seq_len, batch_size=args.pub_batch_size, dataset_only=False)
        self.pub_dataset = self.get_pub_loader(args.pub_data_dir, args.pub_anno_path, num_pub_samples=args.num_pub_samples, img_size=args.resize, max_length=args.seq_len, batch_size=args.pub_batch_size, dataset_only=True)

        super().__init__(args, writer, server_dataset, client_datasets, model_str)

        self.device = args.server_device 


    def _create_clients(self, client_datasets):
        CLINET_CLASS = import_module(f'..client.{self.args.algorithm}client', package=__package__).__dict__[f'{self.args.algorithm.title()}Client']

        def __create_client(identifier, datasets):
            client = CLINET_CLASS(args=self.args, training_set=datasets[0], test_set=datasets[1],task=datasets[2],modality=datasets[3], eval_metrics=['acc1'] if datasets[2]== 'cls' else ['f1'], 
                                  criterion=TASK_2_CRITERION[datasets[2]],
                                  writer=self.writer)
            client.id = identifier
            client.dataset = datasets[4]
            client.device = 'cuda:%d' % (identifier % torch.cuda.device_count()) if torch.cuda.is_available() else 'cpu'
            client.pub_dataset = deepcopy(self.pub_dataset)
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
    
    def _init_model(self, model_str):
        self.args.datasets = self.args.datasets[:-1]
        models = {}
        for i, dataset in enumerate(self.args.datasets): # CIFAR, AGNEWS, Flickr
            # try:
                # self.args.vocab_size = VOCAB_SIZES[dataset] if dataset in VOCAB_SIZES else 30522
                if DATASET_2_MODALITY[dataset] == 'img':
                    models[dataset] = timm.create_model(model_str, pretrained=self.args.pretrained, num_classes=[NUM_CLASS[dataset], None], modalities=[self.args.modalities[i], None], args=self.args, tasks=[DATASET_2_TASK[dataset], None], with_aux=self.args.with_aux, aux_trained=self.args.aux_trained)
                elif DATASET_2_MODALITY[dataset] == 'txt':
                    models[dataset] = timm.create_model(model_str, pretrained=self.args.pretrained, num_classes=[None, NUM_CLASS[dataset]], modalities=[None, self.args.modalities[i]], args=self.args, tasks=[None, DATASET_2_TASK[dataset]], with_aux=self.args.with_aux, aux_trained=self.args.aux_trained)
                elif DATASET_2_MODALITY[dataset] == 'img+txt':
                    models[dataset] = timm.create_model(model_str, pretrained=self.args.pretrained, num_classes=[None, None], modalities=['img', 'txt'], args=self.args, tasks=[DATASET_2_TASK[dataset], DATASET_2_TASK[dataset]], with_aux=self.args.with_aux, aux_trained=self.args.aux_trained)
                    # base_model = models[dataset]
            # except:
                # models[dataset] = timm.create_model(model_str, pretrained=self.args.pretrained, num_classes=NUM_CLASS[dataset])
        # for i, dataset in enumerate(self.args.datasets): # CIFAR, AGNEWS, Flickr
        #     # try:
        #         if DATASET_2_MODALITY[dataset] == 'img':
        #             models[dataset].load_state_dict(base_model.state_dict(), strict=False)
        #         elif DATASET_2_MODALITY[dataset] == 'txt':
        #             models[dataset].load_state_dict(base_model.state_dict(), strict=False)

        return models
    
    
    
    def get_pub_loader(self, root, anno_path, num_pub_samples=1000, img_size=32, max_length=40, batch_size=512, dataset_only=False):

        v_transform = img_transform(img_size=img_size)
        l_transform = txt_transform(max_length=max_length)


        parent_path = os.sep.join(anno_path.split('/')[:-2])
        ids = np.load(os.path.join(parent_path, 'coco_train_ids.npy'))[-num_pub_samples:]

        dataset = CocoCaptionsCap(root, anno_path, transform=v_transform, tokenizer=l_transform, max_length=max_length, ids=ids)

        if dataset_only:
            return dataset

        # dl = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8, persistent_workers=True, collate_fn=public_collate_fn)

        loader_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': False,
            'drop_last': False,
            'num_workers': 4,
            'persistent_workers': True,
        }

        pub_loader = data.DataLoader(**loader_kwargs)
        return pub_loader
    
    def _generate_public_logit(self):

        img_feature, txt_feature = [], []
        distill_index = []

        for dataset in self.global_models.keys():
            if DATASET_2_MODALITY[dataset] == 'img+txt':
                model = self.global_models[dataset]

        model.eval()
        model.to(self.device)

        for idx, (images, captions, image_id, annotation_id, index) in enumerate(self.pub_loader):
            with torch.no_grad():
                images = images.to(self.device)  # [bs, 3, 224, 224]
                captions = captions.to(self.device)  # [bs, seq_len]
                # output = self.model(images, captions, , capt_lens)
                outs = model([images, captions])
                out_img = outs[0]
                out_txt = outs[1]
                out_img = out_img.cpu().detach()
                out_txt = out_txt.cpu().detach()

                img_feature.append(out_img)
                txt_feature.append(out_txt)
                distill_index.extend(index)

        self.global_img_feature = torch.concat(img_feature, dim=0)
        self.global_txt_feature = torch.concat(txt_feature, dim=0)
        self.distill_index = distill_index

        
        del img_feature, txt_feature
        gc.collect()
        torch.cuda.empty_cache()
    
    def _request(self, ids, eval, participated, retain_model, save_raw):
        def __update_clients(client):
            if client.model is None:
                client.download(self.global_models)
            client.args.lr = self.curr_lr

            # if client.modality == 'img':
            client.global_img_feature = self.global_img_feature.to(client.device)
            # elif client.modality == 'txt':
            client.global_txt_feature = self.global_txt_feature.to(client.device)

            client.distill_index = self.distill_index

            update_result = client.update()
            logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...updated client {client.id}!')
            if client.modality != 'img+txt':
                client.update_pub_feature()
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
    

    def _aggregate(self, ids, updated_sizes):
        assert set(updated_sizes.keys()) == set(ids)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')
        
        # calculate mixing coefficients according to sample sizes
        # coefficients = {identifier: float(nuemrator / sum(updated_sizes.values())) for identifier, nuemrator in updated_sizes.items()}
        final_sd = self.global_model.cpu().state_dict()

        for k,v in final_sd.items():
            final_sd[k] = torch.zeros_like(v).float()

        coefficients = {}
        for param_name in self.global_model.state_dict().keys():
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
            if self.args.compensation:
                coefficients[param_name] = {identifier: float(nuemrator / old_sum) for identifier, nuemrator in new_nuemrator.items()}
            else:
                coefficients[param_name] = {identifier: float(nuemrator / sum(new_nuemrator.values())) for identifier, nuemrator in new_nuemrator.items()}

        # accumulate weights
        for identifier in ids:
            local_layers_iterator = dict(self.clients[identifier].upload())
            coefficient = {param: coefficients[param][identifier] for param in coefficients.keys()}
            with torch.no_grad():
                for param in coefficients.keys():
                    if param not in local_layers_iterator.keys() or coefficient[param] == 0:
                        # iterator[param] = None
                        continue
                    final_sd[param] += local_layers_iterator[param] * coefficient[param]
        self.global_model.load_state_dict(final_sd)

        logger.info(f'[{self.args.algorithm.upper()}] [{self.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')
        logger.info(f'[{self.args.algorithm.upper()}] [{self.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...now doing distillation!')


        distill_dict = {int(b): a for a, b in enumerate(self.distill_index)} 

        self.global_model.train()
        self.global_model.to(self.device)

        client_loss_cri = nn.MSELoss()

        self.optimizer = torch.optim.AdamW(self.global_model.parameters(), lr=self.args.p_lr)

        # distill
        for idx, (images, captions, _, _, index) in enumerate(self.pub_loader):
            images = images.to(self.device)  # [bs, 3, 224, 224]
            captions = captions.to(self.device)  # [bs, seq_len]
            # caption_lens = caption_lens.to(self.device)

            # output = self.model(images, captions, txt_mask, index)
            loss = 0

            def code_sim(output, target):
                output = output.sum(axis=1) if len(output.shape) == 3 else output
                target = target.type_as(output)

                return client_loss_cri(output, target.type_as(output))

            outs = self.global_model([images, captions])
            out_img = outs[0]
            out_txt = outs[1]

            d_idx = operator.itemgetter(*index.tolist())(distill_dict)  # idx of the current batch
            target_img = self.img_vec[d_idx, :].type_as(out_img)
            target_txt = self.txt_vec[d_idx, :].type_as(out_txt)

            loss += self.args.kd_weight * (code_sim(out_img, target_img) + code_sim(out_txt, target_txt))

            self.optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad.clip_grad_norm_(self.global_model.parameters(), 2)

            self.optimizer.step()
        
    def update(self):
        """Update the global model through federated learning.
        """
        #################
        # Client Update #
        #################
        self._generate_public_logit()
        selected_ids = self._sample_clients() # randomly select clients
        updated_sizes = self._request(selected_ids, eval=False, participated=True, retain_model=True, save_raw=False) # request update to selected clients
        # _ = self._request(selected_ids, eval=True, participated=True, retain_model=True, save_raw=False) # request evaluation to selected clients 
        
        #################
        # Server Update #
        #################
        # for task in self.global_models.keys():
        #     for modality in self.global_models[task].keys():

        img_vec, img_num = [], []
        txt_vec, txt_num = [], []

        for id in selected_ids:
            if self.clients[id].modality == 'img':
                img_vec.append(self.clients[id].pub_features.to(self.device))
                img_num.append(updated_sizes[id])
                # print(f'img_vec {_vec["img"].shape}')

            elif self.clients[id].modality == 'txt':
                txt_vec.append(self.clients[id].pub_features.to(self.device))
                txt_num.append(updated_sizes[id])
                # print(f'txt_vec {_vec["txt"].shape}')

        def aggregation(i_vec=img_vec, t_vec=txt_vec, i_num=img_num, t_num=txt_num):
            if len(i_vec) > 0: 
                contrastive_w = []
                for vec in i_vec:  # vec: [50000, n_feature], global_txt_feature: [50000, n_feature]
                    logits = torch.matmul(vec, self.global_txt_feature.to(vec.device).T)  # [50000, 50000]
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
                    contrastive_w.append(torch.diagonal(log_prob).reshape(1, -1))
                contrastive_w = torch.softmax(torch.cat(contrastive_w, dim=0), dim=0)
                for i in range(len(i_vec)):
                    i_vec[i] = (i_vec[i] * contrastive_w[i].reshape(-1, 1)).unsqueeze(0)
                i_vec = torch.sum(torch.cat(i_vec, dim=0), dim=0)  # aggregated image vectors
            else:
                i_vec = []

            if len(t_vec) > 0: 
                contrastive_w = []
                for vec in t_vec:  # vec: [50000, n_feature], global_txt_feature: [50000, n_feature]
                    logits = torch.matmul(vec, self.global_img_feature.to(vec.device).T)  # [50000, 50000]
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
                    contrastive_w.append(torch.diagonal(log_prob).reshape(1, -1))
                contrastive_w = torch.softmax(torch.cat(contrastive_w, dim=0), dim=0)
                for i in range(len(t_vec)):
                    t_vec[i] = (t_vec[i] * contrastive_w[i].reshape(-1, 1)).unsqueeze(0)
                t_vec = torch.sum(torch.cat(t_vec, dim=0), dim=0)  # aggregated text vectors

            else:
                t_vec= None
            
            self.global_img_feature.cpu().detach()
            self.global_txt_feature.cpu().detach()

            return i_vec, t_vec
        
        img_vec, txt_vec = aggregation()

        self.img_vec = img_vec
        self.txt_vec = txt_vec

        for dataset in self.global_models.keys():
            if DATASET_2_MODALITY[dataset] == 'img+txt':
                self.global_model = self.global_models[dataset]
                self.task = DATASET_2_TASK[dataset]
                self.modality = DATASET_2_MODALITY[dataset]
                self.dataset = dataset
                self._aggregate(selected_ids, updated_sizes) # aggregate local updates
                self.global_models[dataset] = self.global_model
                base_model = self.global_models[dataset]
            else:
                self.global_model = self.global_models[dataset]
                self.task = DATASET_2_TASK[dataset]
                self.modality = DATASET_2_MODALITY[dataset]
                self.dataset = dataset
                super()._aggregate(selected_ids, updated_sizes, fedavg=True)
        
        if self.round % self.args.lr_decay_step == 0: # update learning rate
            self.curr_lr *= self.args.lr_decay
        
        del img_vec
        del txt_vec
        gc.collect()

        
        # Empty client models
        self._empty_client_models()
        return selected_ids