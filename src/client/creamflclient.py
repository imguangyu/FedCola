import gc
import operator
from .fedavgclient import FedavgClient

from src.datasets.coco import CocoCaptionsCap, img_transform, txt_transform
from torch.utils import data


from src import init_weights, TqdmToLogger, MetricManager

import torch
import torch.nn as nn

from copy import deepcopy

import logging
logger = logging.getLogger(__name__)


class CreamflClient(FedavgClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_pub_loader(self, dataset, batch_size=512):

        loader_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': False,
            'drop_last': False,
            # 'num_workers': 4,
            # 'persistent_workers': True,
        }

        pub_loader = data.DataLoader(**loader_kwargs)
        return pub_loader
    
    def update_pub_feature(self):
        
        dataloader = self.get_pub_loader(self.pub_dataset, batch_size=self.args.pub_batch_size)

        self.model.to(self.device)
        self.model.eval()
        feature = []
        distill_index = []
        # iterate batch
        for idx, (images, captions, _,_, index) in (enumerate(dataloader)):
            with torch.no_grad():
                if self.modality == 'img':
                    images = images.to(self.device)
                    im_feature = self.model([images, None], feat_out=True)[0]

                elif self.modality == 'txt':
                    captions = captions.to(self.device)
                    im_feature = self.model([None, captions], feat_out=True)[1]

                im_feature = im_feature.cpu().detach()
                feature.append(im_feature)
                distill_index.extend(index)
                # print(f'im_feature {im_feature.shape} labels {labels_var.shape}')
                # if is_test and idx == 1:
                #     break
        feature = torch.cat(feature, dim=0)
        # print(f'feature {feature.shape} labels {labels.shape}')
        self.model.to('cpu')

        self.pub_features = feature
        self.distill_index = distill_index



    
    def update(self):
        mm = MetricManager(self.eval_metrics) if self.modality!= 'img+txt' else MetricManager([])
        self.model.train()
        self.model.to(self.device)
        self.old_model = deepcopy(self.model)

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


            if self.args.interintra_weight > 0 and not (self.args.no_mm_contrastive and self.modality == 'img+txt'):
                dataloader = self.get_pub_loader(self.pub_dataset, batch_size=self.args.pub_batch_size)
                distill_dict = {int(b): a for a, b in enumerate(self.distill_index)}


                criterion = nn.CrossEntropyLoss()

                self.old_model.to(self.device)
                self.old_model.eval()
                self.model.to(self.device)
                self.model.train()

                self.global_img_feature = self.global_img_feature.to(self.device)
                self.global_txt_feature = self.global_txt_feature.to(self.device)
                
                for idx, (images, captions, _, _, index) in TqdmToLogger(
                    (enumerate(dataloader)), 
                    logger=logger, 
                    desc=f'[{self.task.upper()}] [{self.modality.upper()}] ...get contrastive on loss client {self.id}... ',
                    total=len(dataloader)
                    ):
                    optimizer.zero_grad()
                    d_idx = operator.itemgetter(*index.tolist())(distill_dict)  # batchidx
                    if self.modality != 'img+txt':
                        if  'img' == self.modality:
                            images = images.to(self.device)
                            im_feature = self.model([images, None], feat_out=True)[0]
                            target_feature = self.global_img_feature[d_idx, :].type_as(im_feature)
                            
                            with torch.no_grad():
                                old_im_feature = self.old_model([images, None], feat_out=True)[0]

                            logits_inter = torch.div(torch.matmul(im_feature, self.global_txt_feature.T), 0.5)
                        elif self.modality == 'txt':
                            captions = captions.to(self.device)
                            im_feature = self.model([None, captions], feat_out=True)[1]
                            target_feature = self.global_txt_feature[d_idx, :].type_as(im_feature)
                            # neg
                            with torch.no_grad():
                                old_im_feature = self.old_model([None, captions], feat_out=True)[1]
                                
                            logits_inter = torch.div(torch.matmul(im_feature, self.global_img_feature.T), 0.5)

                        labels_inter = torch.tensor(d_idx).to(self.device)
                        loss_inter = criterion(logits_inter, labels_inter)

                        # pos
                        pos = torch.sum(im_feature * target_feature, dim=-1)
                        pos = pos.reshape(-1, 1)
                        # neg
                        # neg = cos(im_feature, old_im_feature)
                        neg = torch.sum(im_feature * old_im_feature, dim=-1)
                        logits = torch.cat((pos, neg.reshape(-1, 1)), dim=1)

                        logits /= 0.5  # temperature
                        labels = torch.zeros(images.size(0)).to(self.device).long()

                        loss_moon = criterion(logits, labels)

                        loss = (loss_moon + loss_inter) * self.args.interintra_weight
                    else:
                        if self.args.no_mm_contrastive:
                            continue
                        images = images.to(self.device)
                        captions = captions.to(self.device)

                        outs = self.model([images, captions], feat_out=True)
                        out_img = outs[0]
                        out_txt = outs[1]

                        target_img = self.global_img_feature[d_idx, :].type_as(out_img)
                        target_txt = self.global_txt_feature[d_idx, :].type_as(out_txt)

                        # pos
                        pos_img = torch.sum(out_img * target_img, dim=-1)
                        pos_img = pos_img.reshape(-1, 1)
                        pos_txt = torch.sum(out_txt * target_txt, dim=-1)
                        pos_txt = pos_txt.reshape(-1, 1)
                        # neg
                        with torch.no_grad():
                            old_outs = self.old_model([images, captions], feat_out=True)
                            old_out_img = old_outs[0]
                            old_out_txt = old_outs[1]
                        
                        neg_img = torch.sum(out_img * old_out_img, dim=-1)
                        neg_txt = torch.sum(out_txt * old_out_txt, dim=-1)
                        logits_1 = torch.cat((pos_img, neg_img.reshape(-1, 1)), dim=1)
                        logits_2 = torch.cat((pos_txt, neg_txt.reshape(-1, 1)), dim=1)
                        logits = torch.cat((logits_1, logits_2), dim=0)

                        logits /= 0.5  # temperature
                        labels = torch.zeros(images.size(0) * 2).to(self.device).long()

                        loss_intra = criterion(logits, labels)

                        logits_1_inter = torch.div(torch.matmul(out_img, self.global_txt_feature.T), 0.5)
                        logits_2_inter = torch.div(torch.matmul(out_txt, self.global_img_feature.T), 0.5)

                        labels_inter = torch.tensor(d_idx).to(self.device)

                        loss_inter = criterion(logits_1_inter, labels_inter) + criterion(logits_2_inter, labels_inter)

                        loss = (loss_intra + loss_inter) * self.args.interintra_weight

                    loss.backward()
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 2)
                    optimizer.step()

        if self.args.distributed:
            self.model = self.model.module
        self.model.to('cpu')

        del self.old_model
        gc.collect()
        torch.cuda.empty_cache()
            
        return mm.results
