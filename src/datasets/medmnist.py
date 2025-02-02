import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import medmnist
import logging
from medmnist import INFO

logger = logging.getLogger(__name__)

data_flag_dict = {
    'pat': 'pathmnist',
    'ct': 'organcmnist',
    'img': 'organcmnist',
    'mic': 'bloodmnist',
    'der': 'dermamnist',
}


def fetch_medmnist(args, root, transforms, modality='ct'):
    logger.info('[LOAD] [MedMNIST] Fetching dataset!')
    info = INFO[data_flag_dict[modality]]

    DataClass = getattr(medmnist, info['python_class'])
    
    
    # configure arguments for dataset
    dataset_args = {'root': root, 'transform': transforms[0], 'split': 'train', 'download': True}

    # create dataset instance
    raw_train = DataClass(**dataset_args)

    print(modality, len(raw_train))
    if args.reduce_samples >0 :
        raw_train.imgs = raw_train.imgs[:args.reduce_samples]
        raw_train.labels = raw_train.labels[:args.reduce_samples]
    elif args.reduce_samples_cls_scale>0:
        new_num = int(len(raw_train) * args.reduce_samples_cls_scale)
        raw_train.imgs = raw_train.imgs[:new_num]
        raw_train.labels = raw_train.labels[:new_num]

    print(modality, len(raw_train))

    raw_train.labels = raw_train.labels.squeeze(1)
    raw_train.targets = raw_train.labels
    raw_train.task = 'cls'
    raw_train.modality = modality
    raw_train.name = 'MedMNIST'



    test_args = dataset_args.copy()
    test_args['transform'] = transforms[1]
    test_args['split'] = 'test'

    raw_test = DataClass(**test_args)
    if args.reduce_test_samples >0 and args.reduce_test_samples < len(raw_test):
        raw_test.imgs = raw_test.imgs[:args.reduce_test_samples]
        raw_test.labels = raw_test.labels[:args.reduce_test_samples]
    raw_test.labels = raw_test.labels.squeeze(1)
    raw_test.targets = raw_test.labels
    raw_test.task = 'cls'
    raw_test.modality = modality
    raw_test.name = 'MedMNIST'
    
    logger.info('[LOAD] [MedMNIST] ...fetched dataset!')

    args.in_channels = info['n_channels']
    args.num_classes = len(info['label'])
    
    return raw_train, raw_test, args