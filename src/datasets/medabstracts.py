from torch.utils import data
import numpy as np
import pandas as pd
import os
import logging
import torch

logger = logging.getLogger(__name__)

class MedicalAbstracts(data.Dataset):

    def __init__(self, root, is_train=True, transform=None, dataidxs = None):
        # dataset = torchtext.datasets.AG_NEWS(root, split='train' if is_train else 'test')

        split = 'train' if is_train else 'test'
        df = pd.read_csv(os.path.join(root, 'medical_tc_{}.csv'.format(split)))
        
        self.targets = df['condition_label'].to_list()
        self.data = df['medical_abstract'].to_list()
        # print(type(dataset))
        # 
        self.targets = np.array(self.targets)
        self.targets -= min(self.targets)  # label: {0, 1, 2, 3  4} 5

        self.targets = np.array(self.targets) 

        self.transform = transform

        if not dataidxs:
            dataidxs = range(len(self.targets))

        self.data_idx = dataidxs

    def __getitem__(self, index):

        output = self.data[self.data_idx[index]]

        if self.transform is not None:
            output = self.transform(output)

            return torch.tensor(output['input_ids']).long(), torch.tensor(self.targets[self.data_idx[index]]).long()
        else:
            return output, torch.tensor(self.targets[self.data_idx[index]]).long()

    def __len__(self):
        return len(self.data_idx)
    
def fetch_medabstracts(args, root, transforms, modality='txt'):
    logger.info('[LOAD] [MedicalAbstracts] Fetching dataset!')
    
    # configure arguments for dataset
    dataset_args = {'root': root, 'transform': transforms[0], "is_train": True}

    # create dataset instance
    raw_train = MedicalAbstracts(**dataset_args)
    # if args.reduce_samples >0 :
    #     raw_train._reduce_samples(args.reduce_samples)
    # elif args.reduce_samples_seg_scale>0:
    #     raw_train._reduce_samples(int(len(raw_train) * args.reduce_samples_seg_scale))
    raw_train.task = 'cls'
    raw_train.modality = modality
    raw_train.name = 'MedicalAbstracts'


    test_args = dataset_args.copy()
    test_args['transform'] = transforms[1]
    test_args['is_train'] = False

    raw_test = MedicalAbstracts(**test_args)
    # if args.reduce_test_samples >0 and args.reduce_test_samples < len(raw_test):
    #     raw_test._reduce_samples(args.reduce_samples)
    raw_test.task = 'cls'
    raw_test.modality = modality
    raw_test.name = 'MedicalAbstracts'
    
    logger.info('[LOAD] [MedicalAbstracts] ...fetched dataset!')

    args.in_channels = None
    args.num_classes = 5 # Update it to a filtered number of classes
    
    return raw_train, raw_test, args