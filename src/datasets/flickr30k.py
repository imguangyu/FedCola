import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)

class Flickr30kCap(Dataset):
    def __init__(self, root, split='train', transform=None, tokenizer=None, max_length=40, train_all=False):
        self.root = root
        self.split = split
        self.transform = transform

        if train_all:
            split = 'train_all'

        anno = pd.read_csv(os.path.join(root, f'{split}.csv'), delimiter='|')

        self.images = anno['image_name']
        self.captions = [str(i) for i in anno[' comment'].values]

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.n_images = len(set(self.images))
        self.iid_to_cls = {}


    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(os.path.join(self.root, 'flickr30k_images', image_path)).convert('RGB')
        caption = self.captions[index]
        indice = index

        if self.transform is not None:
            image = self.transform(image)
        
        if self.tokenizer is not None:
            caption = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")['input_ids'][0]

        return image, caption, index // 5 , index, indice

    def __len__(self):
        return len(self.images)

def fetch_flickr30k(args, root, transforms, tokenizer, modality='img+txt'):
    logger.info('[LOAD] [FLICKR] Fetching dataset!')
    
    # configure arguments for dataset
    dataset_args = {'root': root, 'transform': transforms[0], "split": "train", "tokenizer": tokenizer, "max_length": args.seq_len, "train_all": args.flickr_train_all}

    # create dataset instance
    raw_train = Flickr30kCap(**dataset_args)
    # if args.reduce_samples >0 :
    #     raw_train._reduce_samples(args.reduce_samples)
    # elif args.reduce_samples_seg_scale>0:
    #     raw_train._reduce_samples(int(len(raw_train) * args.reduce_samples_seg_scale))
    raw_train.task = 'img+txt'
    raw_train.modality = modality
    raw_train.name = 'Flickr30k'


    test_args = dataset_args.copy()
    test_args['transform'] = transforms[1]
    test_args['split'] = 'test'
    test_args['train_all'] = False

    raw_test = Flickr30kCap(**test_args)
    # if args.reduce_test_samples >0 and args.reduce_test_samples < len(raw_test):
    #     raw_test._reduce_samples(args.reduce_samples)
    raw_test.task = 'img+txt'
    raw_test.modality = modality
    raw_test.name = 'Flickr30k'
    
    logger.info('[LOAD] [FLICKR] ...fetched dataset!')

    args.in_channels = 3
    args.num_classes = None
    
    return raw_train, raw_test, args