import os

import torch
from torchvision import datasets

try:
    import ujson as json
except ImportError:
    import json

from PIL import Image
from pycocotools.coco import COCO

from torch.utils.data import Dataset
from glob import glob

import numpy as np

from torchvision import transforms
from functools import partial
from torch.utils import data
import operator

import logging
logger = logging.getLogger(__name__)

class CocoCaptionsCap(Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        ids (list, optional): list of target caption ids
        extra_annFile (string, optional): Path to extra json annotation file (for training)
        extra_ids (list, optional): list of extra target caption ids (for training)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        instance_annFile (str, optional): Path to instance annotation json (for PMRP computation)
    Example:
        .. code:: python
            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root='dir where images are',
                                    annFile='json annotation file',
                                    transform=transforms.ToTensor())
            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample
            print("Image Size: ", img.size())
            print(target)
        Output: ::
            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']
    """
    def __init__(self, root, annFile, ids=None,
                 extra_annFile=None, extra_ids=None,
                 transform=None, target_transform=None, tokenizer=None, max_length=40,
                 instance_annFile=None, client=-1):
        self.root = os.path.expanduser(root)
        if extra_annFile:
            self.coco = COCO()
            with open(annFile, 'r') as fin1, open(extra_annFile, 'r') as fin2:
                dataset = json.load(fin1)
                extra_dataset = json.load(fin2)
                if not isinstance(dataset, dict) or not isinstance(extra_dataset, dict):
                    raise TypeError('invalid type {} {}'.format(type(dataset),
                                                                type(extra_dataset)))
                if set(dataset.keys()) != set(extra_dataset.keys()):
                    raise KeyError('key mismatch {} != {}'.format(list(dataset.keys()),
                                                                  list(extra_dataset.keys())))
                for key in ['images', 'annotations']:
                    dataset[key].extend(extra_dataset[key])
            self.coco.dataset = dataset
            self.coco.createIndex()
        else:
            self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys()) if ids is None else list(ids)
        if extra_ids is not None:
            self.ids += list(extra_ids)
        self.ids = [int(id_) for id_ in self.ids]
        self.transform = transform
        self.target_transform = target_transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.all_image_ids = set([self.coco.loadAnns(annotation_id)[0]['image_id'] for annotation_id in self.ids])

        iid_to_cls = {}
        if instance_annFile:
            for ins_file in glob(instance_annFile + '/instances_*'):
                with open(ins_file) as fin:
                    instance_ann = json.load(fin)
                for ann in instance_ann['annotations']:
                    image_id = int(ann['image_id'])
                    code = iid_to_cls.get(image_id, [0] * 90)
                    code[int(ann['category_id']) - 1] = 1
                    iid_to_cls[image_id] = code

                seen_classes = {}
                new_iid_to_cls = {}
                idx = 0
                for k, v in iid_to_cls.items():
                    v = ''.join([str(s) for s in v])
                    if v in seen_classes:
                        new_iid_to_cls[k] = seen_classes[v]
                    else:
                        new_iid_to_cls[k] = idx
                        seen_classes[v] = idx
                        idx += 1
                iid_to_cls = new_iid_to_cls

                if self.all_image_ids - set(iid_to_cls.keys()):
                    # print(f'Found mismatched! {self.all_image_ids - set(iid_to_cls.keys())}')
                    print(f'Found mismatched! {len(self.all_image_ids - set(iid_to_cls.keys()))}')

        self.iid_to_cls = iid_to_cls
        self.n_images = len(self.all_image_ids)
    
    def reduce_samples(self, num_samples=1000):

        # sampled = np.random.choice(len(self), num_samples, replace=False)
        sampled = np.arange(-num_samples, 0)

        self.ids = list(operator.itemgetter(*sampled)(self.ids))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a caption for the annotation.
        """
        coco = self.coco
        annotation_id = self.ids[index]
        annotation = coco.loadAnns(annotation_id)[0]
        image_id = annotation['image_id']
        caption = annotation['caption']  # language caption

        path = coco.loadImgs(image_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.tokenizer is not None:
            target = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")['input_ids'][0]
        else:
            target = caption

        return img, target, image_id, annotation_id, index

    def __len__(self):
        return len(self.ids)
    
def txt_transform(max_length=40):
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case="uncased" in 'bert_base_uncased'
    )
    
    train_transform = partial(tokenizer, padding='max_length', max_length=max_length, truncation=True)

    return train_transform

def img_transform(img_size=32):

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([

        transforms.Resize((img_size, img_size)),
        # transforms.RandomCrop(img_size, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return train_transform

def fetch_coco(args, root, transforms, tokenizer, modality='img+txt'):
    logger.info('[LOAD] [COCO] Fetching dataset!')
    
    img_path = os.path.join(root, 'all_images')
    ann_path = os.path.join(root,'annotations','captions_train2014.json')
    ids = np.load(os.path.join(root, 'coco_train_ids.npy'))[:args.reduce_samples]
    # configure arguments for dataset
    dataset_args = {'root': img_path, 'annFile': ann_path,'transform': transforms[0], "tokenizer": tokenizer, "max_length": args.seq_len, 'ids': ids}

    # create dataset instance
    raw_train = CocoCaptionsCap(**dataset_args)
    # elif args.reduce_samples_seg_scale>0:
    #     raw_train._reduce_samples(int(len(raw_train) * args.reduce_samples_seg_scale))
    raw_train.task = 'img+txt'
    raw_train.modality = modality
    raw_train.name = 'Coco'


    test_args = dataset_args.copy()
    test_args['transform'] = transforms[1]
    ann_path = os.path.join(root,'annotations','captions_val2014.json')
    test_args['annFile'] = ann_path
    img_path = os.path.join(root, 'all_images')
    test_args['root'] = img_path
    test_args['ids'] = np.load(os.path.join(root, 'coco_test_ids.npy'))

    raw_test = CocoCaptionsCap(**test_args)

    raw_test.task = 'img+txt'
    raw_test.modality = modality
    raw_test.name = 'Coco'
    
    logger.info('[LOAD] [COCO] ...fetched dataset!')

    args.in_channels = 3
    args.num_classes = None
    
    return raw_train, raw_test, args
    