# [[ECCV2024](https://arxiv.org/abs/2404.12467)] Towards Multi-modal Transformers in Federated Learning

Official repository for Towards Multi-modal Transformers in Federated Learning (ECCV2024). Code will be released soon.

# Citation

```
@inproceedings{sun2024towards,
  title={Towards Multi-modal Transformers in Federated Learning},
  author={Sun, Guangyu and Mendieta, Matias and Dutta, Aritra and Li, Xin and Chen, Chen},
  booktitle={European Conference on Computer Vision},
  pages={229--246},
  year={2024},
  organization={Springer}
}

@article{sun2024towards,
  title={Towards Multi-modal Transformers in Federated Learning},
  author={Sun, Guangyu and Mendieta, Matias and Dutta, Aritra and Li, Xin and Chen, Chen},
  journal={arXiv preprint arXiv:2404.12467},
  year={2024}
}
```

# Get Started

## Environment

Python version: 3.8.0
```
pip install -r requirements.txt
```
## Prepare Data

Option 1: Directly download the entire `data` folder from [google drive](https://drive.google.com/file/d/1MhOE4q2P_D3Y5muyz-fhN6GnVSTgbK16/view?usp=sharing)

Option 2: 

Download Flickr30k dataset and put all images into `data/flickr30k/flickr30k_images`.

Download MS-COCO 2014 and put all images and annotations into `data/coco/all_images` and `data/coco/annotations`

## Wandb for Logging
Set up `wandb.init()` with your own project name and entity.

# Experiments

Please use scripts under `scirpts` to run experiments with the methods and settings to reproduce the results in our paper.

# Model Explaination

We unify the img and text encoders into one model `ModalityAgnosticTransformer` for easier aggregation:

`shared_param`: Shared parameters between same modality in different type of client (i.e., img encoder in img client and img encoder in img-txt client)

`share_scope`: Shared scope during aggregation
              dataset: share parameters only to encoders with the same dataset
              modality: share parameters only to encoders with the same modality
              all: share parameters among all encoders

`colearn_param`: Shared parameters between img and txt encoders

# Method Configurations

To correctly configurate each method, please follow this table:

|     Name | shared_param | share_scope      | algorithm |  Others|
|----------|--------------|------------------|-----------|--------|
| FedAVG   | none         | dataset          | fedavg    |
| FedIoT   | blocks       | modality_exact   | fediot    |
| FedProx  | none         | dataset          | fedprox   |
| CreamFL  | none         | dataset          | creamfl   |
| **FedCola (ours)**  | attn         | modality         | fedavg    | --aux --aux_trained


# Acknowledgement

This codebase is based on [Federated Learning in PyTorch](https://github.com/vaseline555/Federated-Learning-in-PyTorch). We extend it to our multi-modal federated learning setting.

For local complementary training, we adapted code from [here](https://github.com/AILab-CVC/M2PT) to add aux weights from the other modality.

