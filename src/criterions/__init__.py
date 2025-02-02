from .segmentation_loss import *
from .probemb import MCSoftContrastiveLoss
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import ContrastiveLossWithTemperature
import torch

torch.nn.SegLoss = SegLoss
torch.nn.MCSoftContrastiveLoss = MCSoftContrastiveLoss
torch.nn.ContrastiveLoss = ContrastiveLossWithTemperature