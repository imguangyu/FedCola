import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, prediction, target):
        """
        :param prediction: predicted tensor by the network (after sigmoid/softmax activation)
                           with a shape of [batch_size, n_classes, height, width]
        :param target: ground truth tensor with the same shape as the prediction
        :return: Dice loss
        """

        # Flatten the tensors
        prediction = prediction.contiguous().view(prediction.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        intersection = (prediction * target).sum(dim=1)
        union = prediction.sum(dim=1) + target.sum(dim=1)

        dice_coefficient = (2. * intersection) / (union + self.eps)
        loss = 1 - dice_coefficient

        return loss.mean()
    

class SegLoss(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5, eps=1e-7):
        super(SegLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.dice_loss = DiceLoss(eps=eps)

    def forward(self, prediction, target):
        ce_loss = F.cross_entropy(prediction, target[:,0].long())
        # For the dice loss computation, we need the class probabilities, so apply softmax along the channel dimension
        dice_loss = self.dice_loss(F.softmax(prediction, dim=1), F.one_hot(target.long(), prediction.shape[1]).float())

        # logger.info(f'ce_loss: {ce_loss}, dice_loss: {dice_loss}')

        combined_loss = self.weight_ce * ce_loss + self.weight_dice * dice_loss
        return combined_loss