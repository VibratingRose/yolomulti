from __future__ import division, print_function

import torch
import torch.nn.functional as F
from torch import nn

from yolo.utils.register import Loss


@Loss.register
class BCEloss:
    def __init__(self, weight=None, topk=-1):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.topk = topk

    def __call__(self, pred, target):
        bce = self.bce(pred, target)
        if self.topk != -1:
            bce = bce.flatten(2)
            bce = torch.topk(bce, self.topk, -1)[0]
        return bce.mean()


@Loss.register
class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, gamma=1.5, alpha=0.25, reduction='none'):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss()  # must be 
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


@Loss.register
class DiceLoss:
    def __init__(self, pow=1) -> None:
        self.smooth = 1.0
        self.pow = pow

    def __call__(self, pred, target):
        pred = pred.sigmoid()
        intersection = pred * target
        union = torch.pow(pred, self.pow) + torch.pow(target, self.pow)
        return 1 - (2. * intersection.sum() + self.smooth) / (union.sum() + self.smooth)

class SegLoss:
    def __init__(self, bce_weight=0.5, smooth=1.0, topk=-1, power=1):
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.topk = topk
        self.power = power
    
    def __call__(self, prediction, target, smooth=1.0, topk=-1, power=0):
        bce = self.bce(prediction, target)
        dice = self.dice_loss(prediction, target, smooth, power=power)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        return (1- self.bce_weight) * dice + self.bce_weight * bce

    def dice_loss(self, prediction, target, smooth=1.0, power=1.0):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""
        prediction = (prediction > 0.5).int()
        intersection = prediction * target
        union = prediction + target

        return 1 - (2. * intersection.sum() + smooth) / (union.sum() + smooth)

@Loss.register
class ComposeLoss(nn.Module):
    def __init__(self, weight=None, losses=None) -> None:
        super().__init__()
        assert len(weight) == len(losses)
        self.weight = weight
        self.losses = [Loss[l[0]](*l[1:]) for l in losses]

    def __call__(self, pred, targets):
        loss = 0.0
        for w, l in zip(self.weight, self.losses):
            loss += w * l(pred, targets)
        return loss


def threshold_predictions_v(predictions, thr=150):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 255
    return thresholded_preds


def threshold_predictions_p(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds