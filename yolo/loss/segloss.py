from __future__ import print_function, division
from torch import nn
import torch.nn.functional as F


class SegLoss:
    def __init__(self, bce_weight=0.5):
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
    
    def __call__(self, prediction, target):
        bce = self.bce(prediction, target)
        dice = self.dice_loss(prediction.sigmoid(), target)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        return (1- self.bce_weight) * dice + self.bce_weight * bce

    def dice_loss(self, prediction, target):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""
        smooth = 1.0
        prediction = (prediction > 0.5).int()
        intersection = prediction * target
        union = prediction + target

        return 1 - (2. * intersection.sum() + smooth) / (union.sum() + smooth)


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