import torch
from torch import nn


class BalancedBCELoss(nn.Module):
    '''
    Multi-class Focal Loss
    '''
    def __init__(self,pivot=0.5):
        super(BalancedBCELoss, self).__init__()
        self.pivot=pivot
        self.bce_loss = nn.BCELoss()

    def forward(self, input, target,positive_weight=1,negative_weight=1):
        positive_cls_mask = target > self.pivot

        positive_loss = self.bce_loss(input[positive_cls_mask], target[positive_cls_mask]) \
            if positive_cls_mask.sum()>0 else torch.tensor([0],device=input.device)

        negative_loss = self.bce_loss(input[~positive_cls_mask], target[~positive_cls_mask]) \
            if (~positive_cls_mask).sum()>0 else torch.tensor([0],device=input.device)

        return positive_loss*positive_weight+negative_loss*negative_weight