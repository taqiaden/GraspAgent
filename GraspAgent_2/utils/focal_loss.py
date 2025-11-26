import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification using logits.
    Args:
        alpha (float): Weight for positive class, e.g. 0.25.
                       Set to None to disable class balancing.
        gamma (float): Focusing parameter (typically 1–2).
        reduction (str): 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCE with logits, per-sample
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Convert logits → probabilities (for focal factor)
        probs = torch.sigmoid(logits)

        # p_t = p if y=1 else (1-p)
        pt = probs * targets + (1 - probs) * (1 - targets)

        # Focal factor
        focal_weight = (1 - pt).pow(self.gamma)

        # Class balancing
        if self.alpha is not None:
            alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = focal_weight * alpha_weight

        loss = focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
