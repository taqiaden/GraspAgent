import torch


def contrastive_hinge_loss(first, second, margin=1.0, exponent=1.0,k=1):
    return torch.clamp((first - second)*k + margin, 0.) ** exponent
