import torch


def contrastive_hinge_loss(superior, inferior, margin=.0, exponent=1.0):
    return torch.clamp(inferior - superior + margin, 0.) ** exponent