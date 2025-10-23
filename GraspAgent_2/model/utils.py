import torch
from torch import nn
from torch.nn.utils import spectral_norm


def add_spectral_norm_selective(model, layer_types=(nn.Conv3d,nn.Conv2d,nn.Conv1d, nn.Linear)):
    for name, layer in model.named_children():
        if isinstance(layer, layer_types):
            setattr(model, name, spectral_norm(layer, name='weight'))
        else:
            add_spectral_norm_selective(layer, layer_types)
    return model

def replace_activations(module, old_act, new_act):
    for name, child in module.named_children():
        if isinstance(child, old_act):
            setattr(module, name, new_act)
        else:
            replace_activations(child, old_act, new_act)

class sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(w0, dtype=torch.float32, device='cuda'), requires_grad=True)

    def forward(self,x):
        x=torch.sin(x*self.w0)
        return x