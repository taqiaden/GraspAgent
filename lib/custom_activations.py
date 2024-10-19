import torch
import torch.nn.functional as F
from torch import nn

def Mish(input):
    return input * torch.tanh(F.softplus(input))

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        output = input * F.sigmoid(input)
        return output
