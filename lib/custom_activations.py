import torch
import torch.nn.functional as F
from torch import nn

class GripperGraspRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()

    def forward(self, output):
        '''approach normalization'''
        approach = output[:, 0:3, :]
        approach = self.tanh(approach)
        approach = F.normalize(approach, dim=1)

        '''beta normalization'''
        beta = output[:, 3:5, :]
        beta = self.tanh(beta)
        beta = F.normalize(beta, dim=1)

        dist = output[:, 5:6, :]
        dist = self.sigmoid(dist)
        width = output[:, 6:7, :]
        width = self.sigmoid(width)

        normalized_output = torch.cat([approach, beta, dist, width], dim=1)
        return normalized_output

class GripperGraspRegressor2(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, output,clip=False,sigmoid=False):
        '''approach normalization'''
        approach = output[:, 0:3]
        # approach = self.tanh(approach)
        approach = F.normalize(approach, dim=1)

        '''beta normalization'''
        beta = output[:, 3:5]
        # beta = self.tanh(beta)
        beta = F.normalize(beta, dim=1)

        dist = output[:, 5:6]

        width = output[:, 6:7]
        if sigmoid:
            dist=self.sigmoid(dist)
            width=self.sigmoid(width)

        if clip:
            dist= torch.clip(dist, 0.0, 1.)
            width= torch.clip(width, 0.0, 1.)

        normalized_output = torch.cat([approach, beta, dist, width], dim=1)
        return normalized_output

class GrowingCosineUnit(nn.Module):
    def __init__(self):
        super(GrowingCosineUnit, self).__init__()

    def forward(self, input):
        cos_=torch.cos(input)
        return input*cos_

class BiasedTanh(nn.Module):
    def __init__(self,weight=True,bias=True):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.5)) if bias else None
        self.k = nn.Parameter(torch.tensor(1.)) if weight else None

    def forward(self, x):
        if self.b is not None: x=torch.tanh(x) + self.b
        if self.k is not None: x= self.k*x
        return x
