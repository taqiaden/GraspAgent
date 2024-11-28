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
        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()

    def forward(self, output):
        '''approach normalization'''
        approach = output[:, 0:3]
        # approach = self.tanh(approach)
        approach = F.normalize(approach, dim=1)

        '''beta normalization'''
        beta = output[:, 3:5]
        # beta = self.tanh(beta)
        beta = F.normalize(beta, dim=1)

        dist = output[:, 5:6]
        # dist = self.sigmoid(dist)
        width = output[:, 6:7]
        # width = self.sigmoid(width)

        normalized_output = torch.cat([approach, beta, dist, width], dim=1)
        return normalized_output
