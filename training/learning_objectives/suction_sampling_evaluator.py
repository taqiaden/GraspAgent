import torch
from torch import nn
from analytical_suction_sampler import estimate_suction_direction


cos=nn.CosineSimilarity(dim=-1,eps=1e-6)

def suction_sampler_loss(pc,target_normal):

    labels = estimate_suction_direction(pc, view=False)  # inference time on local computer = 1.3 s
    labels = torch.from_numpy(labels).to('cuda')


    '''view output'''
    # view_npy_open3d(pc,normals=normals)
    # normals=masked_prediction.detach().cpu().numpy()
    # view_npy_open3d(pc,normals=normals)
    return (((1 - cos(target_normal, labels.squeeze())) ** 2)).mean()