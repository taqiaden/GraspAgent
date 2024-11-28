import torch
from torch import nn
from analytical_suction_sampler import estimate_suction_direction
from lib.depth_map import depth_to_point_clouds, transform_to_camera_frame
from registration import camera

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)

def suction_sampler_loss(depth,j,normals):
    '''generate labels'''
    pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
    pc = transform_to_camera_frame(pc, reverse=True)
    labels = estimate_suction_direction(pc, view=False)  # inference time on local computer = 1.3 s
    labels = torch.from_numpy(labels).to('cuda')
    '''mask prediction'''
    masked_prediction = normals[j][mask]
    '''view output'''
    # view_npy_open3d(pc,normals=normals)
    # normals=masked_prediction.detach().cpu().numpy()
    # view_npy_open3d(pc,normals=normals)
    return ((1 - cos(masked_prediction, labels.squeeze())) ** 2).mean()