import numpy as np
import torch

from visualiztion import view_npy_open3d


def sphere_mask(pc,center,radius,view=False):
    dist = np.linalg.norm(pc - center, axis=1)
    mask = dist < radius
    if view: view_npy_open3d(pc[mask])
    return mask

def cuboid_mask(pc,center,x,y,z,no_limit_for_z=True,view=False):
    if torch.is_tensor(pc):
        dist=torch.abs(pc-center)
    else:
        dist=np.abs(pc-center)
    mask_x = dist[:,0] < x
    mask_y = dist[:,1] < y
    mask_z = dist[:,2] < z
    mask=mask_x & mask_y if no_limit_for_z else mask_x & mask_y & mask_z
    if view:
        view_npy_open3d(pc[mask].cpu().numpy())
    return mask

def cubic_mask(pc,center,scope,no_limit_for_z=True,view=False):
    size=scope/2
    return cuboid_mask(pc,center,size,size,size,no_limit_for_z,view)