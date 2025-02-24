import numpy as np
import open3d as o3d
import torch

from lib.pc_utils import get_npy_norms, numpy_to_o3d
from visualiztion import view_npy_open3d, view_o3d


def masked_pcd(pc,mask,normal=None):
    masked_pc=pc[mask]
    masked_colors = np.ones_like(masked_pc) * [0.52, 0.8, 0.92]
    pcd = numpy_to_o3d(pc=masked_pc, color=masked_colors)
    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal[mask])
    return pcd



def estimate_suction_direction(point_data,view=False,radius=0.1, max_nn=30):
    pcd = get_npy_norms(point_data,radius=radius,max_nn=max_nn)
    npy_norms = np.asarray(pcd.normals)
    npy_norms[npy_norms[:, 2] < 0] = -npy_norms[npy_norms[:, 2] < 0]

    if view:
        view_o3d(pcd)
    return npy_norms
