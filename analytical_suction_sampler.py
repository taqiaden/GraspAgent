import numpy as np
import open3d as o3d
import torch

from lib.pc_utils import get_npy_norms, numpy_to_o3d


def masked_pcd(pc,mask,normal=None):
    masked_pc=pc[mask]
    masked_colors = np.ones_like(masked_pc) * [0.52, 0.8, 0.92]
    pcd = numpy_to_o3d(pc=masked_pc, color=masked_colors)
    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal[mask])
    return pcd


def view_suction_direction(point_data,npy_norms,view_mask=None):
    torch_norms = torch.from_numpy(npy_norms)
    # torch_norms[~view_mask.squeeze(), 0:3] *= 0.0
    '''with normals'''
    pcd_A=masked_pcd(point_data,view_mask,normal=torch_norms)
    pcd_B=masked_pcd(point_data,~view_mask)

    # colors = np.ones_like(point_data) * [0.52, 0.8, 0.92]
    # pcd = numpy_to_o3d(pc=point_data, color=colors)
    # pcd.normals = o3d.utility.Vector3dVector(torch_norms)
    scene_list = []
    scene_list.append(pcd_A)
    scene_list.append(pcd_B)

    o3d.visualization.draw_geometries(scene_list)

def estimate_suction_direction(point_data,view=False,view_mask=None):
    pcd = get_npy_norms(point_data)
    npy_norms = np.asarray(pcd.normals)
    npy_norms[npy_norms[:, 2] < 0] = -npy_norms[npy_norms[:, 2] < 0]

    if view:
        view_suction_direction(point_data,npy_norms,view_mask)

    return npy_norms
