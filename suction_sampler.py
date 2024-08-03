import numpy as np
import torch
import open3d as o3d

from lib.pc_utils import get_npy_norms, numpy_to_o3d

def view_suction_direction(point_data,npy_norms,view_mask=None,score=None):
    torch_norms = torch.from_numpy(npy_norms)
    torch_norms[~view_mask.squeeze(), 0:3] *= 0.0

    score_mask = score > 0.7
    score_mask = score_mask.squeeze()
    suctionable_mask = view_mask.squeeze() & score_mask
    unsuctionable_mask = view_mask.squeeze() & ~score_mask

    masked_norm1 = torch_norms[suctionable_mask, 0:3].numpy()

    torch_pc = torch.from_numpy(point_data)
    masked_pc1 = torch_pc[suctionable_mask, 0:3].numpy()
    masked_pc2 = torch_pc[unsuctionable_mask, 0:3].numpy()

    rest_pc = torch_pc[~view_mask.squeeze(), 0:3].numpy()

    masked_colors1 = np.ones_like(masked_pc1) * [0., 0., 0.24]
    masked_colors2 = np.ones_like(masked_pc2) * [0., 0., 0.24]

    rest_colors = np.ones_like(rest_pc) * [0.65, 0.65, 0.65]

    masked_pcd1 = numpy_to_o3d(npy=masked_pc1, color=masked_colors1)
    masked_pcd1.normals = o3d.utility.Vector3dVector(masked_norm1)

    masked_pcd2 = numpy_to_o3d(npy=masked_pc2, color=masked_colors2)

    rest_pcd = numpy_to_o3d(npy=rest_pc, color=rest_colors)


    scene_list = []

    scene_list.append(masked_pcd1)
    scene_list.append(masked_pcd2)

    scene_list.append(rest_pcd)

    o3d.visualization.draw_geometries(scene_list)
def estimate_suction_direction(point_data,view=False,view_mask=None,score=None):
    pcd = get_npy_norms(point_data)
    npy_norms = np.asarray(pcd.normals)
    npy_norms[npy_norms[:, 2] < 0] = -npy_norms[npy_norms[:, 2] < 0]

    if view:
        view_suction_direction(point_data,npy_norms,view_mask,score)

    return npy_norms
