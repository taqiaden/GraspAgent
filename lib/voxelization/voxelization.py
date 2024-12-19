import numpy as np
import open3d as o3d
import torch

from lib.mask_utils import cubic_mask
from visualiztion import vis_depth_map


def voxelization_(pc,size,view=False):

    pcd = o3d.geometry.PointCloud()
    # Add the points, colors and normals as Vectors
    if torch.is_tensor(pc): pc=pc.cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(pc)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # pcd.normals = o3d.utility.Vector3dVector(normals)
    # Create a voxel grid from the point cloud with a voxel_size of 0.01
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=size)

    if view :
        vis = o3d.visualization.Visualizer()
        # Create a window, name it and scale it
        vis.create_window(window_name='Visualize', width=800, height=600)

        # Add the voxel grid to the visualizer
        vis.add_geometry(voxel_grid)

        # We run the visualizater
        vis.run()
        # Once the visualizer is closed destroy the window and clean up
        vis.destroy_window()

    if o3d.__version__=='0.15.2':
        voxels = voxel_grid.get_voxels()  # returns list of voxels # open3d 0.15.2
    else:
        voxels=voxel_grid.voxels # open3d 0.8.0
    return voxels
def voxelization(pc,scope,n_voxeles,view=False):
    size=scope/(n_voxeles-0.5)

    return voxelization_(pc,size,view)

def get_voxalized_depth_image(pc, scope, n_voxeles, view=False):
    voxels=voxelization(pc, scope, n_voxeles, view)
    depth_image = np.full((n_voxeles, n_voxeles), 0.0)
    for vx in voxels:
        idx = vx.grid_index
        depth_image[idx[0], idx[1]] = idx[2] #/ n_voxeles
    return depth_image
def get_voxalized_pc(pc, scope, n_voxeles, view=False):
    voxels=voxelization(pc, scope, n_voxeles, view)
    npy = np.full((n_voxeles, n_voxeles, n_voxeles), 0.0)
    for vx in voxels:
        idx = vx.grid_index
        npy[idx[0], idx[1], idx[2]] = 1
    return npy

def get_partial_depth_image(pc, scope, n_voxeles, view=False):
    voxels=voxelization(pc, scope, n_voxeles, view)
    npy = np.full((n_voxeles, n_voxeles, n_voxeles), 0.0)
    depth_image = np.full((n_voxeles, n_voxeles), 0.0)
    for vx in voxels:
        idx = vx.grid_index
        npy[idx[0], idx[1], idx[2]] = 1
        depth_image[idx[0], idx[1]] = idx[2] / n_voxeles

def get_local_depth_image(point_data,index,scope_=0.16,n_voxeles=100,view_cropped_pc=False,view_depth_map=False):
    mask = cubic_mask(point_data, point_data[index], scope_, False)
    sub_pc = point_data[mask]
    depth_image = get_voxalized_depth_image(sub_pc, scope=scope_, n_voxeles=n_voxeles, view=view_cropped_pc)
    if view_depth_map: vis_depth_map(depth_image, view_as_point_cloud=False)

    depth_image = torch.from_numpy(depth_image)
    depth_image = depth_image[None, None, :, :].to('cuda').float()
    return depth_image,mask
