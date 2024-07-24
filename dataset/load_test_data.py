import numpy as np
import torch
from colorama import Fore
from Configurations import config
from Configurations.config import home_dir
from lib.pc_utils import refine_point_cloud, apply_mask, random_down_sampling, get_npy_norms, numpy_to_o3d, \
    closest_point
from visualiztion import view_npy_open3d
import open3d as o3d

sensory_pc_path = home_dir+'pc_tmp_data.npy'

def estimate_suction_direction(point_data,view=False,view_mask=None,score=None):
    pcd = get_npy_norms(point_data)
    npy_norms = np.asarray(pcd.normals)
    npy_norms[npy_norms[:, 2] < 0] = -npy_norms[npy_norms[:, 2] < 0]

    if view:

        torch_norms = torch.from_numpy(npy_norms)
        torch_norms[~view_mask.squeeze(), 0:3] *= 0.0

        score_mask = score > 0.7
        score_mask = score_mask.squeeze()
        suctionable_mask=view_mask.squeeze() & score_mask
        unsuctionable_mask=view_mask.squeeze() & ~score_mask

        masked_norm1=torch_norms[suctionable_mask, 0:3].numpy()

        torch_pc=torch.from_numpy(point_data)
        masked_pc1=torch_pc[suctionable_mask, 0:3].numpy()
        masked_pc2=torch_pc[unsuctionable_mask, 0:3].numpy()

        rest_pc=torch_pc[~view_mask.squeeze(), 0:3].numpy()

        masked_colors1=np.ones_like(masked_pc1)*[0., 0., 0.24]
        masked_colors2=np.ones_like(masked_pc2)*[0., 0., 0.24]

        rest_colors=np.ones_like(rest_pc)*[0.65, 0.65, 0.65]

        masked_pcd1=numpy_to_o3d(npy=masked_pc1, color=masked_colors1)
        masked_pcd1.normals = o3d.utility.Vector3dVector(masked_norm1)

        masked_pcd2=numpy_to_o3d(npy=masked_pc2, color=masked_colors2)


        rest_pcd=numpy_to_o3d(npy=rest_pc, color=rest_colors)

        # npy_norms = torch_norms.numpy()



        # pcd = numpy_to_o3d(npy=point_data, color=colors)


        # pcd.normals = o3d.utility.Vector3dVector(npy_norms)

        scene_list = []

        scene_list.append(masked_pcd1)
        scene_list.append(masked_pcd2)

        scene_list.append(rest_pcd)

        # scene_list.append(axis_pcd)
        # scene_list.append(axis_right_arm)
        o3d.visualization.draw_geometries(scene_list)

    return npy_norms

def get_real_data():
    point_data = np.load(sensory_pc_path) # (<191000, 3) shape is not constant
    # np.save(empty_bin, point_data)

    point_data=refine_point_cloud(point_data)

    np.save(sensory_pc_path, point_data)

    full_point_clouds = apply_mask(point_data)


    point_data_choice=random_down_sampling(full_point_clouds,config.num_points)

    down_sampled_point_clouds = point_data_choice[:, :3]
    # view_npy_open3d(new_point_data,view_coordinate=True)

    return down_sampled_point_clouds,full_point_clouds


def random_sampling_augmentation(center_point, point_data, number_of_points):
    i=0
    while True:
        i+=1
        point_data_=random_down_sampling(point_data,number_of_points)
        if i>10:
            print(Fore.RED,'Warning: Unable to find closest point, after down sampling   ',Fore.RESET )
            return point_data_, None
        index = closest_point(point_data_, center_point)
        if index:
            return point_data_, index


if __name__ == '__main__':


    # pc,mean=get_real_data(config)
    # pc=pc.squeeze()
    # pc=pc.cpu().detach().numpy()
    # print(pc.shape)
    #
    # pc=get_npy_norms(pc)
    # view_o3d(pc)
    point_data = np.load(sensory_pc_path)
    point_data=random_down_sampling(point_data,config.num_points)

    # view_npy_open3d(point_data, True)
    view_npy_open3d(point_data, True)
    


