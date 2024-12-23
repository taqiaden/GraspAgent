import numpy as np
import trimesh

from Configurations import config
from lib.ROS_communication import  save_suction_data
from lib.grasp_utils import get_pose_matrixes
from visualiztion import vis_scene, visualize_suction_pose

exploration_probabilty=0.0
view_masked_grasp_pose = False
grasp_data_path = config.home_dir + 'grasp_data_tmp.npy'
pre_grasp_data_path = config.home_dir + 'pre_grasp_data_tmp.npy'
suction_data_path = config.home_dir + 'suction_data_tmp.npy'
pre_suction_data_path = config.home_dir + 'pre_suction_data_tmp.npy'







def gripper_shift_processing(index, point_clouds, normals, isvis=False):
    normal=normals[index]
    target_point=point_clouds[index]

    v0 = np.array([1, 0, 0])
    a = trimesh.transformations.angle_between_vectors(v0, -normal)
    b = trimesh.transformations.vector_product(v0, -normal)
    T_d = trimesh.transformations.rotation_matrix(a, b)
    T_d[:3, 3] = target_point.T

    width=np.array([0])


    if isvis: vis_scene(T_d, width, npy=point_clouds)

    return True,  width, T_d

def suction_processing(index,point_data,normals,isvis=False):
    normal=normals[index]
    target_point=point_data[index]

    v0 = np.array([1, 0, 0])
    a = trimesh.transformations.angle_between_vectors(v0, -normal)
    b = trimesh.transformations.vector_product(v0, -normal)
    T = trimesh.transformations.rotation_matrix(a, b)
    T[:3, 3] = target_point.T

    pre_grasp_mat, end_effecter_mat = get_pose_matrixes(T, k_end_effector=0.184, k_pre_grasp=0.25)

    save_suction_data(end_effecter_mat, suction_data_path)
    save_suction_data(pre_grasp_mat, pre_suction_data_path)

    if isvis: visualize_suction_pose(target_point, normal.reshape(1, 3) , T, end_effecter_mat,npy=point_data)

    return True,T