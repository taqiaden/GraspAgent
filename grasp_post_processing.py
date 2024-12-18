import numpy as np
import trimesh
from Configurations import config
from Configurations.run_config import simulation_mode
from lib.ROS_communication import save_grasp_data, save_suction_data
from lib.bbox import convert_angles_to_transformation_form
from lib.grasp_utils import  get_pose_matrixes
from lib.gripper_exploration import local_exploration
from visualiztion import vis_scene, visualize_suction_pose

exploration_probabilty=0.0
view_masked_grasp_pose = False
grasp_data_path = config.home_dir + 'grasp_data_tmp.npy'
pre_grasp_data_path = config.home_dir + 'pre_grasp_data_tmp.npy'
suction_data_path = config.home_dir + 'suction_data_tmp.npy'
pre_suction_data_path = config.home_dir + 'pre_suction_data_tmp.npy'

def  gripper_grasp_processing(action_obj,point_clouds,poses,check_collision=True,isvis=False):
    target_point = point_clouds[index]
    relative_pose_5=poses[index]
    T_d, width, distance = convert_angles_to_transformation_form(relative_pose_5, target_point)

    activate_exploration=True if np.random.rand()<exploration_probabilty else False

    if check_collision:
        T_d,distance,width,collision_intensity = local_exploration(T_d, width, distance ,point_clouds, exploration_attempts=5,
                                             explore_if_collision=False, view_if_sucess=view_masked_grasp_pose,explore=activate_exploration)
        is_executable=collision_intensity==0

    else:
        is_executable=True
    if not is_executable:
        return False,width,T_d
    if simulation_mode==False:
        pre_grasp_mat, end_effecter_mat = get_pose_matrixes(T_d, k_end_effector=0.169, k_pre_grasp=0.23)
        save_grasp_data(end_effecter_mat, width, grasp_data_path)
        save_grasp_data(pre_grasp_mat, width, pre_grasp_data_path)
    if isvis: vis_scene(T_d,width,npy=point_clouds)
    return True, width, T_d


def gripper_shift_processing(action_obj, point_clouds, normals, isvis=False):
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

def suction_processing(action_obj,point_data,normals,isvis=False):
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