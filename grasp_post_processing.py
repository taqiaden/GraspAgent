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


def get_suction_pose_( target_point, normal):
    v0 = np.array([1, 0, 0])
    a = trimesh.transformations.angle_between_vectors(v0, -normal)
    b = trimesh.transformations.vector_product(v0, -normal)
    matrix_ori = trimesh.transformations.rotation_matrix(a, b)
    matrix_ori[:3, 3] = target_point.T
    T = matrix_ori
    pre_grasp_mat, end_effecter_mat = get_pose_matrixes(T, k_end_effector=0.184, k_pre_grasp=0.25)
    return target_point, pre_grasp_mat, end_effecter_mat, T, normal


def  gripper_processing(index,point_clouds,poses,isvis):
    target_point = point_clouds[index]
    relative_pose_5=poses[index]
    T_d, width, distance = convert_angles_to_transformation_form(relative_pose_5, target_point)

    activate_exploration=True if np.random.rand()<exploration_probabilty else False
    T_d,distance,width,collision_intensity = local_exploration(T_d, width, distance ,point_clouds, exploration_attempts=5,
                                             explore_if_collision=False, view_if_sucess=view_masked_grasp_pose,explore=activate_exploration)
    success=collision_intensity==0
    if not success:
        return False,width, distance, T_d, target_point
    if simulation_mode==False:
        pre_grasp_mat, end_effecter_mat = get_pose_matrixes(T_d, k_end_effector=0.169, k_pre_grasp=0.23)
        save_grasp_data(end_effecter_mat, width, grasp_data_path)
        save_grasp_data(pre_grasp_mat, width, pre_grasp_data_path)
    if isvis: vis_scene(T_d,width,npy=point_clouds)
    return True, width, distance, T_d, target_point

def suction_processing(index,point_data,normals,isvis):
    normal=normals[index]
    normal=normal[None,:]

    target_point, pre_grasp_mat, end_effecter_mat, T, normal = get_suction_pose_(point_data[index], normal)

    save_suction_data(end_effecter_mat, suction_data_path)
    save_suction_data(pre_grasp_mat, pre_suction_data_path)

    if isvis: visualize_suction_pose(target_point, normal.reshape(1, 3) , T, end_effecter_mat,npy=point_data)

    return True,target_point,normal