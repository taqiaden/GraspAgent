import numpy as np
import trimesh

from Configurations import config
from Configurations.config import home_dir
from lib.grasp_utils import get_pose_matrixes, adjust_final_matrix, shift_a_distance

ROS_communication_file="ros_execute.txt"

gripper_grasp_data_path = config.home_dir + 'gripper_grasp_data_tmp.npy'
gripper_pre_grasp_data_path = config.home_dir + 'gripper_pre_grasp_data_tmp.npy'
suction_grasp_data_path = config.home_dir + 'suction_grasp_data_tmp.npy'
suction_pre_grasp_data_path = config.home_dir + 'suction_pre_grasp_data_tmp.npy'

gripper_pre_shift_data_path = config.home_dir + 'gripper_pre_shift_data_tmp.npy'
gripper_shift_data_path = config.home_dir + 'gripper_shift_data_tmp.npy'
gripper_end_shift_data_path = config.home_dir + 'gripper_end_shift_data_tmp.npy'

suction_shift_data_path = config.home_dir + 'suction_shift_data_tmp.npy'
suction_pre_shift_data_path = config.home_dir + 'suction_pre_shift_data_tmp.npy'
suction_end_shift_data_path = config.home_dir + 'suction_end_shift_data_tmp.npy'


def deploy_action( action):
    if action.is_grasp:
        if action.use_gripper_arm:
            deploy_gripper_grasp_command(action)
        else:
            '''suction grasp'''
            deploy_suction_grasp_command(action)
    else:
        '''shift'''
        if action.use_gripper_arm:
            deploy_gripper_shift_command(action)
        else:
            '''suction grasp'''
            deploy_suction_shift_command(action)

def deploy_gripper_grasp_command( action):
    pre_mat = adjust_final_matrix(action.transformation, x_correction=-0.23)
    end_mat = adjust_final_matrix(action.transformation, x_correction=-0.169)
    save_gripper_data(end_mat, action.real_width, gripper_grasp_data_path)
    save_gripper_data(pre_mat, action.real_width, gripper_pre_grasp_data_path)

def deploy_suction_grasp_command( action):
    pre_mat = adjust_final_matrix(action.transformation, x_correction=-0.25)
    end_mat = adjust_final_matrix(action.transformation, x_correction=-0.184)
    save_suction_data(end_mat, suction_grasp_data_path)
    save_suction_data(pre_mat, suction_pre_grasp_data_path)

def deploy_gripper_shift_command( action):
    pre_mat=adjust_final_matrix(action.transformation, x_correction=-0.23)
    end_mat = action.transformation
    end_mat = shift_a_distance(end_mat, - 0.005)
    end_mat=adjust_final_matrix(end_mat, x_correction=-0.169)

    shift_end_mat = np.copy(action.transformation)
    shift_end_mat[0:3, 3] = action.shift_end_point.cpu().numpy()
    shift_end_mat=shift_a_distance(shift_end_mat, -0.01)
    shift_end_mat=adjust_final_matrix(shift_end_mat, x_correction=-0.169)

    save_gripper_data(pre_mat, action.real_width, gripper_pre_shift_data_path)
    save_gripper_data(end_mat, action.real_width, gripper_shift_data_path)
    save_gripper_data(shift_end_mat, action.real_width, gripper_end_shift_data_path)

def deploy_suction_shift_command( action):
    pre_mat = adjust_final_matrix(action.transformation, x_correction=-0.25)
    end_mat=action.transformation
    end_mat=shift_a_distance(end_mat,- 0.01)
    end_mat = adjust_final_matrix(end_mat, x_correction=-0.184)
    shift_end_mat = np.copy(action.transformation)
    shift_end_mat[0:3, 3] = action.shift_end_point.cpu().numpy()
    shift_end_mat=shift_a_distance(shift_end_mat,- 0.015)
    shift_end_mat = adjust_final_matrix(shift_end_mat, x_correction=-0.184)

    save_suction_data(pre_mat, suction_pre_shift_data_path)
    save_suction_data(end_mat, suction_shift_data_path)
    save_suction_data(shift_end_mat, suction_end_shift_data_path)

def save_suction_data(end_effecter_mat, file_path):
    wxyz = trimesh.transformations.quaternion_from_matrix(end_effecter_mat)
    xyz = end_effecter_mat[:3, 3]
    suction_data = np.array([xyz[0], xyz[1], xyz[2], wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    np.save(file_path, suction_data)
    # print('suction_data: ', suction_data)

def save_gripper_data(end_effecter_mat, grasp_width, file_path):
    wxyz = trimesh.transformations.quaternion_from_matrix(end_effecter_mat)
    xyz = end_effecter_mat[:3, 3]

    grasp_data = np.array([xyz[0], xyz[1], xyz[2], wxyz[1], wxyz[2], wxyz[3], wxyz[0], grasp_width])
    np.save(file_path, grasp_data)
def set_wait_flag():
    with open(config.home_dir + ROS_communication_file, 'w') as f:
        f.write('Wait')

def read_robot_feedback():
    with open(home_dir + ROS_communication_file, 'r') as f:
        txt = f.read()
    return txt
