import numpy as np
import trimesh

from Configurations import config
from Configurations.config import home_dir
from lib.grasp_utils import get_pose_matrixes
from lib.report_utils import wait_indicator as wi

ROS_communication_file="ros_execute.txt"

gripper_grasp_data_path = config.home_dir + 'gripper_grasp_data_tmp.npy'
pre_gripper_grasp_data_path = config.home_dir + 'pre_gripper_grasp_data_tmp.npy'
suction_grasp_data_path = config.home_dir + 'suction_grasp_data_tmp.npy'
pre_suction_grasp_data_path = config.home_dir + 'pre_suction_grasp_data_tmp.npy'

gripper_shift_data_path = config.home_dir + 'gripper_shift_data_tmp.npy'
pre_gripper_shift_data_path = config.home_dir + 'pre_gripper_shift_data_tmp.npy'
suction_shift_data_path = config.home_dir + 'suction_shift_data_tmp.npy'
pre_suction_shift_data_path = config.home_dir + 'pre_suction_shift_data_tmp.npy'


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
    pre_mat, end_mat = get_pose_matrixes(action.transformation, k_end_effector=0.169, k_pre_grasp=0.23)
    save_gripper_data(end_mat, action.width, gripper_grasp_data_path)
    save_gripper_data(pre_mat, action.width, pre_gripper_grasp_data_path)

def deploy_suction_grasp_command( action):
    pre_mat, end_mat = get_pose_matrixes(action.transformation, k_end_effector=0.184, k_pre_grasp=0.25)
    save_suction_data(end_mat, suction_grasp_data_path)
    save_suction_data(pre_mat, pre_suction_grasp_data_path)

def deploy_gripper_shift_command( action):
    pre_mat, end_mat = get_pose_matrixes(action.transformation, k_end_effector=0.169, k_pre_grasp=0.23)
    save_gripper_data(end_mat, action.width, gripper_shift_data_path)
    save_gripper_data(pre_mat, action.width, pre_gripper_shift_data_path)

def deploy_suction_shift_command( action):
    pre_mat, end_mat = get_pose_matrixes(action.transformation, k_end_effector=0.184, k_pre_grasp=0.25)
    save_suction_data(end_mat, suction_shift_data_path)
    save_suction_data(pre_mat, pre_suction_shift_data_path)


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



def wait_for_feedback():
    # wait until grasp or suction finished
    txt = 'Wait'
    wait = wi('Waiting for robot feedback')
    while txt == 'Wait':
        wait.step(0.5)
        with open(home_dir + ROS_communication_file, 'r') as f:
            txt = f.read()
    else:
        wait.end()
        print('Robot state: ' + txt)
    return txt