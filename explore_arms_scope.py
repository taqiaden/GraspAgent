# -*- coding: utf-8 -*-
from datetime import datetime
import math
import torch
import numpy as np
from colorama import Fore
from Configurations.dynamic_config import counters_file_path, get_int, save_key
from grasp_post_processing import grasp_data_path, pre_grasp_data_path, suction_data_path, pre_suction_data_path
from lib.ROS_communication import wait_for_feedback, save_grasp_data, save_suction_data
from lib.bbox import convert_angles_to_transformation_form
from lib.dataset_utils import configure_smbclient, modality_pool
from Configurations import config, ENV_boundaries
import subprocess
from lib.grasp_utils import get_pose_matrixes
from lib.math_utils import seeds

execute_suction_bash = './bash/run_robot_suction.sh'
execute_grasp_bash = './bash/run_robot_grasp.sh'
gripper_scope_data_path='./dataset/scope_data/gripper/'
suction_scope_data_path='./dataset/scope_data/suction/'
scope_data_dir='./dataset/scope_data/'
configure_smbclient()

def catch_random_grasp_point():
    '''pick random point'''
    relative_center_point = np.random.rand(3)
    relative_pose_5 = torch.rand(5)

    '''get the real coordinate'''
    x = relative_center_point[0] * (ENV_boundaries.x_limits[1] - ENV_boundaries.x_limits[0]) + ENV_boundaries.x_limits[0]
    y = relative_center_point[1] * (ENV_boundaries.y_limits[1] - ENV_boundaries.y_limits[0]) + ENV_boundaries.y_limits[0]
    z = relative_center_point[2] * (ENV_boundaries.z_limits[1] - ENV_boundaries.z_limits[0]) + ENV_boundaries.z_limits[0]
    target_point = np.stack([x, y, z])

    '''decode'''
    T_d, width, distance = convert_angles_to_transformation_form(relative_pose_5, target_point)

    return T_d, width

def save_scope_label(T,feasible,pool_object,pool_name):
    index = get_int(pool_name, config_file=counters_file_path) + 1
    transformation = T.reshape(-1).tolist()
    label = np.array([feasible] + transformation)
    pool_object.save_as_numpy(label, str(index).zfill(7))
    save_key(pool_name, index, config_file=counters_file_path)

def save_suction_label(T,feasible,suction_pool):
    save_scope_label(T, feasible, suction_pool, 'suction_scope')

def save_gripper_label(T,feasible,gripper_pool):
    save_scope_label(T, feasible, gripper_pool, 'gripper_scope')

def process_feedback(state_):
    state_ = wait_for_feedback(state_)
    if state_ == 'reachable':
        print(Fore.GREEN, 'Feasible path plan exists for suction', Fore.RESET)
        return 1.
    elif state_ == 'unreachable':
        print(Fore.RED, 'No feasible path plan was found for suction', Fore.RESET)
        return 0.
    else:
        print(Fore.RED, 'undefined state', Fore.RESET)
        return None

def main():
    gripper_pool=modality_pool('gripper_label',parent_dir=scope_data_dir,is_local=True)
    suction_pool=modality_pool('suction_label',parent_dir=scope_data_dir,is_local=True)

    with torch.no_grad():

        for i in range(1000):
            with open(config.home_dir + "ros_execute.txt", 'w') as f:
                f.write('Wait')
            state_='Wait'
            feasible=0.

            '''get matrices'''
            T, grasp_width = catch_random_grasp_point()
            pre_grasp_mat, end_effecter_mat = get_pose_matrixes(T, k_end_effector=0.169, k_pre_grasp=0.23)

            '''publish gripper pose'''
            save_grasp_data(end_effecter_mat, grasp_width, grasp_data_path)
            save_grasp_data(pre_grasp_mat, grasp_width, pre_grasp_data_path)

            '''publish suction pose'''
            save_suction_data(end_effecter_mat, suction_data_path)
            save_suction_data(pre_grasp_mat, pre_suction_data_path)

            '''check gripper feasibility'''
            subprocess.run(execute_grasp_bash)
            feasible=process_feedback(state_)

            '''save gripper label'''
            save_gripper_label(T,feasible,gripper_pool)

            with open(config.home_dir + "ros_execute.txt", 'w') as f:
                f.write('Wait')
            state_ = 'Wait'

            '''check suction feasibility'''
            subprocess.run(execute_suction_bash)
            feasible=process_feedback(state_)

            '''save suction label'''
            save_suction_label(T,feasible,suction_pool)

if __name__ == "__main__":
    time_seed = math.floor(datetime.now().timestamp())
    seeds(time_seed)
    main()
