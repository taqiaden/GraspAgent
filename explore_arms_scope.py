# -*- coding: utf-8 -*-
from datetime import datetime
import math

import torch
import numpy as np
from colorama import Fore

from Configurations.config import ip_address
from Configurations.dynamic_config import get_value, counters_file_path, get_float, get_int, save_key
from grasp_post_processing import grasp_data_path, pre_grasp_data_path, suction_data_path, pre_suction_data_path
from lib.ROS_communication import wait_for_feedback, save_grasp_data, save_suction_data
from lib.bbox import decode_gripper_pose
from lib.dataset_utils import configure_smbclient, modality_pool
from Configurations import config, ENV_boundaries
import subprocess

from lib.grasp_utils import get_pose_matrixes, get_homogenous_matrix, get_grasp_width
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
    center_point = np.stack([x, y, z])

    '''decode'''
    pose_good_grasp = decode_gripper_pose(relative_pose_5, center_point)
    T = get_homogenous_matrix(pose_good_grasp)
    grasp_width = get_grasp_width(pose_good_grasp)

    return T, grasp_width

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
            state_ = wait_for_feedback(state_)
            if state_=='reachable':
                print(Fore.GREEN,'Feasible path plan exists for gripper', Fore.RESET)
                feasible = 1.

            elif state_=='unreachable':
                print(Fore.RED,'No feasible path plan was found for gripper', Fore.RESET)
                feasible = 0.

            else:
                print(Fore.RED,'undefined state', Fore.RESET)
                return

            '''save gripper label'''
            index=get_int('gripper_scope',config_file=counters_file_path)+1
            transformation = T.reshape(-1).tolist()
            label=np.array([feasible]+transformation)
            # path=gripper_scope_data_path+'/'+str(index).zfill(7)+'_scope_label.npy'
            # np.save(path,label)
            gripper_pool.save_as_numpy(label,str(index).zfill(7))
            save_key('gripper_scope',index,config_file=counters_file_path)

            with open(config.home_dir + "ros_execute.txt", 'w') as f:
                f.write('Wait')
            state_ = 'Wait'

            '''check suction feasibility'''
            subprocess.run(execute_suction_bash)
            state_ = wait_for_feedback(state_)
            if state_=='reachable':
                print(Fore.GREEN,'Feasible path plan exists for suction', Fore.RESET)
                feasible = 1.

            elif state_ == 'unreachable':
                print(Fore.RED,'No feasible path plan was found for suction', Fore.RESET)
                feasible = 0.

            else:
                print(Fore.RED, 'undefined state', Fore.RESET)
                return

            '''save suction label'''
            index=get_int('suction_scope',config_file=counters_file_path)+1
            transformation = T.reshape(-1).tolist()
            label=np.array([feasible]+transformation)
            # path=suction_scope_data_path+'/'+str(index).zfill(7)+'_scope_label.npy'
            # np.save(path,label)
            suction_pool.save_as_numpy(label,str(index).zfill(7))
            save_key('suction_scope',index,config_file=counters_file_path)

if __name__ == "__main__":
    time_seed = math.floor(datetime.now().timestamp())
    seeds(time_seed)
    main()
