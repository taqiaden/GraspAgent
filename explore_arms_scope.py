# -*- coding: utf-8 -*-
import math
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
from colorama import Fore

from Configurations import config, ENV_boundaries
from Configurations.dynamic_config import counters_file_path, get_int, save_key
from check_points.check_point_conventions import ModelWrapper
from lib.ROS_communication import save_suction_data, ROS_communication_file, set_wait_flag, \
    deploy_gripper_grasp_command_, deploy_suction_grasp_command_, read_robot_feedback
from lib.bbox import convert_angles_to_transformation_form
from lib.dataset_utils import configure_smbclient, modality_pool
from lib.grasp_utils import get_pose_matrixes
from lib.math_utils import seeds
from lib.models_utils import initialize_model_state
from lib.statistics import random_with_exponent_decay
from models.scope_net import scope_net_vanilla, gripper_scope_module_key, suction_scope_module_key
from records.training_satatistics import MovingRate

execute_suction_bash = './bash/run_robot_suction.sh'
execute_grasp_bash = './bash/run_robot_grasp.sh'
gripper_scope_data_path='./dataset/scope_data/gripper/'
suction_scope_data_path='./dataset/scope_data/suction/'
scope_data_dir='./dataset/scope_data/'
# scope_data_dir=r'/media/taqiaden/42c447a4-49c0-4d74-9b1f-4b4b5cbe7486/taqiaden_hub/scope_data/'
gripper_key='gripper_label'
suction_key='suction_label'
configure_smbclient()

maximum_scope_samples=100000

gripper_feasible_pose_rate = MovingRate('gripper_feasible_pose', min_decay=0.01,initial_val=0.5)
suction_feasible_pose_rate = MovingRate('suction_feasible_pose', min_decay=0.01,initial_val=0.5)


def catch_random_grasp_point():
    '''pick random point'''
    relative_center_point = np.random.rand(3)
    relative_pose_5 = torch.rand(5)
    relative_pose_5[0]=random_with_exponent_decay(2) # more randoms closer to zero
    # print('-----',relative_pose_5[0])
    explore_margin=0.1
    '''get the real coordinate'''
    x = relative_center_point[0] * (ENV_boundaries.x_limits[1] - ENV_boundaries.x_limits[0]+2.*explore_margin) + ENV_boundaries.x_limits[0]-explore_margin
    y = relative_center_point[1] * (ENV_boundaries.y_limits[1] - ENV_boundaries.y_limits[0]+2.*explore_margin) + ENV_boundaries.y_limits[0]-explore_margin
    z = relative_center_point[2] * (ENV_boundaries.z_limits[1] - ENV_boundaries.z_limits[0]+2.*explore_margin) + ENV_boundaries.z_limits[0]-explore_margin
    target_point = np.stack([x, y, z])
    # print(relative_pose_5,target_point)

    '''decode'''
    T_d, width, distance = convert_angles_to_transformation_form(relative_pose_5, target_point)

    return T_d, width

def save_scope_label(T,feasible,pool_object,pool_name):
    index = get_int(pool_name, config_file=counters_file_path) + 1
    if index>maximum_scope_samples:index=1

    transformation = T.reshape(-1).tolist()
    label = np.array([feasible] + transformation)
    pool_object.save_as_numpy(label, str(index).zfill(7))
    save_key(pool_name, index, config_file=counters_file_path)

def save_suction_label(T,feasible,suction_pool):
    save_scope_label(T, feasible, suction_pool, 'suction_scope')

def save_gripper_label(T,feasible,gripper_pool):
    save_scope_label(T, feasible, gripper_pool, 'gripper_scope')

def process_feedback(state_):
    robot_feedback_ = 'Wait'

    while robot_feedback_ == 'Wait' or robot_feedback_.strip() == '':
        robot_feedback_ = read_robot_feedback()
        time.sleep(0.1)


    if robot_feedback_ == 'reachable':
        print(Fore.GREEN, 'Feasible path plan exists', Fore.RESET)
        return 1.
    elif robot_feedback_ == 'unreachable':
        print(Fore.RED, 'No feasible path plan was found', Fore.RESET)
        return 0.
    else:
        print(Fore.RED, 'undefined state', Fore.RESET)
        return None

def sample_pose(model,feasible_rate):
    while True:
        '''get matrices'''
        T, grasp_width = catch_random_grasp_point()

        '''check sample score on scope net'''
        transition = T[0:3, 3].reshape(-1)
        approach = T[0:3, 0]
        pose = np.concatenate([transition, approach])
        pose=torch.from_numpy(pose).to('cuda')[None,:].float()
        score=model(pose)
        pivot_point=1-feasible_rate
        selection_p=1-abs(pivot_point-score.item())
        # print(selection_p,'-----',score)
        if np.random.rand()<selection_p**2.0:
            break
    return T,grasp_width

def generate_gripper_sample(gripper_pool,model):
    set_wait_flag()
    state_ = 'Wait'

    '''sample'''
    T,grasp_width=sample_pose(model,gripper_feasible_pose_rate.val)

    '''deploy gripper pose'''
    deploy_gripper_grasp_command_(T, grasp_width, angle=0.)

    '''check gripper feasibility'''
    subprocess.run(execute_grasp_bash)
    print(Fore.CYAN, '         Gripper planning: ', Fore.RESET,end='')
    feasible = process_feedback(state_)

    '''update rate record'''
    if feasible:gripper_feasible_pose_rate.update(1)
    else: gripper_feasible_pose_rate.update(0)

    '''save gripper label'''
    save_gripper_label(T, feasible, gripper_pool)

def report_failed_gripper_path_plan(T):
    gripper_pool=modality_pool(gripper_key,parent_dir=scope_data_dir,is_local=True)
    save_gripper_label(T, 0., gripper_pool)

def generate_suction_sample(suction_pool,model):
    with open(config.home_dir + ROS_communication_file, 'w') as f:
        f.write('Wait')
    state_ = 'Wait'

    '''sample'''
    T,grasp_width=sample_pose(model,suction_feasible_pose_rate.val)

    '''publish suction pose'''
    deploy_suction_grasp_command_(T)

    '''check suction feasibility'''
    subprocess.run(execute_suction_bash)
    print(Fore.CYAN, '         Suction planning: ', Fore.RESET,end='')
    feasible = process_feedback(state_)

    '''update rate record'''
    if feasible:suction_feasible_pose_rate.update(1)
    else: suction_feasible_pose_rate.update(0)

    '''save suction label'''
    save_suction_label(T, feasible, suction_pool)

def report_failed_suction_path_plan(T):
    suction_pool=modality_pool(suction_key,parent_dir=scope_data_dir,is_local=True)
    save_suction_label(T, 0., suction_pool)

def load_models():
    gripper_scope_model = ModelWrapper(model=scope_net_vanilla(in_size=6), module_key=gripper_scope_module_key)
    gripper_scope_model.ini_model(train=False)
    suction_scope_model = ModelWrapper(model=scope_net_vanilla(in_size=6), module_key=suction_scope_module_key)
    suction_scope_model.ini_model(train=False)
    return gripper_scope_model.model,suction_scope_model.model

def main():
    gripper_scope_model,suction_scope_model=load_models()

    gripper_pool=modality_pool(gripper_key,parent_dir=scope_data_dir,is_local=True)
    suction_pool=modality_pool(suction_key,parent_dir=scope_data_dir,is_local=True)

    with torch.no_grad():

        for i in range(1000000):
            generate_gripper_sample(gripper_pool,gripper_scope_model)
            generate_suction_sample(suction_pool,suction_scope_model)

            if i%10==0: # regularly update checkpoints

                gripper_feasible_pose_rate.view()
                suction_feasible_pose_rate.view()

                gripper_feasible_pose_rate.save()
                suction_feasible_pose_rate.save()
                try:
                    gripper_scope_model, suction_scope_model = load_models()
                except:
                    pass


if __name__ == "__main__":

    time_seed = math.floor(datetime.now().timestamp())
    seeds(time_seed)
    main()
