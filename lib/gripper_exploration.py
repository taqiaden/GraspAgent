import copy
import numpy as np
import torch
from colorama import Fore
from Configurations import config
from lib.bbox import transformation_to_relative_angle_form, convert_angles_to_transformation_form
from lib.collision_unit import get_distance_step, grasp_collision_detection
from lib.grasp_utils import get_target_point_2, shift_a_distance

explore_theta_p=1.0
explore_phi_p=1.0
increase_depth=True
width_exploration=True
activate_angle_exploration=False
view_collision_check = False

def update_orientation(pose,update_range=0.5):
    # if np.random.rand() > 0.7:
    #     pose[0,0:3]*=0
    #     if np.random.rand() > 0.8:return pose
    if np.random.rand()<explore_theta_p:
        random_val=((torch.rand(1)*2.)-1.).to('cuda')*update_range*0.5*0.5
        tmp=pose[0,0]
        pose[0,0]=pose[0,0]+random_val
        if pose[0,0]>1 or pose[0,0]<0:pose[0,0]=tmp-random_val
    if np.random.rand()<explore_phi_p:
        random_val=((torch.rand(1)*2.)-1.).to('cuda')*update_range*0.2
        pose[0, 1] =pose[0, 1]+ random_val
        if pose[0,1]>1:pose[0,1]=pose[0,1]-1
        if pose[0,1]<0:pose[0,1]=1+pose[0,1]
    random_val=((torch.rand(1)*2.)-1.).to('cuda')#*update_range
    pose[0, 2] =pose[0, 2]+ random_val
    if pose[0,2]>1:pose[0,2]=pose[0,2]-1
    if pose[0,2]<0:pose[0,2]=1+pose[0,2]
    # print(f'New beta==========={pose[0,2]}')
    return pose

def get_noncollisonal_width(T_d,width_, point_data,n_attempts=50):
    max_width = config.width_scope
    width_step = 0.005
    min_width=0.005
    # If the gripper is already at the maximum opening, try to find a solution while incrementally closing the gripper
    original_width=width_
    width=width_
    n = 1 if np.random.rand()>((width)/(max_width))**2 else -1
    for i in range(n_attempts):

        width += width_step * n
        if not max_width >= width >= min_width:
            return False,original_width

        collision_intensity=grasp_collision_detection(T_d,width,point_data, visualize=False)
        if collision_intensity==0:
            return True,width
    else:
        return False, original_width

def get_noncollisonal_rotation(T_d,distance,width,point_data, n_attempts=20, diversity_factor=1.):
    target_point=get_target_point_2(T_d,distance)

    for i in range(n_attempts):


        relative_pose_5 = transformation_to_relative_angle_form(T_d,distance,width)

        relative_pose_5=update_orientation(relative_pose_5,update_range=diversity_factor)
        T_d,width,distance=convert_angles_to_transformation_form(relative_pose_5, target_point)

        collision_intensity =grasp_collision_detection(T_d,width, point_data,visualize=False)
        if  collision_intensity==0:
            return True,T_d
    else:
        return False,T_d

def resolve_collssion(T_d,distance,width,point_data,n_attempts=5,diversity_factor=1.,explore_angles=True):
    print('     Try to resolve collision')
    tmp_T_d = copy.deepcopy(T_d)

    for i in range(n_attempts):
        angles_diversity=min(1.,(1+(i/n_attempts))*diversity_factor)
        if explore_angles:
            sucess,tmp_T_d=get_noncollisonal_rotation(tmp_T_d,distance,width,point_data, n_attempts=3,diversity_factor=angles_diversity)
            if sucess:
                print('     Collision resolved with alternative rotation')
                return True,tmp_T_d,width
            elif width_exploration==False:
                return False, T_d,width
        if width_exploration:
            sucess,new_width=get_noncollisonal_width(tmp_T_d,width,point_data)
            if sucess:
                print('     Collision resolved with alternative gripper width')
                return True,tmp_T_d,new_width
            elif explore_angles==False:
                return False, T_d,width

    else:
        return False,T_d,width

def local_exploration(T_d, width, distance ,point_data, exploration_attempts=5,explore_if_collision=False,view_if_sucess=False,explore=True):

    '''maintain track of the latest feasible gasp'''
    best_T_d,best_width,best_distance = copy.deepcopy(T_d),copy.deepcopy(width),copy.deepcopy(distance)

    '''set the distance perturbation step'''
    distance_step = get_distance_step(distance)

    prediction_has_collision=True
    depth_is_altered=False
    evaluation_metric=None
    for i in range(exploration_attempts):
        if explore:print('- Exploration attempt ', i + 1)
        '''initial collision check'''
        collision_intensity =grasp_collision_detection(T_d,width ,point_data,visualize=False)
        if i==0:evaluation_metric=collision_intensity
        if collision_intensity>0:
            if not explore: break
            # diversity controls the step size of angles and gripper width when trying to resolve the collision
            diversity=1.-i/exploration_attempts
            n=1*(exploration_attempts-i)
            if i==0:
                success_of_current_attempt,T_d,width=resolve_collssion(T_d,distance,width,point_data,n_attempts=n,diversity_factor=diversity,explore_angles=activate_angle_exploration and distance_step>0 and i==0)

                if success_of_current_attempt:
                    if i > 0: i -= 1
                    best_T_d, best_width, best_distance = copy.deepcopy(T_d), copy.deepcopy(width), copy.deepcopy(
                        distance)

                    evaluation_metric = 0

                    if i != 0: depth_is_altered = True
                else:
                    print('     Failed to resolve the collision')
                    break
        else:
            if i>0:i-=1
            if i != 0: depth_is_altered = True
            if i==0:prediction_has_collision=False
            best_T_d, best_width, best_distance = copy.deepcopy(T_d), copy.deepcopy(width), copy.deepcopy(distance)

            evaluation_metric=0
            if explore_if_collision and i==0: break


        if not increase_depth: break
        # Check for the distance scope
        if distance+distance_step < 0 or distance+distance_step > config.distance_scope:
            break

        if distance_step<0 and evaluation_metric==0: break # In case of trying to decrease the distance, if a solution is found break the process.

        '''Alter distance'''
        T_d = shift_a_distance(T_d, distance_step)
        distance+=distance_step


    '''view result'''
    if view_if_sucess and evaluation_metric==0:
        grasp_collision_detection(best_T_d,best_width, point_data, visualize=True)

    '''Report and final check'''
    if evaluation_metric==0 and prediction_has_collision: print('Grasp pose has been modified after exploration')
    if depth_is_altered:print('Approach distance has been altered')
    if evaluation_metric==0:
        collision_intensity = grasp_collision_detection(best_T_d,best_width, point_data, visualize=False)
        if evaluation_metric==0 and collision_intensity>0:print(Fore.RED,f'Conflict in the collision detection unit, type B, recorded metric={evaluation_metric}, actual metric={collision_intensity}',Fore.RESET)
        evaluation_metric=collision_intensity
    print(Fore.RESET)
    return best_T_d,best_distance,best_width,evaluation_metric
