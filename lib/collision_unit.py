import copy
import math

import numpy as np
import torch
import trimesh
from colorama import Fore

from Configurations import config
from Configurations.ENV_boundaries import dist_allowance
from lib.bbox import decode_gripper_pose, encode_gripper_pose
from lib.grasp_utils import get_homogenous_matrix, get_center_point, shift_a_distance, update_pose_
from lib.mesh_utils import construct_gripper_mesh

increase_depth=True
width_exploration=True
activate_angle_exploration=False

view_collision_check = False
activate_width_clip=False
explore_theta_p=1.0
explore_phi_p=1.0

def grasp_collision_detection(pose_good_grasp_,point_data, visualize=False):
    pose_good_grasp=np.copy(pose_good_grasp_)
    assert np.any(np.isnan(pose_good_grasp))==False,f'{pose_good_grasp}'
    #########################################################
    # allow for distance inference in the approach direction
    pose_good_grasp=adjust_distance(pose_good_grasp)


    T=get_homogenous_matrix(pose_good_grasp)
    width=pose_good_grasp[0,0]

    point_data =copy.deepcopy(point_data)

    assert point_data.shape[0]>0,f'{point_data.shape}'

    check_result_fast = collision_detection_fast(width, T, point_data)

    if  visualize or view_collision_check:
        mesh = construct_gripper_mesh(width, T)
        scene = trimesh.Scene()
        scene.add_geometry([trimesh.PointCloud(point_data), mesh])
        scene.show()


    return check_result_fast

def get_noncollisonal_width(pose_good_grasp, point_data,n_attempts=50):
    max_width = config.width_scope
    width_step = config.width_bin_size
    min_width=0.005
    # If the gripper is already at the maximum opening, try to find a soltuion while incrementally closing the gripper
    original_width=pose_good_grasp[0, 0]
    width=pose_good_grasp[0, 0]
    n = 1 if np.random.rand()>((width)/(max_width))**2 else -1
    # if n>0:print('try to increase width')
    # else:print('try to decrease width')
    for i in range(n_attempts):

        width += width_step * n
        # print('new width=',width)
        if not max_width >= width >= min_width:
            pose_good_grasp[0,0]=original_width
            return False,pose_good_grasp

        pose_good_grasp[0, 0] = width
        collision_intensity=grasp_collision_detection(pose_good_grasp,point_data, visualize=False)
        if collision_intensity==0:
            # grasp_collision_detection(pose_good_grasp[0, 0], T,visualize=True)
            return True,pose_good_grasp
    else:
        pose_good_grasp[0, 0] = original_width
        return False, pose_good_grasp

def get_angle_with_horizon(T):
    hypotenuse = math.sqrt(T[0, 0] ** 2 + T[1, 0] ** 2)
    angle_with_horizon = math.pi / 2 if abs(hypotenuse) < 0.00001 else -math.atan(T[2, 0] / hypotenuse)
    return angle_with_horizon

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
def get_noncollisonal_rotation(pose_good_grasp,point_data, n_attempts=20, diversity_factor=1.):
    center_point=get_center_point(pose_good_grasp)

    for i in range(n_attempts):
        # print('new pose')
        # print(pose_good_grasp)

        pose = encode_gripper_pose(pose_good_grasp)

        pose=update_orientation(pose,update_range=diversity_factor)
        pose_good_grasp = decode_gripper_pose(pose, center_point)

        collision_intensity =grasp_collision_detection(pose_good_grasp, point_data,visualize=False)
        if  collision_intensity==0:
            return True,pose_good_grasp
    else:
        return False,pose_good_grasp

def resolve_collssion(pose_good_grasp,point_data,n_attempts=5,diversity_factor=1.,explore_angles=True):
    # if collission free is found return a new pose_good_grasp otherwise return the orginal pose_good_grasp
    print('     Try to rersolve collision')
    tmp_pose = copy.deepcopy(pose_good_grasp)

    for i in range(n_attempts):
        # Try to change the gripper orientation
        angles_diversity=min(1.,(1+(i/n_attempts))*diversity_factor)
        if explore_angles:
            sucess,tmp_pose=get_noncollisonal_rotation(tmp_pose,point_data, n_attempts=3,diversity_factor=angles_diversity)
            if sucess:
                # r=grasp_collision_detection(tmp_pose, point_data, visualize=False)
                # assert r==0
                print('     Collision resolved with alternative rotation')
                return True,tmp_pose
            elif width_exploration==False:
                return False, pose_good_grasp
        # Try to change the gripper width
        # else:
        if width_exploration:
            sucess,tmp_pose=get_noncollisonal_width(tmp_pose,point_data)
            if sucess:
                # r = grasp_collision_detection(tmp_pose, point_data, visualize=False)
                # assert r == 0
                print('     Collision resolved with alternative gripper width')
                return True,tmp_pose
            elif explore_angles==False:
                return False, pose_good_grasp

    else:
        return False,pose_good_grasp

def get_distance_step(pose_good_grasp):
    distance_step = 0.00375 # in meter
    current_distance = pose_good_grasp[0, -1]

    if np.random.rand() > ((current_distance) / (config.distance_scope)) ** 2:
        print('------Try to increase distance')
        distance_step = distance_step
    else:
        print('------Try to decrease distance')
        distance_step = -distance_step

    return distance_step

def clip_width(pose_good_grasp, width_step,point_data,min_width=0.005):

    # If the gripper is already at the maximum opening, try to find a solution while incrementally closing the gripper
    width=copy.deepcopy(pose_good_grasp[0, 0])
    pose_good_grasp_copy = copy.deepcopy(pose_good_grasp)

    while True:

        width += width_step
        if not  width >= min_width:
            break
        pose_good_grasp_copy[0,0]=width
        collision_intensity_tmp=grasp_collision_detection(pose_good_grasp_copy,point_data, visualize=False)
        if collision_intensity_tmp>0:
            break
        else:
            print('Width clip to size=', width)
            pose_good_grasp[0, 0]=width

    return pose_good_grasp

def distance_forward(pose_good_grasp,distance_step):
    T = get_homogenous_matrix(pose_good_grasp)
    T = shift_a_distance(T, distance_step)
    pose_good_grasp = update_pose_(T, pose_good_grasp, distance=pose_good_grasp[0, -1] + distance_step)
    print(f'new distance={pose_good_grasp[0, -1]}')
    return pose_good_grasp
def adjust_distance(pose_good_grasp):
    center_point = get_center_point(pose_good_grasp)
    pose_good_grasp_2 = np.copy(pose_good_grasp)

    pose_good_grasp_2[0, -1] -= dist_allowance
    pose = encode_gripper_pose(pose_good_grasp_2)
    pose_good_grasp_2 = decode_gripper_pose(pose, center_point)

    return pose_good_grasp_2

def correct_theta(pose_good_grasp):
    # return pose_good_grasp
    center_point = get_center_point(pose_good_grasp)
    if center_point[-1] < 0.082:
        pose = encode_gripper_pose(pose_good_grasp)
        # pose=torch.tensor([theta,phi,beta,distance, width]).to('cuda').float()
        k=(100/3)*(0.082-center_point[-1]) # theta_correction_factor
        corrected_theta=pose[0,0]*(1-k)
        pose[0,0]=corrected_theta
        pose_good_grasp = decode_gripper_pose(pose, center_point)
    return pose_good_grasp
def local_exploration(pose_good_grasp,point_data, exploration_attempts=5,explore_if_collision=False,view_if_sucess=False,explore=True):

    pose_good_grasp=correct_theta(pose_good_grasp)

    # grasp_collision_detection(pose_good_grasp, point_data, visualize=True)

    best_pose = copy.deepcopy(pose_good_grasp)
    distance_step = get_distance_step(pose_good_grasp)

    prediction_has_collision=True
    depth_is_altered=False
    evaluation_metric=None
    for i in range(exploration_attempts):
        if explore:print('- Exploration attempt ', i + 1)
        collision_intensity =grasp_collision_detection(pose_good_grasp ,point_data,visualize=False)
        if i==0:evaluation_metric=collision_intensity
        if collision_intensity>0:
            if not explore: break
            print('     Collision warning detected')
            # diversity controls the step size of angles and gripper width when trying to resolve the collision
            diversity=1.-i/exploration_attempts
            n=1*(exploration_attempts-i)
            if i==0:
                success_of_current_attempt,pose_good_grasp=resolve_collssion(pose_good_grasp,point_data,n_attempts=n,diversity_factor=diversity,explore_angles=activate_angle_exploration and distance_step>0 and i==0)

                if success_of_current_attempt:
                    if i > 0: i -= 1
                    best_pose = copy.deepcopy(pose_good_grasp)
                    evaluation_metric = 0

                    if i != 0: depth_is_altered = True

                else:
                    print('     Failed to resolve the collision')
                    break
        else:
            if i>0:i-=1
            if i != 0: depth_is_altered = True
            if i==0:prediction_has_collision=False
            best_pose = copy.deepcopy(pose_good_grasp)
            evaluation_metric=0
            if explore_if_collision and i==0: break


        if not increase_depth: break
        # Check for the distance scope
        if pose_good_grasp[0, -1]+distance_step < 0 or pose_good_grasp[0, -1]+distance_step > config.distance_scope:
            break

        if distance_step<0 and evaluation_metric==0: break # In case of trying to decrease the distance, if a solution is found break the process.
        # start the test with extra downward incremental movement

        pose_good_grasp=distance_forward(pose_good_grasp,distance_step)
        pose_good_grasp=correct_theta(pose_good_grasp)

    if evaluation_metric==0:
        # r = grasp_collision_detection(best_pose, point_data, visualize=False)
        # assert r == 0, f'{r}'
        # find better width fit if solution is found
        if activate_width_clip:
            pose_good_grasp=clip_width(best_pose, -1*config.width_bin_size, point_data, min_width=0.005)
            # r = grasp_collision_detection(pose_good_grasp, point_data, visualize=False)
            # assert r == 0,f'{r}'
            best_pose = copy.deepcopy(pose_good_grasp)

        pose = encode_gripper_pose(best_pose)
        print(Fore.CYAN, f'Altered pose metrics {pose}', Fore.RESET)
    if view_if_sucess and evaluation_metric==0:
        collision_intensity = grasp_collision_detection(best_pose, point_data, visualize=True)
    # print(collision_intensity)
    # view_local_grasp(best_pose, point_data)
    # print(f'----------------------{success},           {evaluation_metric},   {best_pose}')
    print(Fore.RESET)
    if evaluation_metric==0 and prediction_has_collision: print('Grasp pose has been modified after exploration')
    if depth_is_altered:print('Approach distance has been altered')

    #final check
    if evaluation_metric==0:
        collision_intensity = grasp_collision_detection(best_pose, point_data, visualize=False)
        if evaluation_metric==0 and collision_intensity>0:print(Fore.RED,f'Conflict in the collision detection unit, type B, recorded metric={evaluation_metric}, actual metric={collision_intensity}',Fore.RESET)
        evaluation_metric=collision_intensity
    return best_pose,evaluation_metric


def collision_detection_fast(width, T, points):
    '''
    width: Gripper total width（m）： float
    T： The position of the gripper in the scene ： numpy_array (4*4)
    points: Object points in the scene ： numpy_array (n*3)
    '''
    # 1、Multiply the point cloud by the inv of T
    ones_p = np.ones([points.shape[0], 1], dtype=points.dtype)
    points_ = np.concatenate([points, ones_p], axis=-1)
    inverse_trans = np.linalg.inv(T)
    points_ = np.matmul(inverse_trans, points_.T).T
    object_points = points_[:, :3]

    # 2、Find the x,y,z boundary of the gripper according to width
    z_min = -0.007
    z_max = 0.007
    x_max = 0.004
    x_large_min = -0.09725
    x_small_min = -0.0355
    y_large_min = -0.0005 - width / 2 - 0.004
    y_large_max = 0.0005 + width / 2 + 0.004
    y_small_min = -0.0005 - width / 2
    y_small_max = 0.0005 + width / 2

    # 3、Determine whether a collision occurs based on the boundary, and whether there is a point inside the large cube and outside the small cube
    x = object_points[:, 0]
    y = object_points[:, 1]
    z = object_points[:, 2]
    val_x_large = (x < x_max) & (x > x_large_min)
    val_y_large = (y < y_large_max) & (y > y_large_min)
    val_z_large = (z < z_max) & (z > z_min)
    val_large = val_x_large & val_y_large & val_z_large
    collision_points = object_points[val_large]
    if collision_points.size == 0:
        return 0

    x = collision_points[:, 0]
    y = collision_points[:, 1]
    z = collision_points[:, 2]
    val_x_small = (x < x_max) & (x > x_small_min)
    val_y_small = (y < y_small_max) & (y > y_small_min)
    val_z_small = (z < z_max) & (z > z_min)
    val_small = val_x_small & val_y_small & val_z_small
    if np.any(~val_small):
        # print(collision_points[~val_small])
        return 1
    else:
        return 0