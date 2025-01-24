import copy
import math

import numpy as np
import trimesh

from Configurations import config
from Configurations.ENV_boundaries import dist_allowance
from lib.grasp_utils import shift_a_distance
from lib.mesh_utils import construct_gripper_mesh


def grasp_collision_detection(T_d_,width,point_data, visualize=False,with_allowance=True):
    T_d=np.copy(T_d_)
    assert np.any(np.isnan(T_d))==False,f'{T_d}'
    #########################################################
    '''Push the gripper to the maximum inference allowance'''
    if with_allowance:
        T_d=shift_a_distance(T_d, -dist_allowance)

    point_data =copy.deepcopy(point_data)

    assert point_data.shape[0]>0,f'{point_data.shape}'

    has_collision = fast_singularity_check(width, T_d, point_data)

    if  visualize:
        mesh = construct_gripper_mesh(width, T_d)
        scene = trimesh.Scene()
        scene.add_geometry([trimesh.PointCloud(point_data), mesh])
        scene.show()

    return has_collision

def gripper_firmness_check(T_d_,width,point_data, visualize=False,with_allowance=True):
    T_d=np.copy(T_d_)
    assert np.any(np.isnan(T_d))==False,f'{T_d}'
    #########################################################
    '''Push the gripper to the maximum inference allowance'''
    if with_allowance:
        T_d=shift_a_distance(T_d, -dist_allowance)

    point_data =copy.deepcopy(point_data)

    assert point_data.shape[0]>0,f'{point_data.shape}'

    has_collision,firmness_val,collision_val = fast_singularity_check_with_firmness_evaluation(width, T_d, point_data)

    if  visualize:
        mesh = construct_gripper_mesh(width, T_d)
        scene = trimesh.Scene()
        scene.add_geometry([trimesh.PointCloud(point_data), mesh])
        scene.show()

    return has_collision,firmness_val,collision_val

def get_angle_with_horizon(T):
    hypotenuse = math.sqrt(T[0, 0] ** 2 + T[1, 0] ** 2)
    angle_with_horizon = math.pi / 2 if abs(hypotenuse) < 0.00001 else -math.atan(T[2, 0] / hypotenuse)
    return angle_with_horizon

def get_distance_step(distance):
    distance_step = 0.00375 # in meter

    if np.random.rand() > ((distance) / (config.distance_scope)) ** 2:
        # print('------Try to increase distance')
        distance_step = distance_step
    else:
        # print('------Try to decrease distance')
        distance_step = -distance_step

    return distance_step

def clip_width(T_d,width_, width_step,point_data,min_width=0.005):

    # If the gripper is already at the maximum opening, try to find a solution while incrementally closing the gripper
    new_width=copy.deepcopy(width_)

    while True:

        new_width += width_step
        if not  new_width >= min_width:
            break
        collision_intensity_tmp=grasp_collision_detection(T_d,new_width,point_data, visualize=False)
        if collision_intensity_tmp>0:
            break
        else:
            print('Width clip to size=', width)
            width=new_width

    return new_width

def fast_singularity_check(width, T, points):
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

    # print('points in gripper cavity=',points_in_cavity)

    if np.any(~val_small):
        # print(collision_points[~val_small])
        return 1
    else:
        return 0


def fast_singularity_check_with_firmness_evaluation(width, T, points):
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
        return 0, 0,0

    x = collision_points[:, 0]
    y = collision_points[:, 1]
    z = collision_points[:, 2]
    val_x_small = (x < x_max) & (x > x_small_min)
    val_y_small = (y < y_small_max) & (y > y_small_min)
    val_z_small = (z < z_max) & (z > z_min)
    val_small = val_x_small & val_y_small & val_z_small
    dist_ = x_max - x
    firmness_weight = dist_[val_small].sum()

    # print('points in gripper cavity=',points_in_cavity)

    if np.any(~val_small):
        collision_weight=dist_[~val_small].sum()
        # print(collision_points[~val_small])
        return 1, firmness_weight,collision_weight
    else:
        return 0, firmness_weight,0