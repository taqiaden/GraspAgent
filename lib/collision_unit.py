import copy
import math

import numpy as np
import trimesh
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from Configurations import config
from Configurations.ENV_boundaries import dist_allowance, floor_elevation
from Configurations.run_config import activate_grasp_quality_check
from lib.grasp_utils import shift_a_distance
from lib.mesh_utils import construct_gripper_mesh


def grasp_collision_detection(T_d_,width,point_data, visualize=False,with_allowance=True,allowance=dist_allowance,check_floor_collision=True,floor_elevation_=None):
    T_d=np.copy(T_d_)
    assert np.any(np.isnan(T_d))==False,f'{T_d}'
    #########################################################
    '''Push the gripper to the maximum inference allowance'''
    if with_allowance:
        T_d=shift_a_distance(T_d, -allowance)

    point_data =copy.deepcopy(point_data)

    assert point_data.shape[0]>0,f'{point_data.shape}'

    has_collision,grasp_quality = fast_singularity_check(width, T_d, point_data,check_floor_collision=check_floor_collision,floor_elevation_=floor_elevation_)

    if  visualize:
        mesh = construct_gripper_mesh(width, T_d)
        scene = trimesh.Scene()
        scene.add_geometry([trimesh.PointCloud(point_data), mesh])
        scene.show()

    return has_collision,(grasp_quality<0.5 if grasp_quality is not None else False)

def gripper_firmness_check(T_d_,width,point_data, visualize=False,with_allowance=True,check_floor_collision=True,floor_elevation_=None):
    T_d=np.copy(T_d_)
    assert np.any(np.isnan(T_d))==False,f'{T_d}'
    assert np.any(np.isnan(width))==False,f'{width}'
    assert np.any(np.isnan(point_data))==False,f'{point_data}'

    #########################################################
    '''Push the gripper to the maximum inference allowance'''
    if with_allowance:
        T_d=shift_a_distance(T_d, -dist_allowance)

    point_data =copy.deepcopy(point_data)

    assert point_data.shape[0]>0,f'{point_data.shape}'

    has_collision,firmness_val,quality,collision_val = fast_singularity_check_with_firmness_evaluation(width, T_d, point_data,check_floor_collision=check_floor_collision,floor_elevation_=floor_elevation_)

    if  visualize:
        mesh = construct_gripper_mesh(width, T_d)
        scene = trimesh.Scene()
        scene.add_geometry([trimesh.PointCloud(point_data), mesh])
        scene.show()

    return has_collision,firmness_val,quality,collision_val

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
        collision_intensity_tmp,low_quality_grasp=grasp_collision_detection(T_d,new_width,point_data, visualize=False)
        if collision_intensity_tmp>0:
            break
        else:
            print('Width clip to size=', width)
            width=new_width

    return new_width

def transform_points(points,T,inverse=False):
    if inverse:
        # object_points: shape [N, 3]
        ones_p = np.ones([points.shape[0], 1], dtype=points.dtype)
        object_points_ = np.concatenate([points, ones_p], axis=-1)  # [N, 4]

        # Apply original transformation T (NOT inverse)
        points_ = np.matmul(T, object_points_.T).T  # [N, 4]

        # Extract original 3D points
        return points_[:, :3]
    else:
        ones_p = np.ones([points.shape[0], 1], dtype=points.dtype)
        points_ = np.concatenate([points, ones_p], axis=-1)
        inverse_trans = np.linalg.inv(T)
        points_ = np.matmul(inverse_trans, points_.T).T
        object_points = points_[:, :3]
        return object_points

def fast_singularity_check(width, T, points,check_floor_collision=True,floor_elevation_=None):
    '''
    width: Gripper total width（m）： float
    T： The position of the gripper in the scene ： numpy_array (4*4)
    points: Object points in the scene ： numpy_array (n*3)
    '''
    if floor_elevation_ is None:floor_elevation_=floor_elevation
    # 1、Multiply the point cloud by the inv of T
    object_points = transform_points(points,T,inverse=False)

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

    lower_extreme_points=np.array([[x_max,y_large_max,z_max],[x_max,y_large_max,z_min],[x_max,y_large_min,z_max],[x_max,y_large_min,z_min]])

    if collision_points.size == 0:
        if check_floor_collision:
            original_lower_extremes = transform_points(lower_extreme_points, T, inverse=True)
            collision_mask = original_lower_extremes[:, -1] < floor_elevation_-dist_allowance
            if collision_mask.any():
                # print(original_lower_extremes)
                # mesh = construct_gripper_mesh(width, T)
                # scene = trimesh.Scene()
                # scene.add_geometry([trimesh.PointCloud(points), mesh])
                # scene.show()
                return 1,None
        return 0,0

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
        return 1,None
    else:
        if check_floor_collision:
            original_lower_extremes = transform_points(lower_extreme_points, T, inverse=True)
            collision_mask = original_lower_extremes[:, -1] < floor_elevation_-dist_allowance
            if collision_mask.any():
                # print(original_lower_extremes)
                # mesh = construct_gripper_mesh(width, T)
                # scene = trimesh.Scene()
                # scene.add_geometry([trimesh.PointCloud(points), mesh])
                # scene.show()
                # set quality to zero if collision with unseen floor is detected
                return 0,0

        firmness_points = collision_points[val_small]

        tight_y_min=firmness_points[:,1].min()
        tight_y_max=firmness_points[:,1].max()

        def nearest_point(points,target):
            distances = np.sum((points - target) ** 2, axis=1)
            min_index = np.argmin(distances)
            nearest_point = points[min_index]
            return nearest_point

        p1=nearest_point(firmness_points[:,1:],np.array([[tight_y_min,z_min]]))
        p2=nearest_point(firmness_points[:,1:],np.array([[tight_y_min,z_max]]))
        v1=p2-p1
        p3=nearest_point(firmness_points[:,1:],np.array([[tight_y_max,z_min]]))
        p4=nearest_point(firmness_points[:,1:],np.array([[tight_y_max,z_max]]))
        v2=p4-p3

        quality=vector_alignment(v1, v2)
        v_m=(v1+v2)/2.
        v_ref=np.array([0.,1.])
        alignment=vector_alignment(v_m, v_ref)

        # print(quality)
        # print(alignment)

        # mesh = construct_gripper_mesh(width, T)
        # scene = trimesh.Scene()
        # scene.add_geometry([trimesh.PointCloud(points), mesh])
        # scene.show()

        return 0, quality*alignment

def vector_alignment(v1, v2):
    """
    Returns a continuous measure of vector alignment between 0 and 1.

    1 = perfectly parallel
    0 = perfectly orthogonal
    Intermediate values = degrees of alignment

    Parameters:
    v1, v2: Input vectors (numpy arrays or lists)

    Returns:
    Alignment value between 0 and 1
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    # Handle zero vectors
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0  # Consider orthogonal to avoid undefined cases

    # Compute normalized dot product (cosine of angle between vectors)
    cosine_similarity = np.dot(v1, v2) / (norm1 * norm2)

    # Convert to alignment measure (0 to 1 scale)
    # alignment = abs(cosine_similarity)
    alignment = cosine_similarity**2


    return alignment


def smooth_point_cloud(points, max_neighbors=10, max_distance=0.1, iterations=1):
    """
    Smooth a point cloud with constraints on neighbor count and distance.

    Args:
        points (np.ndarray): Input point cloud of shape (N, 3).
        max_neighbors (int): Maximum number of neighbors to consider.
        max_distance (float): Maximum allowable distance to a neighbor.
        iterations (int): Number of smoothing passes.

    Returns:
        np.ndarray: Smoothed point cloud.
    """
    smoothed_points = points.copy()
    nbrs = NearestNeighbors(n_neighbors=max_neighbors, algorithm='kd_tree').fit(smoothed_points)

    for _ in range(iterations):
        distances, indices = nbrs.kneighbors(smoothed_points)

        for i in range(len(smoothed_points)):
            # Filter neighbors by distance
            valid_mask = distances[i] <= max_distance
            valid_indices = indices[i][valid_mask]

            if len(valid_indices) > 0:  # Only smooth if neighbors exist
                smoothed_points[i] = np.mean(smoothed_points[valid_indices], axis=0)

        # Update neighbors for next iteration
        nbrs.fit(smoothed_points)

    return smoothed_points

def fast_singularity_check_with_firmness_evaluation(width, T, points,check_floor_collision=True,floor_elevation_=None):
    '''
    width: Gripper total width（m）： float
    T： The position of the gripper in the scene ： numpy_array (4*4)
    points: Object points in the scene ： numpy_array (n*3)
    Note: the axis changed after the transformation, the z axis becomes the x, and y is the gripper opening
    '''
    if floor_elevation_ is None: floor_elevation_=floor_elevation
    # 1、Multiply the point cloud by the inv of T
    object_points = transform_points(points, T, inverse=False)

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

    lower_extreme_points=np.array([[x_max,y_large_max,z_max],[x_max,y_large_max,z_min],[x_max,y_large_min,z_max],[x_max,y_large_min,z_min]])

    if collision_points.size == 0:
        '''no points in the big box'''
        if check_floor_collision:
            original_lower_extremes=transform_points(lower_extreme_points, T, inverse=True)
            collision_mask=original_lower_extremes[:,-1]<floor_elevation_-dist_allowance
            if collision_mask.any():
                # print(original_lower_extremes)
                # mesh = construct_gripper_mesh(width, T)
                # scene = trimesh.Scene()
                # scene.add_geometry([trimesh.PointCloud(points), mesh])
                # scene.show()
                # set quality to zero if collision with unseen floor is detected
                return 0,0,0,0
        return 0, 0,0,0

    x = collision_points[:, 0]
    y = collision_points[:, 1]
    z = collision_points[:, 2]
    val_x_small = (x < x_max) & (x > x_small_min)
    val_y_small = (y < y_small_max) & (y > y_small_min)
    val_z_small = (z < z_max) & (z > z_min)
    val_small = val_x_small & val_y_small & val_z_small # points inside the small box
    dist_ = x_max - x
    firmness_points_dist=dist_[val_small]
    firmness_weight = firmness_points_dist.mean() if firmness_points_dist.size>0 else 0
    firmness_weight /= config.distance_scope

    # print('points in gripper cavity=',points_in_cavity)

    if np.any(~val_small):
        collision_points = dist_[~val_small]
        collision_weight = collision_points.mean() if collision_points.size > 0 else 0
        # print(collision_points[~val_small])

        return 1, firmness_weight,0,collision_weight
    else:
        if check_floor_collision:
            original_lower_extremes=transform_points(lower_extreme_points, T, inverse=True)
            collision_mask=original_lower_extremes[:,-1]<floor_elevation_-dist_allowance
            if collision_mask.any():
                # print(original_lower_extremes)
                # mesh = construct_gripper_mesh(width, T)
                # scene = trimesh.Scene()
                # scene.add_geometry([trimesh.PointCloud(points), mesh])
                # scene.show()
                return 1,0,0,1

        firmness_points = collision_points[val_small]

        tight_y_min=firmness_points[:,1].min()
        tight_y_max=firmness_points[:,1].max()

        def nearest_point(points,target):
            distances = np.sum((points - target) ** 2, axis=1)
            min_index = np.argmin(distances)
            nearest_point = points[min_index]
            return nearest_point

        p1=nearest_point(firmness_points[:,1:],np.array([[tight_y_min,z_min]]))
        p2=nearest_point(firmness_points[:,1:],np.array([[tight_y_min,z_max]]))
        v1=p2-p1
        p3=nearest_point(firmness_points[:,1:],np.array([[tight_y_max,z_min]]))
        p4=nearest_point(firmness_points[:,1:],np.array([[tight_y_max,z_max]]))
        v2=p4-p3

        quality=vector_alignment(v1, v2)
        v_m=(v1+v2)/2.
        v_ref=np.array([0.,1.])
        alignment=vector_alignment(v_m, v_ref)

        return 0, firmness_weight,quality*alignment, 0