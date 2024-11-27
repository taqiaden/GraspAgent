import numpy as np
import torch
from torch import nn

from lib.math_utils import rotation_matrix_from_vectors, angle_between_vectors_cross
from lib.pc_utils import circle_to_points, compute_curvature

suction_zone_radius = 0.012
curvature_radius = 0.0025
curvature_deviation_threshold = 0.0025
angle_threshold_degree = 5.0
seal_ring_deviation = 0.002
suction_area_deflection = 0.005

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)
l1_loss=nn.L1Loss()
l1_smooth_loss=nn.SmoothL1Loss(beta=1.0)
mse_loss=nn.MSELoss()
bce_loss=nn.BCELoss()


def normals_check(normals,dist_mask,target_normal):
    '''region normals'''
    region_normals = normals[dist_mask]
    average_region_normal = np.mean(region_normals, axis=0)

    angle_radians, angle_degrees = angle_between_vectors_cross(average_region_normal, target_normal)
    # if angle_degrees < angle_threshold_degree:
    #     print(f'Angle difference between normals = {angle_degrees}')
    #
    # else:
    #     print(Fore.RED, f'Angle difference between normals = {angle_degrees}', Fore.RESET)

    return angle_degrees < angle_threshold_degree

def curvature_check(points_at_seal_region):
    curvature = compute_curvature(points_at_seal_region, radius=curvature_radius)
    curvature = np.array(curvature)
    curvature_std = curvature.std()
    # if curvature_std < curvature_deviation_threshold:
    #     print(f'curvature deviation= {curvature_std}')
    # else:
    #     print(Fore.RED, f'curvature deviation= {curvature_std}', Fore.RESET)

    return curvature_std < curvature_deviation_threshold

def deflection_check(target_normal,points_at_seal_region):
    R = rotation_matrix_from_vectors(target_normal, np.array([0, 0, 1]))
    transformed_points_at_seal_region = np.matmul(R, points_at_seal_region.T).T
    seal_deflection = np.max(transformed_points_at_seal_region[:, 2]) - np.min(transformed_points_at_seal_region[:, 2])
    # if seal_deflection < suction_area_deflection:
    #     print(f'seal deflection = {seal_deflection}')
    # else:
    #     print(Fore.RED, f'seal deflection = {seal_deflection}', Fore.RESET)

    return seal_deflection < suction_area_deflection

def seal_check(target_point,points_at_seal_region):
    seal_test_points = circle_to_points(radius=suction_zone_radius, number_of_points=100, x=target_point[0],
                                        y=target_point[1], z=target_point[2])

    xy_dist = np.linalg.norm(seal_test_points[:, np.newaxis, 0:2] - points_at_seal_region[np.newaxis, :, 0:2], axis=-1)
    min_xy_dist = np.min(xy_dist, axis=1)
    seal_deviation = np.max(min_xy_dist)
    # if seal_deviation < seal_ring_deviation:
    #     print(f'maximum seal deviation = {seal_deviation}')
    # else:
    #     print(Fore.RED, f'maximum seal deviation = {seal_deviation}', Fore.RESET)

    return seal_deviation < seal_ring_deviation

def suction_seal_loss(target_point,pc,normals,target_index,prediction_,statistics):
    '''mask suction region'''
    dist_ = np.linalg.norm(target_point[np.newaxis] - pc, axis=-1)
    dist_mask = dist_ < suction_zone_radius
    target_normal = normals[target_index]
    '''circle to points'''
    points_at_seal_region = pc[dist_mask]

    '''suction criteria'''
    first_criteria = normals_check(normals, dist_mask, target_normal)
    second_criteria = curvature_check(points_at_seal_region)
    third_criteria = deflection_check(target_normal, points_at_seal_region)
    fourth_criteria = seal_check(target_point, points_at_seal_region)

    '''suction seal loss'''
    if first_criteria and second_criteria and third_criteria and fourth_criteria:
        label = torch.ones_like(prediction_)
    else:
        label = torch.zeros_like(prediction_)

    statistics.update_confession_matrix(label, prediction_)
    return bce_loss(prediction_, label)