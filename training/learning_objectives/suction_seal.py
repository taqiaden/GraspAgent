import numpy as np
import torch
from colorama import Fore
from matplotlib import pyplot as plt
from torch import nn

from lib.math_utils import rotation_matrix_from_vectors, angle_between_vectors_cross
from lib.pc_utils import circle_to_points, compute_curvature, circle_cavity_to_point
from visualiztion import view_suction_zone, view_npy_open3d

suction_zone_radius = 0.02
# curvature_radius = 0.0025
# curvature_deviation_threshold = 0.01
# angle_threshold_degree = 5.0
seal_ring_deviation = 0.003
# suction_area_deflection = 0.005

minium_number_of_points = 150

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)
l1_loss=nn.L1Loss()
l1_smooth_loss=nn.SmoothL1Loss(beta=1.0)
mse_loss=nn.MSELoss()
bce_loss=nn.BCELoss()

def view_suction_area(pc,dist_mask,target_point,direction,spatial_mask):
    colors = np.zeros_like(pc)
    colors[spatial_mask, 0] += 1.
    colors[dist_mask, 1] += 1.
    view_suction_zone(target_point,direction, pc, colors)

# def normals_check(normals,dist_mask,target_normal):
#     '''region normals'''
#     region_normals = normals[dist_mask]
#     average_region_normal = np.mean(region_normals, axis=0)
#
#     angle_radians, angle_degrees = angle_between_vectors_cross(average_region_normal, target_normal)
#     # if angle_degrees < angle_threshold_degree:
#     #     print(f'Angle difference between normals = {angle_degrees}')
#     #
#     # else:
#     #     print(Fore.RED, f'Angle difference between normals = {angle_degrees}', Fore.RESET)
#
#     return angle_degrees < angle_threshold_degree

# def curvature_check(points_at_seal_region,visualize=False):
#     curvature = compute_curvature(points_at_seal_region, radius=curvature_radius)
#     curvature = np.array(curvature)
#     curvature_std = curvature.std()
#     if visualize:
#         plt.plot(curvature)
#         plt.show()
#         print(np.max(curvature)-np.min(curvature))
#         if curvature_std < curvature_deviation_threshold:
#             print(f'curvature deviation= {curvature_std}')
#         else:
#             print(Fore.RED, f'curvature deviation= {curvature_std}', Fore.RESET)
#
#     return curvature_std < curvature_deviation_threshold

def transform_point_to_normal_in_plane(target_normal,points_at_seal_region):
    R = rotation_matrix_from_vectors(target_normal, np.array([0, 0, 1]))
    transformed_points_at_seal_region = np.matmul(R, points_at_seal_region.T).T
    return transformed_points_at_seal_region
# def deflection_check(transformed_points_at_seal_region):
#     # R = rotation_matrix_from_vectors(target_normal, np.array([0, 0, 1]))
#     # transformed_points_at_seal_region = np.matmul(R, points_at_seal_region.T).T
#     seal_deflection = np.max(transformed_points_at_seal_region[:, 2]) - np.min(transformed_points_at_seal_region[:, 2])
#     # if seal_deflection < suction_area_deflection:
#     #     print(f'seal deflection = {seal_deflection}')
#     # else:
#     #     print(Fore.RED, f'seal deflection = {seal_deflection}', Fore.RESET)
#
#     return seal_deflection < suction_area_deflection
def seal_check_A(target_point,points_within_spherical_seal_region,visualize=False):
    # seal_test_points = circle_to_points(radius=suction_zone_radius, number_of_points=30, x=target_point[0],
    #                                     y=target_point[1], z=target_point[2])
    seal_cavity_points=circle_cavity_to_point(radius=suction_zone_radius, n_circles=15, x_center=target_point[0],
                                        y_center=target_point[1], z_center=target_point[2])

    xy_dist = np.linalg.norm(seal_cavity_points[:, np.newaxis, 0:2] - points_within_spherical_seal_region[np.newaxis, :, 0:2], axis=2)
    # print(xy_dist.shape)
    # print((seal_test_points[:, np.newaxis, 0:2] - points_within_spherical_seal_region[np.newaxis, :, 0:2]).shape)

    min_xy_dist = np.min(xy_dist, axis=1)
    # print(min_xy_dist.shape)
    # plt.plot(min_xy_dist)
    # plt.show()

    seal_deviation = np.max(min_xy_dist)
    if visualize:
        if seal_deviation < seal_ring_deviation:
            print(f'maximum seal deviation = {seal_deviation}')
        else:
            print(Fore.RED, f'maximum seal deviation = {seal_deviation}', Fore.RESET)

    return seal_deviation < seal_ring_deviation

# def seal_check_B(target_point,points_within_spherical_seal_region,visualize=False):
#     seal_test_points = circle_to_points(radius=suction_zone_radius, number_of_points=30, x=target_point[0],
#                                         y=target_point[1], z=target_point[2])
#     xyz_dist = np.linalg.norm(
#         seal_test_points[:, np.newaxis, 0:2] - points_within_spherical_seal_region[np.newaxis, :, 0:2],
#         axis=2)
#
#     nearest_in_xy = np.argmin(xyz_dist, axis=1)
#
#     z_dist = seal_test_points[:, 2] - points_within_spherical_seal_region[nearest_in_xy, 2]
#
#     grad = np.gradient(z_dist)
#     grad = np.abs(np.gradient(grad))
#
#
#     grad_max=np.max(grad)
#
#     c=0.002
#     if visualize:
#         if grad_max< c:
#             print(f'maximum change in seal distance = {grad_max}')
#         else:
#             print(Fore.RED, f'maximum change in seal distance  = {grad_max}', Fore.RESET)
#
#     return np.max(grad) < c

def get_suction_seal_loss(pc,normals,target_index,prediction_,statistics,spatial_mask,visualize=False):
    target_normal = normals[target_index]
    target_point=pc[target_index]
    shifted_pc=pc-target_point[np.newaxis]

    transformed_pc=transform_point_to_normal_in_plane(target_normal, shifted_pc)
    transformed_target_point=transformed_pc[target_index]

    '''mask suction region'''
    xyz_dist_ = np.linalg.norm(transformed_target_point[np.newaxis] - transformed_pc, axis=-1)

    spherical_mask = xyz_dist_ < suction_zone_radius

    '''circle to points'''
    points_within_spherical_seal_region = transformed_pc[spherical_mask]

    '''suction criteria'''
    criteria = seal_check_A(transformed_target_point, points_within_spherical_seal_region, visualize=visualize)

    '''suction seal loss'''
    if criteria:
        label = torch.ones_like(prediction_)
    else:
        label = torch.zeros_like(prediction_)

    if visualize:
        print(f'suction seal label= {label}, prediction = {prediction_}')
        # view_suction_area(pc, spherical_mask, target_point, target_normal, spatial_mask)
        view_suction_area(pc, spherical_mask, target_point, target_normal, spatial_mask)

    statistics.update_confession_matrix(label, prediction_.detach())

    return bce_loss(prediction_, label)