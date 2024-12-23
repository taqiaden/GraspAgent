import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from Configurations import config
from lib.grasp_utils import shift_a_distance, remove_dist
from lib.math_utils import asCartesian, rotation_matrix_from_vectors, asSpherical


def rotate_matrix_around_axis(rotation_degrees, rotation_axis=np.array([0, 0, 1])):
    rotation_radians = np.radians(rotation_degrees)

    rotation_vector = rotation_radians * rotation_axis

    rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
    return rotation

def apply_rotation_on_matrix(matrix, rotation_degree, rotation_axis=np.array([0, 0, 1])):
    if rotation_degree == 0.0: return matrix
    r = rotate_matrix_around_axis(rotation_degree, rotation_axis=rotation_axis)
    new_matrix = r.dot(matrix)

    return new_matrix

def transform_to_vec(mat, target_vec, ref_vec=np.array([.0, 0, 1.0])):
    if target_vec[0] == 0 and target_vec[1] == 0: return mat

    r = rotation_matrix_from_vectors(target_vec, ref_vec)

    mat = r.dot(mat)
    # print(mat)

    return mat

def angles_to_rotation_matrix(theta, phi, beta):
    standard_transformation = np.array([[0., 0., 1.],
                                        [0., 1., 0.],
                                        [-1., 0., 0.]])
    spherical_coordinate = np.array([1., theta, phi])

    matr = np.copy(standard_transformation)

    matr = apply_rotation_on_matrix(matrix=matr, rotation_degree=beta)

    target_vector = asCartesian(spherical_coordinate)
    target_vector = target_vector / np.linalg.norm(target_vector)
    matr = transform_to_vec(matr, target_vector)

    return matr

def rotation_matrix_to_angles(matr):
    standard_transformation = np.array([[0., 0., 1.],
                                        [0., 1., 0.],
                                        [-1., 0., 0.]])

    ref_vec = np.array([.0, 0., 1.0])
    arbitrary_vec = standard_transformation.dot(ref_vec)
    arbitrary_vec = arbitrary_vec / np.linalg.norm(arbitrary_vec)

    vec1 = standard_transformation.dot(arbitrary_vec)
    vec2 = matr.dot(arbitrary_vec)
    r = rotation_matrix_from_vectors(vec1, vec2)
    approach_vec = r.transpose().dot(ref_vec)
    spherical_coordinate = asSpherical(approach_vec)
    theta = spherical_coordinate[1]
    phi = spherical_coordinate[2]
    if phi < 0: phi = 360. + phi

    aligned_mat = r.transpose().dot(matr)
    found_r = aligned_mat.dot(standard_transformation.transpose())
    beta = ((Rotation.from_matrix(found_r).as_rotvec() / np.pi) * 180)[2]
    if beta < 0: beta = 180. + beta

    return theta, phi, beta

def rotation_from_vector(approach_vector, closing_vector):
    temp = -(approach_vector[:, 0] * closing_vector[:, 0] + approach_vector[:, 1] * closing_vector[:, 1]) / approach_vector[:, 2]
    closing_vector[:, 2] = temp
    closing_vector = torch.div(closing_vector.transpose(0, 1), torch.norm(closing_vector, dim=1)).transpose(0, 1)
    z_axis = torch.cross(approach_vector, closing_vector, dim = 1)
    R = torch.stack((approach_vector, closing_vector, z_axis), dim=-1)
    return R

def grasp_angle_to_vector(grasp_angle):
    device = grasp_angle.device
    x_ = torch.cos(grasp_angle/180*math.pi)
    y_ = torch.sin(grasp_angle/180*math.pi)
    z_ = torch.zeros(grasp_angle.shape[0]).to(device)
    closing_vector = torch.stack((x_, y_, z_), axis=0)
    return closing_vector


def encode_gripper_pose_2(distance, width, rotation_matrix):
    width = torch.tensor([width]) / config.width_scope
    distance = torch.tensor([distance]) / config.distance_scope
    theta, phi, beta = rotation_matrix_to_angles(rotation_matrix)
    theta= theta / config.theta_scope
    phi= phi / config.phi_scope
    beta= beta / config.beta_scope
    relative_pose_5=torch.tensor([theta,phi,beta,distance, width]).float()
    relative_pose_5 = relative_pose_5[None, :]
    return relative_pose_5

def transformation_to_relative_angle_form(T_d,distance,width):
    T_0=remove_dist(T_d,distance)
    R=T_0[ 0:3, 0:3]
    relative_pose_5=encode_gripper_pose_2(distance, width, R)
    return relative_pose_5

def construct_transformation(point, rotation):
    T = np.zeros((4, 4))
    T[0:3, 0:3] = rotation
    T[0:3, 3] = point
    T[3, 3] = 1
    return T

def unit_metrics_to_real_scope(relative_pose_5):
    theta2, phi2, beta_ratio =relative_pose_5[0],relative_pose_5[1],relative_pose_5[2]
    theta2=theta2*config.theta_scope
    phi2=phi2*config.phi_scope
    beta2=beta_ratio*config.beta_scope
    distance=relative_pose_5[-2]*config.distance_scope
    width=relative_pose_5[-1]
    width=np.clip(width,0.2,1)*config.width_scope
    return theta2,phi2,beta2,distance,width

def convert_angles_to_transformation_form(relative_pose_5,center_point):
    # shape of center_point is (3,)
    # shape of relative_pose_5 is torch.Size([5])

    '''clip and reshape'''
    assert relative_pose_5.shape==(1,5) or relative_pose_5.shape==(5,), f'{relative_pose_5.shape}'
    relative_pose_5=torch.clip(relative_pose_5,0.,1.)

    relative_pose_5=relative_pose_5.squeeze().detach().cpu().numpy()

    '''convert to real scope'''
    theta,phi,beta,distance,width=unit_metrics_to_real_scope(relative_pose_5)

    '''angles to rotation matrix'''
    rotation_matrix=angles_to_rotation_matrix(theta, phi, beta).reshape(3,3)

    '''transformation matrix'''
    T_0=construct_transformation(center_point, rotation_matrix)

    '''adjust the penetration distance for the transformation'''
    T_d = shift_a_distance(T_0, distance)
    assert T_d[0:3, 3].shape == center_point.shape,f'{T_d[0:3, 3].shape},  {center_point.shape}'
    assert T_d[0:3, 0:3].shape == rotation_matrix.shape

    return T_d,width,distance


if __name__ == "__main__":
    pass
