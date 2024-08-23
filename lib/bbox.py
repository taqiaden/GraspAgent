import numpy as np
import torch
import math

from scipy.spatial.transform import Rotation

from Configurations import config
from lib.grasp_utils import get_gripper_pose_primitives, shift_a_distance, update_pose_

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
    # test
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

    pose=torch.tensor([theta,phi,beta,distance, width]).float()
    pose = pose[None, :]

    return pose
def encode_gripper_pose(pose_good_grasp):

    distance, width, rotation_matrix, center_point = get_gripper_pose_primitives(pose_good_grasp)

    return encode_gripper_pose_2(distance, width, rotation_matrix)

def decode_gripper_pose(output,center_point):
    assert output.shape==(1,5) or output.shape==(5,)

    output=torch.clip(output,0.,1.)
    output=output.squeeze().detach().cpu().numpy()
    # rotation_matrix=rotation_matrix.squeeze().detach().cpu().numpy()
    # dist_and_width=dist_and_width.squeeze().detach().cpu().numpy()
    # rotation_matrix = Rotation.from_quat(output[0:4]).as_matrix()
    theta2, phi2, beta_ratio =output[0],output[1],output[2]
    theta2=theta2*config.theta_scope
    phi2=phi2*config.phi_scope
    beta2=beta_ratio*config.beta_scope

    rotation_matrix=angles_to_rotation_matrix(theta2, phi2, beta2)

    rotation_matrix=rotation_matrix.reshape(3,3)

    distance=output[-2]*config.distance_scope
    width=output[-1]
    width=np.clip(width,0.2,1)*config.width_scope

    # print(R)

    T = np.zeros((4, 4))
    T[0:3, 0:3] = rotation_matrix
    # print(center_point.shape)
    T[0:3, 3] = center_point
    T[3, 3] = 1
    T = shift_a_distance(T, distance)
    assert T[0:3, 3].shape == center_point.shape,f'{T[0:3, 3].shape},  {center_point.shape}'
    assert T[0:3, 0:3].shape == rotation_matrix.shape

    pose_good_grasp = update_pose_(T, width=width, distance=distance)

    return pose_good_grasp

if __name__ == "__main__":
    pass
