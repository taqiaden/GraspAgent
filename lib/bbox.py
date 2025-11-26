import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import torch.nn.functional as F

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

def rotate_around_z_axis(matrix, beta):
    """
    Rotate a matrix beta degrees around the z-axis [0,0,1]

    Args:
        matrix: Input matrix (3x3 tensor)
        beta: Rotation angle in 2d vector form

    Returns:
        Rotated matrix
    """

    # Convert degrees to radians
    beta_rad = beta * math.pi / 180.0

    # Create rotation matrix around z-axis
    cos_beta = torch.cos(beta_rad)
    sin_beta = torch.sin(beta_rad)

    zero = torch.zeros((), device=cos_beta.device, dtype=cos_beta.dtype)
    one = torch.ones((), device=cos_beta.device, dtype=cos_beta.dtype)

    rotation_matrix = torch.stack([
        torch.stack([cos_beta, -sin_beta, zero]),
        torch.stack([sin_beta, cos_beta, zero]),
        torch.stack([zero, zero, one])
    ])

    # Apply rotation: R * M
    rotated_matrix = rotation_matrix @ matrix

    return rotated_matrix

def transform(matrix, beta_degrees,vector1):
    """
    Rotate a matrix beta degrees around the z-axis [0,0,1]
    then transform in the direction from [0,0,1] to vector1
    Args:
        matrix: Input matrix (3x3 tensor)
        beta_degrees: Rotation angle in degrees
        vector1: approach vector
    Returns:
        Rotated matrix
    """
    # Convert degrees to radians
    beta_rad = torch.tensor(beta_degrees * math.pi / 180.0)

    # Create rotation matrix around z-axis
    cos_beta = torch.cos(beta_rad)
    sin_beta = torch.sin(beta_rad)

    rotation_matrix = torch.tensor([
        [cos_beta, -sin_beta, 0.0],
        [sin_beta, cos_beta, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=matrix.dtype)

    # Apply rotation: R * M
    rotation_matrix = rotation_matrix @ matrix


    vector2=torch.tensor([.0, 0, 1.0]).float()
    # Normalize vectors
    v1 = F.normalize(vector1, dim=0, eps=1e-8)
    v2 = F.normalize(vector2, dim=0, eps=1e-8)

    # Calculate quaternion representing the rotation from v1 to v2
    dot = torch.dot(v1, v2)

    if dot > 0.999999:
        # Identity quaternion
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=vector1.dtype, device=vector1.device)
    elif dot < -0.999999:
        # 180 degree rotation - need to find axis
        axis = torch.cross(torch.tensor([1.0, 0.0, 0.0], device=vector1.device), v1)
        if torch.norm(axis) < 1e-8:
            axis = torch.cross(torch.tensor([0.0, 1.0, 0.0], device=vector1.device), v1)
        axis = F.normalize(axis, dim=0, eps=1e-8)
        quat = torch.tensor([0.0, axis[0], axis[1], axis[2]], dtype=vector1.dtype, device=vector1.device)
    else:
        # General case
        axis = torch.cross(v1, v2)
        quat = torch.cat([
            torch.sqrt((1 + dot) / 2).unsqueeze(0),
            axis / (2 * torch.sqrt((1 + dot) / 2))
        ])
        quat = F.normalize(quat, dim=0, eps=1e-8)

    # Convert quaternion to rotation matrix
    qw, qx, qy, qz = quat
    R = torch.tensor([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]
    ], dtype=vector1.dtype, device=vector1.device)

    # Apply rotation to the input rotation matrix
    return R @ rotation_matrix



def transform_to_vec(mat, target_vec, ref_vec=np.array([.0, 0, 1.0])):
    if target_vec[0] == 0 and target_vec[1] == 0: return mat

    r = rotation_matrix_from_vectors(target_vec, ref_vec)

    mat = r.dot(mat)
    # print(mat)

    return mat


def apply_rotation_quaternion(vector1, vector2, rotation_matrix):
    """
    Differentiable & continuous quaternion-based rotation
    Rotates from vector1 to vector2 and applies to rotation_matrix.
    """
    # Normalize input vectors
    v1 = F.normalize(vector1, dim=0, eps=1e-8)
    v2 = F.normalize(vector2, dim=0, eps=1e-8)

    dot = torch.clamp(torch.dot(v1, v2), -1.0, 1.0)  # numerical safety

    # Compute quaternion smoothly
    w = torch.sqrt((1.0 + dot) * 0.5 + 1e-8)  # scalar part
    xyz = torch.cross(v1, v2)
    xyz = xyz / (2.0 * w + 1e-8)              # vector part
    quat = torch.cat([w.unsqueeze(0), xyz])   # (4,)

    quat = F.normalize(quat, dim=0, eps=1e-8) # ensure unit quaternion

    # Convert quaternion -> rotation matrix
    qw, qx, qy, qz = quat
    R = torch.stack([
        torch.stack([1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)]),
        torch.stack([2*(qx*qy + qz*qw),         1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)]),
        torch.stack([2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)])
    ])

    return R @ rotation_matrix


def rotation_from_xxy_z(xy_proj: torch.Tensor, z_new: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Construct a rotation matrix given:
        - xy_proj: 2D vector [cosθ, sinθ] representing the projection of new x-axis in XY plane
        - z_new: 3D vector representing the new z-axis
    Returns:
        R: 3x3 rotation matrix
    """
    # normalize z_new
    z_new = z_new / (z_new.norm() + eps)

    # lift xy_proj to 3D and normalize
    x_proj = torch.tensor([xy_proj[0], xy_proj[1], 0.0], dtype=z_new.dtype, device=z_new.device)
    x_proj = x_proj / (x_proj.norm() + eps)

    # make x_new orthogonal to z_new
    x_new = x_proj - (x_proj @ z_new) * z_new
    x_new = x_new / (x_new.norm() + eps)

    # y-axis via right-hand rule
    y_new = torch.cross(z_new, x_new)
    y_new = y_new / (y_new.norm() + eps)

    # construct rotation matrix (columns = axes)
    R = torch.stack([x_new, y_new, z_new], dim=1)
    return R


def angles_to_rotation_matrix(approach,beta):
    standard_transformation = torch.tensor([[0., 0., 1.],
                                        [0., 1., 0.],
                                        [-1., 0., 0.]],device=approach.device).float()

    R=rotate_around_z_axis(standard_transformation,beta)

    matr=apply_rotation_quaternion(approach, torch.tensor([.0, 0, 1.0],device=approach.device).float(),R)


    return matr
def angles_to_rotation_matrix_old(theta, phi, beta):
    standard_transformation = np.array([[0., 0., 1.],
                                        [0., 1., 0.],
                                        [-1., 0., 0.]])
    matr = np.copy(standard_transformation)


    matr = apply_rotation_on_matrix(matrix=matr, rotation_degree=beta)

    spherical_coordinate = np.array([1., theta, phi])
    target_vector = asCartesian(spherical_coordinate)
    target_vector = target_vector / np.linalg.norm(target_vector)

    matr = transform_to_vec(matr,target_vector)

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
    T = torch.zeros((4, 4),device=rotation.device)
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
    width=np.clip(width,0.0,1)*config.width_scope
    return theta2,phi2,beta2,distance,width
import torch

def rotation_matrix_from_normal_target(normal: torch.Tensor, target_point: torch.Tensor):
    """
    normal: [3] tensor
    target_point: [3] tensor
    returns: [4,4] transformation matrix (rotation + translation)
    """
    v0 = torch.tensor([1.0, 0.0, 0.0], device=normal.device)

    # Ensure normal is normalized
    normal = normal / normal.norm()

    # Compute rotation axis and angle
    axis = torch.cross(v0, -normal)
    axis_norm = axis.norm()
    if axis_norm < 1e-8:  # v0 and -normal are collinear
        R = torch.eye(3, device=normal.device)
        if torch.dot(v0, -normal) < 0:
            # 180 degree rotation
            R = -R
    else:
        axis = axis / axis_norm
        angle = torch.acos(torch.clamp(torch.dot(v0, -normal), -1.0, 1.0))

        # Rodrigues' rotation formula
        K = torch.tensor([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], device=normal.device)
        R = torch.eye(3, device=normal.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

    # Build homogeneous transformation
    T = torch.eye(4, device=normal.device)
    T[:3, :3] = R
    T[:3, 3] = target_point
    return T



# def convert_angles_to_transformation_form(pose_7,center_point,approach=None):
#     # shape of center_point is (3,)
#     # shape of relative_pose_5 is torch.Size([5])
#
#     '''clip and reshape'''
#     # assert relative_pose_5.shape==(1,5) or relative_pose_5.shape==(5,), f'{relative_pose_5.shape}'
#     # relative_pose_5=torch.clip(relative_pose_5,0.,1.)
#
#     # pose_7[0:3]=torch.tensor([0.,1,1])
#     # print(pose_7[0:3])
#     # relative_pose_5 = pose_7_to_pose_5(pose_7)
#
#     beta= angle_ratio_from_sin_cos(pose_7[ 3:4], pose_7[ 4:5])*config.beta_scope
#     distance = pose_7[ 5:6 ]*config.distance_scope
#     width = pose_7[6:7]*config.width_scope
#
#     approach=approach
#
#
#     '''angles to rotation matrix'''
#     rotation_matrix=angles_to_rotation_matrix(approach,beta)
#
#     '''transformation matrix'''
#     T_0=construct_transformation(center_point, rotation_matrix)
#
#     '''adjust the penetration distance for the transformation'''
#     # print(distance)
#     T_d = shift_a_distance(T_0, distance)
#     assert T_d[0:3, 3].shape == center_point.shape,f'{T_d[0:3, 3].shape},  {center_point.shape}'
#     assert T_d[0:3, 0:3].shape == rotation_matrix.shape
#
#     return T_d,width,distance


if __name__ == "__main__":
    pass
