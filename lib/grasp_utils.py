import numpy as np
from Configurations import config


def update_pose_(T, pose_good_grasp=None,width=None,distance=None):
    # Note: distance is assumed to be incorporated in T
    if pose_good_grasp is None: pose_good_grasp=np.zeros((1,14))
    if width:pose_good_grasp[0,0]=width
    #rotation update
    rotation_matrix = T[:3, :3]
    rotation_matrix = rotation_matrix.T
    pose_good_grasp[0, [4, 5, 6, 7, 8, 9, 10, 11, 12]] = rotation_matrix.reshape(-1, 9)
    #position update
    center_point=T[:3,3]
    pose_good_grasp[0,[1,2,3]]=center_point
    if distance:pose_good_grasp[0, -1]=distance
    return pose_good_grasp

def remove_dist(T,distance):
    assert T.shape[-1]==4 and T.shape[-2]==4
    T[0:3, 3] = T[0:3, 3] - T[0:3, 0] * distance
    return T

def get_gripper_pose_primitives(pose_good_grasp):
    distance = pose_good_grasp[0, -1]
    width = pose_good_grasp[0, 0]
    T = get_homogenous_matrix(pose_good_grasp)
    T=remove_dist(T,distance)
    pose_good_grasp = update_pose_(T, width=-width, distance=-distance)
    center_point=pose_good_grasp[0, 1:4]
    R=T[ 0:3, 0:3]
    return distance, width, R, center_point
def shift_a_distance(matrix, step_distance):
    approach_vec=matrix[0:3,0]

    matrix[0:3,3]+=approach_vec*step_distance
    return matrix

def verfy_distance(pose_good_grasp, point, mean_point):
    grasp_posed = np.copy(pose_good_grasp)
    data_ = np.copy(point)
    x = grasp_posed[0, 1:4]
    x = x - mean_point.numpy()
    x = x - data_
    x = x / grasp_posed[0, 4:7]
    print('distance=', x)
    print(pose_good_grasp[0, -1])


def get_homogenous_matrix(pose_good_grasp):
    position = pose_good_grasp[0, [1, 2, 3]]
    matrix_ori = pose_good_grasp[0, [4, 5, 6, 7, 8, 9, 10, 11, 12]].reshape(-1, 3, 3)
    matrix_tmp = matrix_ori.transpose(0, 2, 1)
    T = np.eye(4)
    T[:3, :3] = matrix_tmp
    T[:3, 3] = position.T
    # The columns of the rotation part of the homogenous matrix T are:
        # Column 1: approach_vector
        # Column 2: closing_vector
        # Column 3: grasp_o_vector
    return T


def get_center_point(pose_good_grasp):
    T = get_homogenous_matrix(pose_good_grasp)
    center_point= T[0:3, 3] - T[0:3, 0] * pose_good_grasp[0, -1]
    return center_point

def get_grasp_pose( distance, width, R, center_point):
    T = np.zeros((4, 4))
    T[ 0:3, 0:3] = R
    T[0:3,3]=center_point
    T[3,3]=1
    T=shift_a_distance(T,distance)

    pose_good_grasp=update_pose_(T,width=width,distance=distance)

    return pose_good_grasp

def get_pose_matrixes(T,k_end_effector,k_pre_grasp):
    rot1 = [[0, 0, 1, -k_end_effector],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]]
    end_effecter_mat = np.dot(T, rot1) # [[],[],[],[]] matrix 4*4

    rot2 = [[0, 0, 1, -k_pre_grasp],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]]
    pre_grasp_mat = np.dot(T, rot2)

    return pre_grasp_mat, end_effecter_mat

def get_grasp_width(pose_good_grasp):
    # max width 1/22 equivelent to 25 after the transformation
    grasp_width = pose_good_grasp[:, 0].squeeze()
    grasp_width = grasp_width * config.width_scale
    if grasp_width > 25:
        grasp_width = 25
    elif grasp_width < 0:
        grasp_width = 0
    # print("grasp width", grasp_width)
    return grasp_width







def distance_based_reorder(results,grasp_score_pred,grasp_pose_pred,k=1.0):
    if k==0: return results
    results=np.asarray(results)
    idxs = results[:, 1].astype(int)
    sub_distance = grasp_pose_pred[0][idxs]

    sub_scores = grasp_score_pred[idxs]

    sub_scores = sub_scores*(1-k)+sub_scores * sub_distance*k
    new_order = np.argsort(-sub_scores)

    results = (results[new_order])


    results[:,1]=results[:,1].astype('int')


    return results.tolist()
