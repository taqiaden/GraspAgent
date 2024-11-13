import numpy as np


def remove_dist(T,distance):
    assert T.shape[-1]==4 and T.shape[-2]==4
    T[0:3, 3] = T[0:3, 3] - T[0:3, 0] * distance
    return T


def shift_a_distance(T, step_distance):
    approach_vec=T[0:3,0]
    T[0:3,3]+=approach_vec*step_distance
    return T


def get_target_point_2(T,distance):
    approach=T[0:3, 0]
    transition=T[0:3, 3]
    center_point= transition - approach * distance
    return center_point


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
