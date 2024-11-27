import torch
from torch import nn

from lib.collision_unit import grasp_collision_detection
from pose_object import pose_7_to_transformation

bce_loss=nn.BCELoss()


def gripper_collision_loss(gripper_target_pose, gripper_target_point,pc,prediction_,statistics,visualize=False):
    T_d, width, distance = pose_7_to_transformation(gripper_target_pose, gripper_target_point)
    collision_intensity = grasp_collision_detection(T_d, width, pc, visualize=visualize)
    label = torch.zeros_like(prediction_) if collision_intensity > 0 else torch.ones_like(prediction_)
    statistics.update_confession_matrix(label, prediction_)
    if visualize: print(f'gripper collision label ={label}, prediction= {prediction_}')
    return bce_loss(prediction_, label)