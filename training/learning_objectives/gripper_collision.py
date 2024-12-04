import torch
from torch import nn
import torch.nn.functional as F
from Configurations.config import theta_cos_scope
from lib.collision_unit import grasp_collision_detection
from lib.depth_map import pixel_to_point, transform_to_camera_frame, depth_to_point_clouds
from pose_object import pose_7_to_transformation
from registration import camera

bce_loss=nn.BCELoss()
l1_loss=nn.L1Loss()

def evaluate_grasps(batch_size,pixel_index,depth,generated_grasps,pose_7,visualize=False):
    '''Evaluate generated grasps'''
    collision_state_list = []
    firmness_state_list = []
    out_of_scope_list = []
    for j in range(batch_size):
        pix_A = pixel_index[j, 0]
        pix_B = pixel_index[j, 1]
        depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()

        '''get pose parameters'''
        target_point = pixel_to_point(pixel_index[j].cpu().numpy(), depth_value, camera)
        target_point = transform_to_camera_frame(target_point[None, :], reverse=True)[0]

        target_pose=generated_grasps[j, :, pix_A, pix_B]
        T_d, width, distance=pose_7_to_transformation(target_pose, target_point)
        # if j == 0: print(f'Example _pose = {generated_grasps[j, :, pix_A, pix_B]}')

        '''get point clouds'''
        pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
        pc = transform_to_camera_frame(pc, reverse=True)

        '''check collision'''
        collision_intensity = grasp_collision_detection(T_d,width, pc, visualize=visualize )
        collision_state_=collision_intensity > 0
        collision_state_list.append(collision_state_)

        '''check parameters are within scope'''
        ref_approach = torch.tensor([0.0, 0.0, 1.0],device=target_pose.device)  # vertical direction

        approach_cos = F.cosine_similarity(target_pose[0:3], ref_approach, dim=0)
        in_scope = 1.0 > generated_grasps[j, -2, pix_A, pix_B] > 0.0 and 1.0 > generated_grasps[
            j, -1, pix_A, pix_B] > 0.0 and approach_cos > theta_cos_scope
        out_of_scope_list.append(not in_scope)

        '''check firmness'''
        label_dist = pose_7[j, -2]
        generated_dist = generated_grasps[j, -2, pix_A, pix_B]
        firmness_=1 if generated_dist.item() > label_dist.item() and not collision_state_ and in_scope else 0
        firmness_state_list.append(firmness_)

    return collision_state_list,firmness_state_list,out_of_scope_list


def gripper_collision_loss(gripper_target_pose, gripper_target_point,pc,prediction_,statistics,visualize=False):
    T_d, width, distance = pose_7_to_transformation(gripper_target_pose, gripper_target_point)
    collision_intensity = grasp_collision_detection(T_d, width, pc, visualize=visualize)
    label = torch.zeros_like(prediction_) if collision_intensity > 0 else torch.ones_like(prediction_)
    statistics.update_confession_matrix(label, prediction_.detach())
    if visualize: print(f'gripper collision label ={label}, prediction= {prediction_}')
    return bce_loss(prediction_, label)