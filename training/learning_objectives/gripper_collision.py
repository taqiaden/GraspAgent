import torch
import torch.nn.functional as F
from torch import nn
from Configurations.config import theta_cos_scope
from lib.collision_unit import grasp_collision_detection, gripper_firmness_check
from lib.depth_map import pixel_to_point, transform_to_camera_frame, depth_to_point_clouds
from pose_object import pose_7_to_transformation
from registration import camera

bce_loss=nn.BCELoss()
l1_loss=nn.L1Loss()

ref_approach = torch.tensor([0.0, 0.0, 1.0], device='cuda')  # vertical direction


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

        target_generated_pose=generated_grasps[j, :, pix_A, pix_B]
        T_d, width, distance=pose_7_to_transformation(target_generated_pose, target_point)
        # if j == 0: print(f'Example _pose = {generated_grasps[j, :, pix_A, pix_B]}')

        '''get point clouds'''
        pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
        pc = transform_to_camera_frame(pc, reverse=True)

        '''check collision'''
        has_collision,pred_firmness_val,collision_val = gripper_firmness_check(T_d,width, pc, visualize=visualize )
        collision_state_list.append(has_collision)

        '''check firmness of the label'''
        T_d_label, width_label, distance_label = pose_7_to_transformation(pose_7[j], target_point)
        _, label_firmness_val,_ = gripper_firmness_check(T_d_label, width_label, pc, visualize=visualize)

        '''check parameters are within scope'''

        approach_cos = F.cosine_similarity(target_generated_pose[0:3], ref_approach, dim=0)
        in_scope = 1.0 > generated_grasps[j, -2, pix_A, pix_B] > 0.0 and 1.0 > generated_grasps[
            j, -1, pix_A, pix_B] > 0.0 and approach_cos > theta_cos_scope

        out_of_scope_list.append(not in_scope)

        '''check firmness'''
        firmness_state_list.append(int(pred_firmness_val>label_firmness_val))

    return collision_state_list,firmness_state_list,out_of_scope_list

def scope_drift_intensity(pose_7):

    approach_cos = F.cosine_similarity(pose_7[0:3], ref_approach, dim=0)

    k= [max(0, theta_cos_scope - approach_cos.item()), max(0, -pose_7[-2].item()), max(0, pose_7[-2].item() - 1), max(0, -pose_7[-1].item()),
        max(0, pose_7[-1].item() - 1)]

    return sum(k)
def evaluate_grasps3(target_point,target_generated_pose,target_ref_pose,pc,visualize=False):

    gen_out_of_scope=scope_drift_intensity(target_generated_pose)
    ref_out_of_scope=scope_drift_intensity(target_ref_pose)

    if gen_out_of_scope==0:
        '''check collision'''

        T_d, width, distance = pose_7_to_transformation(target_generated_pose, target_point)
        gen_has_collision,pred_firmness_val,collision_val = gripper_firmness_check(T_d,width, pc, visualize=visualize )
        # print(gen_has_collision,pred_firmness_val,collision_val)

    else:
        gen_has_collision, pred_firmness_val, collision_val = 1, 0, 1

    if ref_out_of_scope==0:
        '''check firmness of the label'''
        T_d_label, width_label, distance_label = pose_7_to_transformation(target_ref_pose, target_point)
        ref_has_collision,ref_firmness_val,ref_collision_val = gripper_firmness_check(T_d_label, width_label, pc, visualize=visualize)
        # print(ref_has_collision,ref_firmness_val,ref_collision_val)
        # if ref_has_collision==0:
        #     print('ref: ', target_ref_pose)
        #     print('G: ', target_generated_pose)
    else:
        ref_has_collision, ref_firmness_val, ref_collision_val=1,0,1
    #
    # print('ref: ', target_ref_pose)
    # print('G: ', target_generated_pose)


    return (collision_val,ref_collision_val), (gen_out_of_scope,ref_out_of_scope),(pred_firmness_val,ref_firmness_val)



def evaluate_grasps2(batch_size,pixel_index,depth,pcs,generated_grasps,gripper_pose_ref,visualize=False):
    '''Evaluate generated grasps'''
    collision_state_list = []
    firmness_state_list = []
    out_of_scope_list = []
    for j in range(batch_size):
        pix_A = pixel_index[j, 0]
        pix_B = pixel_index[j, 1]
        depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()
        pc=pcs[j]

        '''get pose parameters'''
        target_point = pixel_to_point(pixel_index[j].cpu().numpy(), depth_value, camera)
        target_point = transform_to_camera_frame(target_point[None, :], reverse=True)[0]

        target_generated_pose=generated_grasps[j, :, pix_A, pix_B]
        target_ref_pose=gripper_pose_ref[j, :, pix_A, pix_B]

        gen_out_of_scope=scope_drift_intensity(target_generated_pose)
        ref_out_of_scope=scope_drift_intensity(target_ref_pose)

        if gen_out_of_scope==0:
            '''check collision'''
            T_d, width, distance=pose_7_to_transformation(target_generated_pose, target_point)
            gen_has_collision,pred_firmness_val,collision_val = gripper_firmness_check(T_d,width, pc, visualize=visualize )
        else:
            gen_has_collision, pred_firmness_val, collision_val=1,0,1

        if ref_out_of_scope==0:
            '''check firmness of the label'''
            T_d_label, width_label, distance_label = pose_7_to_transformation(target_ref_pose, target_point)
            ref_has_collision,ref_firmness_val,ref_collision_val = gripper_firmness_check(T_d_label, width_label, pc, visualize=visualize)
        else:
            ref_has_collision, ref_firmness_val, ref_collision_val=1,0,1

        collision_state_list.append((gen_has_collision,ref_has_collision))

        out_of_scope_list.append((gen_out_of_scope,ref_out_of_scope))

        '''check firmness'''
        firmness_state_list.append((pred_firmness_val,ref_firmness_val))

    return collision_state_list,firmness_state_list,out_of_scope_list


def gripper_collision_loss(gripper_target_pose, gripper_target_point,pc,objects_mask,prediction_,objects_collision_statistics,bin_collision_statistics,visualize=False):
    T_d, width, distance = pose_7_to_transformation(gripper_target_pose, gripper_target_point)
    collision_with_objects = grasp_collision_detection(T_d, width, pc[objects_mask], visualize=visualize)
    collision_with_bin= grasp_collision_detection(T_d, width, pc[~objects_mask], visualize=visualize)

    object_collision_label = torch.ones_like(prediction_[0]) if collision_with_objects  else torch.zeros_like(prediction_[0])
    bin_collision_label = torch.ones_like(prediction_[1]) if collision_with_bin  else torch.zeros_like(prediction_[1])
    object_collision_pred=prediction_[0]
    bin_collision_pred=prediction_[1]

    objects_collision_statistics.update_confession_matrix(object_collision_label, object_collision_pred)
    bin_collision_statistics.update_confession_matrix(bin_collision_label, bin_collision_pred)

    if visualize: print(f'bin collision (label ={bin_collision_label}, prediction= {bin_collision_pred}),  objects collision (label ={object_collision_label}, prediction= {object_collision_pred})')
    object_collision_loss=bce_loss(object_collision_pred, object_collision_label)
    bin_collision_loss=bce_loss(bin_collision_pred, bin_collision_label)

    with torch.no_grad():
        objects_collision_statistics.loss=object_collision_loss.item()
        bin_collision_statistics.loss=bin_collision_loss.item()

    return (object_collision_loss+bin_collision_loss)/2