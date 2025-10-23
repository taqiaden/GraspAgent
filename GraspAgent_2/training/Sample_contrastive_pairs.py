import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from lib.IO_utils import custom_print
from lib.collision_unit import gripper_firmness_check
from lib.rl.masked_categorical import MaskedCategorical
from pose_object import pose_7_to_transformation
from training.learning_objectives.gripper_collision import evaluate_grasps3
from colorama import Fore

max_n = 50
print = custom_print

def check_valid_pair(c_, s_, f_, q_,annealing_factor,prediction_uniqeness):
    if sum(s_) > 0:
        if s_[1] == 0 and c_[1] == 0 and f_[1] > 0 and q_[1] > 0.5:
            return True, 1, 1
        else:
            return False,  0, 0
    elif sum(s_) == 0 and sum(c_) > 0:
        if c_[1] == 0 and f_[1] > 0. and q_[1] > 0.5 and c_[0]>0:
            return True, 1, 1
        elif c_[0] == 0 and f_[0] > 0. and q_[0] > 0.5 and c_[1]>0:
            # return False, 1, -1

            if np.random.rand()<prediction_uniqeness:
                return True,  1, -1
            else:
                return False, 0, 0
        else:
            return False,  0, 0

    elif sum(s_) == 0 and sum(c_) == 0:
        if (f_[1] - f_[0] > 0.) and q_[1] > 0.5:
            relative_firmness = (abs(f_[1] - f_[0]) / (f_[1] + f_[0]))
            if relative_firmness**2<np.random.rand(): return False,  0,0
            return True, relative_firmness, 1
        elif (f_[0] - f_[1] > 0.) and q_[0] > 0.5:
            relative_firmness = (abs(f_[1] - f_[0]) / (f_[1] + f_[0]))
            # return True, relative_firmness, -1

            if np.random.rand() < prediction_uniqeness:
                return True,   relative_firmness, -1
            else:
                return False, 0, 0
        else:
            return False,  0, 0
    else:
        return False, 0, 0

def view_grasp(point_data,target_generated_pose, target_point):
    from Configurations.ENV_boundaries import dist_allowance
    from lib.grasp_utils import shift_a_distance
    from lib.mesh_utils import construct_gripper_mesh

    T_d, width, distance = pose_7_to_transformation(target_generated_pose, target_point)

    T_d = shift_a_distance(T_d, -dist_allowance)
    mesh = construct_gripper_mesh(width.squeeze().detach().cpu().numpy(), T_d.detach().cpu().numpy())
    scene = trimesh.Scene()
    scene.add_geometry([trimesh.PointCloud(point_data.detach().cpu().numpy()), mesh])
    scene.show()

def compute_sampling_probability(sampling_centroid,gripper_pose_ref_PW,gripper_pose_PW,pc,bin_mask,grasp_quality):
    if sampling_centroid is None:
        selection_p = torch.rand_like(gripper_pose_PW[:,0])
    else:
        ref_gripper_pose2_ =gripper_pose_ref_PW.detach().clone()
        ref_gripper_pose2_[: ,5: ] =torch.clamp(ref_gripper_pose2_[: ,5:] ,0. ,1.)
        gripper_pose2_ =gripper_pose_PW.detach().clone()
        gripper_pose2_[: ,5: ] =torch.clamp(gripper_pose2_[: ,5:] ,0. ,1.)

        def norm_(gamma ,expo_=1.0,min=0.01):
            gamma = (gamma - gamma.min()) / (
                    gamma.max() - gamma.min())
            gamma = gamma ** expo_
            gamma=torch.clamp(gamma,min)
            return gamma

        # grasp_quality=torch.clamp(grasp_quality,0,1)
        # gamma_quality=norm_(grasp_quality,1)

        gamma_dive = norm_((1.001 - F.cosine_similarity(gripper_pose2_[: ,:5],
                                                        sampling_centroid[None, :5], dim=-1) ) /2 ,1)
        gamma_dive *= norm_((1.001 - F.cosine_similarity(ref_gripper_pose2_[: ,:5],
                                                        sampling_centroid[None, :5], dim=-1) ) /2 ,1)
        gamma_firmness =torch.clamp(ref_gripper_pose2_[: ,-2].detach().clone() ,0.001 ,0.99 )**0.3
        gamma_rand =torch.rand_like(gamma_dive)

        gamma_dive =norm_(gamma_dive)

        range =pc[~bin_mask][: ,-1].max( ) -pc[~bin_mask][: ,-1].min()

        d_gamma =norm_( 1 -torch.abs((pc[: ,-1 ] -pc[~bin_mask][: ,-1].min()).cuda( ) /range) ,2.0)


        selection_p = (gamma_dive *   gamma_rand * d_gamma *gamma_firmness) ** (1 / 5)


        assert not torch.isnan(selection_p).any(), f'selection_p: {selection_p}'

    return selection_p

def diversity_promotion_probability(sampling_centroid,gripper_pose_ref_PW,gripper_pose_PW):
    if sampling_centroid is None:
        selection_p = torch.rand_like(gripper_pose_PW[:,0])
    else:
        ref_gripper_pose2_ =gripper_pose_ref_PW.detach().clone()
        ref_gripper_pose2_[: ,5: ] =torch.clamp(ref_gripper_pose2_[: ,5:] ,0. ,1.)
        gripper_pose2_ =gripper_pose_PW.detach().clone()
        gripper_pose2_[: ,5: ] =torch.clamp(gripper_pose2_[: ,5:] ,0. ,1.)

        def norm_(gamma ,expo_=1.0,min=0.01):
            gamma = (gamma - gamma.min()) / (
                    gamma.max() - gamma.min())
            gamma = gamma ** expo_
            gamma=torch.clamp(gamma,min)
            return gamma



        gamma_dive = norm_((1.001 - F.cosine_similarity(gripper_pose2_[: ,:5],
                                                        sampling_centroid[None, :5], dim=-1) ) /2 ,1)
        gamma_dive *= norm_((1.001 - F.cosine_similarity(ref_gripper_pose2_[: ,:5],
                                                        sampling_centroid[None, :5], dim=-1) ) /2 ,1)


        gamma_dive =norm_(gamma_dive)

        selection_p = (gamma_dive ) ** (1 / 2)


        assert not torch.isnan(selection_p).any(), f'selection_p: {selection_p}'

    return selection_p

def sample_contrastive_pairs(  pc, mask, bin_mask, gripper_pose, gripper_pose_ref,
                             sampling_centroid,batch_size,annealing_factor,grasp_quality,superior_A_model_moving_rate):

    pairs = []

    selection_mask = ~bin_mask

    gripper_pose_PW=gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask]
    gripper_pose_ref_PW=gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :][mask]

    grasp_quality=grasp_quality.permute(0, 2, 3, 1)[0, :, :, 0][mask]
    max_=grasp_quality.max()
    min_=grasp_quality.min()
    grasp_quality=(grasp_quality-min_)/(max_-min_)

    # selection_p = compute_sampling_probability(sampling_centroid, gripper_pose_ref_PW, gripper_pose_PW, pc, bin_mask,grasp_quality)
    selection_p = torch.rand_like(gripper_pose_PW[:, 0])

    avaliable_iterations = selection_mask.sum()

    n = int(min(max_n, avaliable_iterations))

    counter = 0
    coords = torch.nonzero(mask, as_tuple=False)

    t = 0
    while True:
        if t > n: break


        dist = MaskedCategorical(probs=selection_p, mask=selection_mask)

        target_index = dist.sample().item()

        selection_mask[target_index] = False
        avaliable_iterations -= 1
        target_point = pc[target_index]

        target_generated_pose = gripper_pose_PW[target_index]
        target_ref_pose = gripper_pose_ref_PW[target_index]

        # print(target_ref_pose)

        prediction_uniqeness= (1- grasp_quality[target_index].item())*(target_generated_pose[-2].item()**0.5)
        # print(f'prediction_uniqeness {prediction_uniqeness}')
        c_, s_, f_, q_ = evaluate_grasps3(target_point, target_generated_pose, target_ref_pose, pc, bin_mask,
                                          visualize=False)

        counted ,margin,k =check_valid_pair(c_, s_, f_, q_,annealing_factor,prediction_uniqeness)

        if k==1:
            superior_A_model_moving_rate.update(0.)
        elif k==-1:
            superior_A_model_moving_rate.update(1.)


        if counted:
            '''improve firmness'''
            # if k>0:
            #     last_valid_pose=target_ref_pose.clone()
            #     last_valid_firmness_value=f_[1]
            #     distance_increment=0.15
            #     while True:
            #         target_ref_pose[-2]+=distance_increment
            #         T_d, width, distance = pose_7_to_transformation(target_ref_pose, target_point)
            #         detect_collision, pred_firmness_val, pred_quality, collision_val = gripper_firmness_check(T_d,
            #                                                                                                   width, pc[
            #                                                                                                       ~bin_mask],
            #                                                                                                   with_allowance=True,
            #                                                                                                   visualize=False)
            #         if detect_collision>0:
            #             f_[1]=last_valid_firmness_value
            #             gripper_pose_ref_PW[target_index]=last_valid_pose
            #             gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :][mask]=gripper_pose_ref_PW
            #             break
            #         else:
            #             last_valid_firmness_value = pred_firmness_val*pred_quality
            #             last_valid_pose = target_ref_pose.clone()

            counter+=1
            t = 1

            hh = (counter / batch_size) ** 2
            n = int(min(hh * max_n + n, avaliable_iterations))

            # if k>0:
            #     view_grasp(pc, target_generated_pose, target_point)
            # else:
            #     view_grasp(pc, target_ref_pose, target_point)

            pixel_index = coords[target_index]

            pairs.append((target_index, margin,k,pixel_index))

            superior_pose=target_ref_pose if k>0 else  target_generated_pose

            if sampling_centroid is None:
                sampling_centroid = superior_pose.detach().clone()
            else:
                sampling_centroid = sampling_centroid * 0.99 + superior_pose.detach().clone() * 0.01

            if counter==batch_size:break

        t += 1

    if counter== batch_size:
        return True, pairs, sampling_centroid
    else:
        return False, pairs, sampling_centroid


def sample_contrastive_pairs1d(  pc,  bin_mask, gripper_pose_PW, gripper_pose_ref_PW,
                             sampling_centroid,batch_size,annealing_factor,grasp_quality,superior_A_model_moving_rate):

    pairs = []

    selection_mask = ~bin_mask


    max_=grasp_quality.max()
    min_=grasp_quality.min()
    grasp_quality=(grasp_quality-min_)/(max_-min_)

    selection_p = diversity_promotion_probability(sampling_centroid, gripper_pose_ref_PW, gripper_pose_PW)
    # selection_p = torch.rand_like(gripper_pose_PW[:, 0])

    avaliable_iterations = selection_mask.sum()

    n = int(min(max_n, avaliable_iterations))

    counter = 0
    # coords = torch.nonzero(mask, as_tuple=False)

    t = 0
    while True:
        if t > n: break


        dist = MaskedCategorical(probs=selection_p, mask=selection_mask)

        target_index = dist.sample().item()

        selection_mask[target_index] = False
        avaliable_iterations -= 1
        target_point = pc[target_index]

        target_generated_pose = gripper_pose_PW[target_index]
        target_ref_pose = gripper_pose_ref_PW[target_index]

        # print(target_ref_pose)

        prediction_uniqeness= ((1- grasp_quality[target_index].item())**2)*(target_generated_pose[-2].item()**0.5)
        # print(f'prediction_uniqeness {prediction_uniqeness}')
        c_, s_, f_, q_ = evaluate_grasps3(target_point, target_generated_pose, target_ref_pose, pc, bin_mask,
                                          visualize=False)

        counted ,margin,k =check_valid_pair(c_, s_, f_, q_,annealing_factor,prediction_uniqeness)

        if k==1:
            superior_A_model_moving_rate.update(0.)
        elif k==-1:
            superior_A_model_moving_rate.update(1.)


        if counted:
            '''improve firmness'''
            # if k>0:
            #     last_valid_pose=target_ref_pose.clone()
            #     last_valid_firmness_value=f_[1]
            #     distance_increment=0.15
            #     while True:
            #         target_ref_pose[-2]+=distance_increment
            #         T_d, width, distance = pose_7_to_transformation(target_ref_pose, target_point)
            #         detect_collision, pred_firmness_val, pred_quality, collision_val = gripper_firmness_check(T_d,
            #                                                                                                   width, pc[
            #                                                                                                       ~bin_mask],
            #                                                                                                   with_allowance=True,
            #                                                                                                   visualize=False)
            #         if detect_collision>0:
            #             f_[1]=last_valid_firmness_value
            #             gripper_pose_ref_PW[target_index]=last_valid_pose
            #             gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :][mask]=gripper_pose_ref_PW
            #             break
            #         else:
            #             last_valid_firmness_value = pred_firmness_val*pred_quality
            #             last_valid_pose = target_ref_pose.clone()

            counter+=1
            t = 1

            hh = (counter / batch_size) ** 2
            n = int(min(hh * max_n + n, avaliable_iterations))

            # if k>0:
            #     view_grasp(pc, target_generated_pose, target_point)
            # else:
            #     view_grasp(pc, target_ref_pose, target_point)

            # pixel_index = coords[target_index]

            pairs.append((target_index, margin,k,target_index))

            superior_pose=target_ref_pose if k>0 else  target_generated_pose

            if sampling_centroid is None:
                sampling_centroid = superior_pose.detach().clone()
            else:
                sampling_centroid = sampling_centroid * 0.99 + superior_pose.detach().clone() * 0.01

            if counter==batch_size:break

        t += 1

    if counter== batch_size:
        return True, pairs, sampling_centroid
    else:
        return False, pairs, sampling_centroid

