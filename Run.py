# -*- coding: utf-8 -*-
import math
import torch
from colorama import Fore
from Configurations.run_config import simulation_mode, skip_factor, \
    report_result, isvis, view_grasp_suction_points, max_suction_candidates, max_grasp_candidates, \
    view_score_gradient, chances_ref, \
    shuffling_probability, suction_limit, gripper_limit, suction_factor, gripper_factor
from grasp_post_processing import gripper_processing, suction_processing
from lib.ROS_communication import  wait_for_feedback
from lib.dataset_utils import configure_smbclient
from lib.depth_map import transform_to_camera_frame, point_clouds_to_depth, depth_to_point_clouds, depth_to_mesh_grid
from lib.models_utils import initialize_model, initialize_model_state
from Configurations import config
import subprocess

from masks import initialize_masks
import time
import numpy as np
from lib.image_utils import check_image_similarity
from models.Grasp_GAN import gripper_sampler_net, gripper_sampler_path
from models.gripper_quality import gripper_quality_net, gripper_quality_model_state_path, \
    gripper_scope_model_state_path
from models.point_net_base.gripper_D import gripper_discriminator, dense_gripper_discriminator_path
from models.point_net_base.suction_D import affordance_net, affordance_net_model_path
from models.scope_net import scope_net_vanilla
from models.suction_quality import suction_quality_net, suction_quality_model_state_path, \
    suction_scope_model_state_path
from models.suction_sampler import suction_sampler_net, suction_sampler_model_state_path
from pose_object import  vectors_to_ratio_metrics
from process_perception import get_new_perception, get_side_bins_images, get_real_data
from registration import camera
from visualiztion import visualize_grasp_and_suction_points, \
    view_score

execute_suction_bash = './bash/run_robot_suction.sh'
execute_grasp_bash = './bash/run_robot_grasp.sh'

configure_smbclient()

poses=None
normals=None

gripper_D_model=None
suction_D_model=None
gripper_G_model=None
Suction_G_model=None
suction_scope_model=None
gripper_scope_model=None

def point_net_quality_inference(depth,normals_pixels,poses_pixels):
    '''pointnet processing'''
    voxel_pc, mask = depth_to_point_clouds(depth, camera)
    voxel_pc = transform_to_camera_frame(voxel_pc, reverse=True)
    choices = np.random.choice(voxel_pc.shape[0], 50000, replace=False)
    down_sampled_pc=voxel_pc[choices,:]
    down_sampled_pc=torch.from_numpy(down_sampled_pc).to('cuda')[None,...].float()
    down_sampled_normals=normals_pixels.squeeze().permute(1,2,0)[mask][choices,...][None,...]
    down_sampled_poses=poses_pixels.squeeze().permute(1,2,0)[mask][choices,...][None,...].permute(0,2,1)
    pc_with_normals=torch.cat([down_sampled_pc,down_sampled_normals],dim=-1).float()
    suction_quality_score = suction_D_model(pc_with_normals.clone())
    gripper_quality_score = gripper_D_model(down_sampled_pc.clone(),down_sampled_poses.clone())
    suction_quality_score = suction_quality_score.detach().cpu().numpy()
    gripper_quality_score = gripper_quality_score.detach().cpu().numpy()
    voxel_pc=down_sampled_pc.squeeze().detach().cpu().numpy()
    normals = down_sampled_normals.squeeze()
    poses = down_sampled_poses.squeeze().permute(1,0)

    return gripper_quality_score,suction_quality_score,normals,poses,voxel_pc


def get_coordinates_from_pixels(batch_size=1):
    xymap = depth_to_mesh_grid(camera)
    xymap = xymap.repeat(batch_size, 1, 1, 1)
    return xymap

def model_inference(pc,max_grasp_candidates_,
                             max_suction_candidates_,view_grasp_suction=False,mask_inversion=False):
    global normals
    global poses
    global gripper_D_model
    global gripper_G_model
    global suction_D_model
    global Suction_G_model
    global suction_scope_model
    global gripper_scope_model

    '''get depth'''
    transformed_pc = transform_to_camera_frame(pc)
    depth = point_clouds_to_depth(transformed_pc, camera)
    depth_torch=torch.from_numpy(depth)[None,None,...].to('cuda').float()

    '''sample suction'''
    normals_pixels=Suction_G_model(depth_torch.clone())

    '''sample gripper'''
    poses_pixels=gripper_G_model(depth_torch.clone())

    '''suction scope'''
    voxel_pc, mask = depth_to_point_clouds(depth, camera)
    voxel_pc = transform_to_camera_frame(voxel_pc, reverse=True)
    positions=torch.from_numpy(voxel_pc).to('cuda').float()
    normals = normals_pixels.squeeze().permute(1, 2, 0)[mask]
    suction_scope=suction_scope_model(torch.cat([positions,normals],dim=-1)).squeeze().detach().cpu().numpy()
    suction_scope=np.clip(suction_scope,0,1)
    # suction_scope[suction_scope>=0.3]=1.
    # suction_scope[suction_scope<0.3]=0.0

    '''gripper scope'''
    # gripper_scope=gripper_scope_model(poses_pixels[:,0:3,...].clone())

    '''suction quality'''
    suction_quality_score=suction_D_model(depth_torch.clone(),normals_pixels.clone()).squeeze()

    '''gripper quality'''
    gripper_quality_score=gripper_D_model(depth_torch.clone(),poses_pixels.clone()).squeeze()

    '''res u net processing'''
    voxel_pc, mask = depth_to_point_clouds(depth, camera)
    voxel_pc = transform_to_camera_frame(voxel_pc, reverse=True)
    suction_quality_score=suction_quality_score[mask].detach().cpu().numpy()
    gripper_quality_score=gripper_quality_score[mask].detach().cpu().numpy()
    normals=normals_pixels.squeeze().permute(1,2,0)[mask]
    poses=poses_pixels.squeeze().permute(1,2,0)[mask]

    '''pointnet processing'''
    # gripper_quality_score,suction_quality_score,normals,poses,voxel_pc=point_net_quality_inference(depth,normals_pixels,poses_pixels)

    '''final scores'''
    # suction_scores=suction_quality_score*suction_scope
    suction_scores=suction_scope
    gripper_scores=gripper_quality_score+1#*gripper_scope
    suction_scores=suction_scores.squeeze()
    gripper_scores=gripper_scores.squeeze()


    '''set limits'''
    suction_scores[suction_scores < suction_limit] = -1
    gripper_scores[gripper_scores<gripper_limit]=-1
    suction_scores=suction_scores*suction_factor
    gripper_scores=gripper_scores*gripper_factor


    '''post processing'''
    best_gripper_scores=gripper_scores>1.0
    suction_scores[best_gripper_scores]=-1

    # if view_sampled_suction:
    #     target_mask = get_spatial_mask(data)
    #     estimate_suction_direction(data.cpu().numpy().squeeze(), view=True, view_mask=target_mask,score=suction_score_pred)
    # if view_sampled_griper:
    #     dense_grasps_visualization(pc_with_normals.clone(), poses.clone(),grasp_score_pred.clone(),target_mask)
    # suction_score_pred[suction_score_pred < suction_limit] = -1
    # grasp_score_pred[grasp_score_pred<gripper_limit]=-1

    poses= vectors_to_ratio_metrics(poses)
    normals=normals.detach().cpu().numpy()


    grasp_score_pred_mask, suction_score_pred_mask = initialize_masks(gripper_scores, suction_scores, voxel_pc,
                                                                      max_grasp_candidates_,max_suction_candidates_,mask_inversion)

    if view_grasp_suction: visualize_grasp_and_suction_points(suction_score_pred_mask, grasp_score_pred_mask,
                                                                     voxel_pc)
    if view_score_gradient:
        view_score(voxel_pc, grasp_score_pred_mask, gripper_scores)
        view_score(voxel_pc, suction_score_pred_mask, suction_scores)

    index_suction_cls = [i for i in range(len(suction_score_pred_mask)) if suction_score_pred_mask[i]]
    index_grasp_cls = [j for j in range(len(grasp_score_pred_mask)) if grasp_score_pred_mask[j]]


    score_grasp_value_ind = list((item * 1, index_grasp_cls[index], 0) for index, item in
                                 enumerate(gripper_scores[grasp_score_pred_mask]))
    score_suction_value_ind_ = list((item * 1, index_suction_cls[index], 1) for index, item in
                                    enumerate(suction_scores[suction_score_pred_mask]))

    candidates = score_grasp_value_ind + score_suction_value_ind_

    if len(candidates)>1:
        candidates = sorted(candidates, reverse=True)

    return candidates,gripper_scores,suction_scores,voxel_pc

def run_robot( point_data, candidates,grasp_score_pred,suction_score_pred,isvis):

    actions, states, data = [], [], []
    state_ = ''
    if len(candidates) == 0:
        actions.append(state_)
        states.append('No Object')
        data.append((0, 0, np.eye(4), [1, 0, 0], [0,0,0]))
        return actions,states,data
    # global chances_ref
    chances=chances_ref

    print(f'Total number of candidates = {len(candidates)}')
    if len(candidates)>100:
        if shuffling_probability==0.0:
            chunk_size=math.floor(0.01*len(candidates))
        else:
            chunk_size=math.floor(0.1*len(candidates))
    else:
        chunk_size=100

    for i in range(0,len(candidates),chunk_size):

        end_index = min(i + chunk_size, len(candidates))
        sub_candidates =candidates[i:end_index]


        for candidate_ in sub_candidates:

            t=time.time()
            print(f'target candidate {candidate_}')
            with open(config.home_dir+"ros_execute.txt", 'w') as f:
                f.write('Wait')
            state_ = 'Wait'
            action_ = ''

            if candidate_[-1] == 0:  # grasp

                index = int(candidate_[1])

                success,grasp_width, distance, T, center_point=\
                    gripper_processing(index,point_data,poses,isvis)

                print(f'prediction time = {time.time() - t}')

                if not success :
                    print('Candidate has collision')
                    if simulation_mode:
                        if chances == 0:
                            return actions, states, data
                        chances -= 1
                    continue



                print('***********************use grasp***********************')
                print('Distance = ',distance, ' , width = ',grasp_width)
                print('Score = ', grasp_score_pred[index])

                if simulation_mode:
                    if chances == 0:
                        return actions, states, data
                    chances-=1
                    continue

                action_ = 'grasp'

                subprocess.run(execute_grasp_bash)
                # os.system(execute_grasp_bash)
                chances -= 1

            else:  # suction
                index = int(candidate_[1])
                skip_probability = skip_factor * (math.exp(-5 * suction_score_pred[index]) - math.exp(-5))
                if np.random.rand() < skip_probability:
                    continue

                success, suction_xyz,pred_approch_vector = suction_processing(index, point_data,normals, isvis)
                if not success: continue
                print('***********************use suction***********************')
                print('Suction score = ',suction_score_pred[index])


                if simulation_mode:
                    if chances == 0:
                        return actions, states, data
                    chances -= 1
                    continue

                action_ = 'suction'
                subprocess.run(execute_suction_bash)

                # os.system(execute_suction_bash)
                chances -= 1

            state_=wait_for_feedback(state_)
            if action_ == 'grasp':
                actions.append(action_)
                states.append(state_)
                data.append((grasp_width, distance, T, [1, 0, 0], center_point))
            if action_ == 'suction':
                actions.append(action_)
                states.append(state_)
                data.append((0, 0, np.eye(4), pred_approch_vector, suction_xyz))

            if  state_ == 'Succeed' or state_=='reset' or chances==0:
                return actions,states,data

    else:
        print(Fore.RED, 'No feasible pose is found in the current forward output', Fore.RESET)
        return actions,states,data

def main():
    inititalize_modules()

    first_loop=True

    while True:
        if first_loop:
            get_new_perception()
            first_loop=False

        img_suction_pre, img_grasp_pre = get_side_bins_images()

        with torch.no_grad():
            full_pc = get_real_data()
            full_pc_torch = torch.from_numpy(full_pc).float().unsqueeze(0).cuda(non_blocking=True)
            # view_npy_open3d(data.cpu().squeeze().numpy(), view_coordinate=True)

            candidates, grasp_score_pred, suction_score_pred,voxel_pc=model_inference(full_pc,max_grasp_candidates_=max_grasp_candidates
                                                                               ,max_suction_candidates_=max_suction_candidates
                                                                                   ,view_grasp_suction=view_grasp_suction_points)
            # view_npy_open3d(data.cpu().squeeze().numpy(), view_coordinate=True)

            # down_sampled_pc_ = full_pc_torch.detach().cpu().numpy().squeeze()
            print('Alternative poses = ',len(candidates))
            # results is defined as follows: [alternative_poses,score,index,grasp_or_suction]

            actions,states,data = run_robot( voxel_pc,candidates,grasp_score_pred,suction_score_pred,isvis=isvis)
            if simulation_mode==True: get_new_perception()

            for action_,state_,data_ in zip(actions,states,data):
                if empty_bin_check(state_):
                    inititalize_modules()
                    break

                award=process_feedback(action_, state_, data_, img_grasp_pre,img_suction_pre,full_pc)
def inititalize_modules():
    global suction_D_model
    global gripper_G_model
    global gripper_D_model
    global Suction_G_model
    global suction_scope_model
    global gripper_scope_model

    # from training.suction_D_training import affordance_net, initialize_model_state, affordance_net_model_path
    '''load architecture'''
    print(1)
    suction_D_model=initialize_model(suction_quality_net,suction_quality_model_state_path)
    # suction_D_model=initialize_model(affordance_net, affordance_net_model_path)
    print(2)

    suction_scope_model=initialize_model_state(scope_net_vanilla(in_size=6), suction_scope_model_state_path)
    print(3)

    gripper_D_model=initialize_model(gripper_quality_net,gripper_quality_model_state_path)
    # gripper_D_model=initialize_model(gripper_discriminator, dense_gripper_discriminator_path)
    print(4)

    # gripper_scope_model=initialize_model(gripper_scope_net,gripper_scope_model_state_path)
    # print(5)

    gripper_G_model=initialize_model(gripper_sampler_net, gripper_sampler_path)
    print(6)

    Suction_G_model=initialize_model(suction_sampler_net, suction_sampler_model_state_path)
    print(7)

    '''set evaluation mode'''
    gripper_G_model.eval()
    gripper_D_model.eval()
    suction_D_model.eval()
    Suction_G_model.eval()
    suction_scope_model.eval()
    # gripper_scope_model.eval()

def empty_bin_check(state):
    if state == 'No Object':
        print('state')
        subprocess.run('bash/testUSBCAN')
        # os.system('./testUSBCAN')

        time.sleep(5)
        get_new_perception()
        return True
    else:return False

def process_feedback(action,state,trail_data,img_grasp_pre,img_suction_pre,full_pc):
    award = 0

    #states
        #0 grasp action is executed
        #1 failed to find a planning path
        #2 reset
        #3 collision
    if state == 'Succeed' or state=='reset' : get_new_perception()
    if report_result==False:return None

    img_suction_after, img_grasp_after = get_side_bins_images()

    if action == 'grasp' and state == 'Succeed':
        # compare img_grasp_pre and img_grasp_after
        print(action, ':')

        award = check_image_similarity(img_grasp_pre, img_grasp_after)
        # if award is not None:
        #     save_new_data_sample(full_pc,trail_data[0], trail_data[1], trail_data[2], trail_data[3], trail_data[4], 1,
        #                          0, award,state=0)

    elif action == 'suction' and state == 'Succeed':

        # compare img_suction_pre and img_suction_after
        print(action, ':')

        award = check_image_similarity(img_suction_pre, img_suction_after)
        # if award is not None:
        #     save_new_data_sample(full_pc,trail_data[0], trail_data[1], trail_data[2], trail_data[3],
        #                      trail_data[4], 0, 1, award,state=0)

    elif action == 'grasp and suction' and state == 'Succeed':
        # compare img_grasp_pre and img_grasp_after
        print('grasp:')

        award = check_image_similarity(img_grasp_pre, img_grasp_after)

        # compare img_suction_pre and img_suction_after
        print('suction:')
        award = check_image_similarity(img_suction_pre, img_suction_after)

    elif state == 'reset' or state == 'Failed':
        print(action)
        s=None
        if state=='Failed':s=1
        if state=='reset':s=2
        # if action == 'grasp':
        #     save_new_data_sample(full_pc,trail_data[0], trail_data[1], trail_data[2], trail_data[3], trail_data[4], 1,
        #                          0, 0,state=s)
        # if action == 'suction':
        #     save_new_data_sample(full_pc,trail_data[0], trail_data[1], trail_data[2], trail_data[3],
        #                          trail_data[4], 0, 1, 0,state=s)

    else:
        print('No movement is executed, State: ',state,' ， Action： ',action)

    return award

if __name__ == "__main__":
    main()

