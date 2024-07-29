# -*- coding: utf-8 -*-
import math
import random
import torch
from colorama import Fore
from Online_data_audit.process_feedback import save_new_data_sample
from grasp_post_processing import gripper_processing, suction_processing
from lib.ROS_communication import  wait_for_feedback
from lib.models_utils import  initialize_model
from Configurations import config
import subprocess
from masks import get_spatial_mask, initialize_masks
from models.GAGAN import gripper_generator, dense_gripper_generator_path
from models.gripper_D import gripper_discriminator, dense_gripper_discriminator_path
import time
import numpy as np
from lib.image_utils import check_image_similarity
from pose_object import  vectors_to_ratio_metrics
from process_perception import get_new_perception, get_side_bins_images, get_real_data
from suction_sampler import estimate_suction_direction
from visualiztion import  visualize_grasp_and_suction_points, \
    view_score, dense_grasps_visualization

shuffling_probability= 0.0
skip_factor= 0.0 # [1,10] Bigger value will increase the chance for skipping low score candidates

chances_ref = 500

max_grasp_candidates = None # if None then no limitation to the number of candidates
max_suction_candidates = None

view_grasp_suction_points = False
isvis= True
view_score_gradient= True
simulation_mode = True

report_result=False

view_sampled_griper=False
view_sampled_suction=False

suction_limit=0.7
gripper_limit=0.3

score_threshold=0.0

gripper_D_model=None
suction_D_model=None
gripper_G_model=None

execute_suction_bash = './bash/run_robot_suction.sh'
execute_grasp_bash = './bash/run_robot_grasp.sh'

poses=None
normals=None

def model_inference(data,max_grasp_candidates_,
                             max_suction_candidates_,view_grasp_suction=False,mask_inversion=False):

    global normals
    global poses
    global gripper_D_model
    global gripper_G_model
    global suction_D_model


    normals=estimate_suction_direction(data.cpu().numpy().squeeze())
    normals_torch=torch.from_numpy(normals).to('cuda').float()[None,:,:]
    pc_with_normals=torch.cat([data,normals_torch],dim=-1)

    suction_score_pred,_ = suction_D_model(pc_with_normals.clone())

    if view_sampled_suction:
        target_mask = get_spatial_mask(data)
        estimate_suction_direction(data.cpu().numpy().squeeze(), view=True, view_mask=target_mask,score=suction_score_pred)

    _ ,_,poses= gripper_G_model(data.clone(),output_theta_phi=False)
    assert torch.any(torch.isinf(poses))==False,f'{poses}'

    # print(poses)
    grasp_score_pred,_=gripper_D_model(pc_with_normals,poses,use_collision_module=True,use_quality_module=True)
    if view_sampled_griper:
        dense_grasps_visualization(pc_with_normals.clone(), poses.clone(),grasp_score_pred.clone(),target_mask)

    suction_score_pred[suction_score_pred < suction_limit] = -1
    grasp_score_pred[grasp_score_pred<gripper_limit]=-1


    poses= vectors_to_ratio_metrics(poses)

    grasp_score_pred=grasp_score_pred.detach().cpu().numpy().squeeze()
    suction_score_pred=suction_score_pred.detach().cpu().numpy().squeeze()

    data_ = data.detach().cpu().numpy().squeeze()

    grasp_score_pred_mask, suction_score_pred_mask = initialize_masks(grasp_score_pred, suction_score_pred, data_,
                                                                      max_grasp_candidates_,max_suction_candidates_,mask_inversion)

    if view_grasp_suction: visualize_grasp_and_suction_points(suction_score_pred_mask, grasp_score_pred_mask,
                                                                     data_)
    if view_score_gradient:
        view_score(data_, grasp_score_pred_mask, grasp_score_pred)
        view_score(data_, suction_score_pred_mask, suction_score_pred)

    index_suction_cls = [i for i in range(len(suction_score_pred_mask)) if suction_score_pred_mask[i]]
    index_grasp_cls = [j for j in range(len(grasp_score_pred_mask)) if grasp_score_pred_mask[j]]


    score_grasp_value_ind = list((item * 1, index_grasp_cls[index], 0) for index, item in
                                 enumerate(grasp_score_pred[grasp_score_pred_mask]))
    score_suction_value_ind_ = list((item * 1, index_suction_cls[index], 1) for index, item in
                                    enumerate(suction_score_pred[suction_score_pred_mask]))

    candidates = score_grasp_value_ind + score_suction_value_ind_

    if len(candidates)>1:
        candidates = sorted(candidates, reverse=True)
    # print(candidates[0:100])

    return candidates,grasp_score_pred,suction_score_pred

def run_robot( point_data, candidates,grasp_score_pred,suction_score_pred,isvis):
    actions, states, data = [], [], []
    state_ = ''
    if len(candidates) == 0:
        actions.append(state_)
        states.append('No Object')
        data.append((0, 0, np.eye(4), [1, 0, 0], [0,0,0]))
        return actions,states,data
    global chances_ref
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
        # print(f'average score of the top {np.asarray(sub_candidates)[:,0].mean()}')

        # if np.random.rand()<shuffling_probability:
        random.shuffle(sub_candidates)



        for candidate_ in sub_candidates:
            print(f'target candidate {candidate_}')
            with open(config.home_dir+"ros_execute.txt", 'w') as f:
                f.write('Wait')
            state_ = 'Wait'
            action_ = ''

            if candidate_[-1] == 0:  # grasp

                index = int(candidate_[1])

                success,pose_good_grasp,grasp_width, distance, T, center_point=\
                    gripper_processing(index,point_data,isvis)

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

                success, suction_xyz,pred_approch_vector = suction_processing(index, point_data, isvis)
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
            down_sampled_pc,full_pc = get_real_data()
            down_sampled_pc = torch.from_numpy(down_sampled_pc).float().unsqueeze(0).cuda(non_blocking=True)
            # view_npy_open3d(data.cpu().squeeze().numpy(), view_coordinate=True)

            candidates, grasp_score_pred, suction_score_pred=model_inference(down_sampled_pc,max_grasp_candidates_=max_grasp_candidates
                                                                               ,max_suction_candidates_=max_suction_candidates
                                                                                   ,view_grasp_suction=view_grasp_suction_points)
            # view_npy_open3d(data.cpu().squeeze().numpy(), view_coordinate=True)

            down_sampled_pc_ = down_sampled_pc.detach().cpu().numpy().squeeze()
            print('Alternative poses = ',len(candidates))
            # results is defined as follows: [alternative_poses,score,index,grasp_or_suction]

            actions,states,data = run_robot( down_sampled_pc_,candidates,grasp_score_pred,suction_score_pred,isvis=isvis)
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
    from training.suction_D_training import affordance_net, initialize_model_state, affordance_net_model_path
    suction_D_model = affordance_net()
    gripper_G_model = initialize_model(gripper_generator, dense_gripper_generator_path)
    suction_D_model = initialize_model_state(suction_D_model, affordance_net_model_path)
    gripper_D_model = initialize_model(gripper_discriminator, dense_gripper_discriminator_path)
    gripper_G_model.eval()
    gripper_D_model.eval()
    suction_D_model.eval()

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
    global shuffling_probability
    #states
        #0 grasp action is performed
        #1 failed to find a planning path
        #2 reset
        #3 collision
    if state == 'Succeed' or state=='reset' : get_new_perception()
    if report_result==False:return None

    img_suction_after, img_grasp_after = get_side_bins_images()

    if action == 'grasp' and state == 'Succeed':
        shuffling_probability=0.0

        # compare img_grasp_pre and img_grasp_after
        print(action, ':')

        award = check_image_similarity(img_grasp_pre, img_grasp_after)
        if award is not None:
            save_new_data_sample(full_pc,trail_data[0], trail_data[1], trail_data[2], trail_data[3], trail_data[4], 1,
                                 0, award,state=0)

    elif action == 'suction' and state == 'Succeed':
        shuffling_probability=0.0

        # compare img_suction_pre and img_suction_after
        print(action, ':')

        award = check_image_similarity(img_suction_pre, img_suction_after)
        if award is not None:
            save_new_data_sample(full_pc,trail_data[0], trail_data[1], trail_data[2], trail_data[3],
                             trail_data[4], 0, 1, award,state=0)

    elif action == 'grasp and suction' and state == 'Succeed':
        shuffling_probability=0.0
        # compare img_grasp_pre and img_grasp_after
        print('grasp:')

        award = check_image_similarity(img_grasp_pre, img_grasp_after)

        # compare img_suction_pre and img_suction_after
        print('suction:')
        award = check_image_similarity(img_suction_pre, img_suction_after)

    elif state == 'reset' or state == 'Failed':
        shuffling_probability=1.0
        print(action)
        s=None
        if state=='Failed':s=1
        if state=='reset':s=2
        if action == 'grasp':
            save_new_data_sample(full_pc,trail_data[0], trail_data[1], trail_data[2], trail_data[3], trail_data[4], 1,
                                 0, 0,state=s)
        if action == 'suction':
            save_new_data_sample(full_pc,trail_data[0], trail_data[1], trail_data[2], trail_data[3],
                                 trail_data[4], 0, 1, 0,state=s)

    else:
        print('No movement is executed, State: ',state,' ， Action： ',action)

    return award

if __name__ == "__main__":
    main()

