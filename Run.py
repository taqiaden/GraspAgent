# -*- coding: utf-8 -*-
import math
import os
import random

import torch
from colorama import Fore

from lib.ROS_communication import save_suction_data, wait_for_feedback,save_grasp_data
from lib.models_utils import  initialize_model
from lib.report_utils import wait_indicator as wi
from Configurations import config
from lib.IO_utils import save_to_file, load_file
from lib.bbox import  decode_gripper_pose, rotation_matrix_to_angles
from lib.grasp_utils import get_pose_matrixes, get_homogenous_matrix, get_grasp_width

import cv2 as cv
from dataset.load_test_data import get_real_data, estimate_suction_direction, sensory_pc_path
import subprocess
import trimesh.viewer
from lib.report_utils import save_error_log
from models.GAGAN import gripper_generator, dense_gripper_generator_path
from models.gripper_D import gripper_discriminator, dense_gripper_discriminator_path

from lib.dataset_utils import online_data
import time
import numpy as np
from lib.image_utils import check_image_similarity
from pose_object import output_processing, vectors_to_ratio_metrics, approach_vec_to_theta_phi
from visualiztion import vis_scene, visualize_suction_pose, visualize_grasp_and_suction_points, \
    view_score, view_npy_open3d
from lib.dataset_utils import rehearsal_data
from lib.collision_unit import local_exploration

rehearsal_data=rehearsal_data()

online_data=online_data()

shuffling_probability= 0.0
skip_factor= 0.0 # [1,10] Bigger value will increase the chance for skipping low score candidates

chances_ref = 500

max_grasp_candidates = None # if None then no limitation to the number of candidates
max_suction_candidates = None

view_grasp_suction_points = False
view_masked_grasp_pose = False
isvis= True
view_score_gradient= True
simulation_mode = True
offline_point_cloud= True

report_result=False

exploration_probabilty=0.0
suction_limit=0.7
gripper_limit=0.3

score_threshold=0.0

gripper_D_model=None
suction_D_model=None
gripper_G_model=None

sampling_last_index_path='dataset/last_index.txt'
get_point_bash='./bash/get_point.sh'
execute_suction_bash = './bash/run_robot_suction.sh'
execute_grasp_bash = './bash/run_robot_grasp.sh'
grasp_data_path = config.home_dir + 'grasp_data_tmp.npy'
pre_grasp_data_path = config.home_dir + 'pre_grasp_data_tmp.npy'
suction_data_path = config.home_dir + 'suction_data_tmp.npy'
pre_suction_data_path = config.home_dir + 'pre_suction_data_tmp.npy'
texture_image_path = config.home_dir + 'texture_image.jpg'

def get_spatial_mask(pc):
    x = pc[:, :, 0:1]
    y = pc[:, :, 1:2]
    z = pc[:, :, 2:3]
    x_mask = (x > 0.280 + 0.00) & (x < 0.582 - 0.00)
    y_mask = (y > -0.21 + 0.00) & (y < 0.21 - 0.00)
    z_mask = (z > config.z_limits[0]) & (z < config.z_limits[1])
    # print(z)
    spatial_mask = x_mask & y_mask & z_mask
    return spatial_mask

def dense_grasps_visualization(pc, generated_pose_7,grasp_score_pred,target_mask):


    # Method 1
    pose_good_grasp_list = []
    for i in range(5000):
        random_index = np.random.randint(0, config.num_points)
        if target_mask[0, random_index, 0] == False: continue

        if grasp_score_pred[0, 0, random_index] < 0.4: continue

        tmp_pose = generated_pose_7[0:1, :, random_index]

        poses_7 = tmp_pose[:, None, :].clone()

        # visualize verfication
        pc_ = pc[0, :, 0:3].cpu().numpy()  # + mean

        center_point = pc_[random_index]
        target_pose = poses_7[0, :, :]
        approach = target_pose[:, 0:3]

        theta, phi_sin, phi_cos = approach_vec_to_theta_phi(approach)
        target_pose[:, 0:1] = theta
        target_pose[:, 1:2] = phi_sin
        target_pose[:, 2:3] = phi_cos
        pose_5 = output_processing(target_pose[:, :, None]).squeeze(-1)

        pose_good_grasp = decode_gripper_pose(pose_5, center_point[0:3])

        # extreme_z = center_point[-1] - poses_7[0,0, -2] * config.distance_scope
        # if  extreme_z < config.z_limits[0]: continue

        pose_good_grasp_list.append(pose_good_grasp)
    if len(pose_good_grasp_list) == 0: return
    pose_good_grasp = np.concatenate(pose_good_grasp_list, axis=0)
    # print(pose_good_grasp.shape)
    # print(target_mask.squeeze().shape)
    # print(pc.shape)
    # masked_pc=pc[0,target_mask.squeeze(),0:3]
    # print(masked_pc.shape)
    # print(pc[0, :, 0:3].shape)

    vis_scene(pose_good_grasp[:, :], npy=pc[0, :, 0:3].cpu().numpy())

    return

poses=None
normals=None

def get_suction_pose_( suction_xyz, normal):
    suction_pose = normal.reshape(1, 3)  # Normal [[xn,yn,zn]]
    suction_xyz=suction_xyz.reshape(1,3)

    pose_good_suction = np.concatenate((suction_xyz, suction_pose), axis=1)  # [[x,y,z,xn,yn,zn]]
    position = pose_good_suction[0, [0, 1, 2]]  # [x,y,z]
    v0 = np.array([1, 0, 0])
    v1 = -pose_good_suction[0, [3, 4, 5]]  # [-xn,-yn,-zn]
    pred_approch_vector = pose_good_suction[0, [3, 4, 5]]
    a = trimesh.transformations.angle_between_vectors(v0, v1)
    b = trimesh.transformations.vector_product(v0, v1)
    matrix_ori = trimesh.transformations.rotation_matrix(a, b)
    matrix_ori[:3, 3] = position.T
    T = matrix_ori

    pre_grasp_mat, end_effecter_mat = get_pose_matrixes(T, k_end_effector=0.184, k_pre_grasp=0.25)

    suction_xyz = suction_xyz.squeeze()

    return suction_xyz, pre_grasp_mat, end_effecter_mat, suction_pose, T, pred_approch_vector

def get_suction_pose(index, point_data, normal):
    return  get_suction_pose_(point_data[index],normal)

def get_point(get_last_data=False):
    ctime_stamp = os.path.getctime(texture_image_path)
    ctime_stamp_pc = os.path.getctime(sensory_pc_path)
    if simulation_mode and offline_point_cloud == True:
        # get new data from data pool
        pc=online_data.load_random_pc()
        np.save(sensory_pc_path,pc)

    subprocess.run(get_point_bash)
    # os.system(get_point_bash)
    wait = wi('Waiting for new perception data')
    i=0
    while os.path.getctime(texture_image_path)==ctime_stamp or ctime_stamp_pc == os.path.getctime(sensory_pc_path):
        # if (i>1000 and get_last_data==True) or offline_point_cloud==True:
        if  offline_point_cloud == True:
            print(Fore.RED, 'Camera is not available. Load data from pool', Fore.RESET)
            break
        wait.step(0.01)
        i+=1
    wait.end()

def crop_side_tray_image(img):
    img_suction = img[200:540, 0:160, 0]
    img_grasp = img[180:570, 780:1032, 0]
    return img_suction, img_grasp

def verify_label_is_in_standard_form(label):
    assert len(label)==28, f'label size is {len(label)}, the standard size is 27'
    list_of_types = [float, float, float, int, int, float, float,
                     float, float, float, float, float, float, float,
                     float, float, float
                    , float, float, float, float, np.float64,
                     np.float64, int, int, int, int,int]
    for i in range(len(label)):
        if i==3:
            # the socre can be saved in either float and int
            assert np.float32 == type(label[i]) or int == type(label[i]), f'label[{i}] type is {type(label[i])}, expected type is {np.float32} or {int}'
        else:
            assert list_of_types[i]==type(label[i]), f'label[{i}] type is {type(label[i])}, expected type is {list_of_types[i]}'

def standard_label_structure(width, distance, transformation, normal, center_point, grasp, suction, score,state):
    label = center_point.tolist() + [score]

    transformation = transformation.reshape(-1)

    if grasp:
        label = label + [grasp] + transformation.tolist() + [width, distance]
        label = label + [0] * 4

    if suction:
        label = label + [0] * 19
        label = label + [suction] + normal.tolist()

    label=label+[state]
    # verify_label_is_in_standard_form(label)

    return np.array(label)

def save_new_data_sample(full_pc,width, distance, transformation, normal, center_point, grasp, suction, score ,state,dataset=online_data):
    # return
    # get the last nummber\
    assert simulation_mode==False
    index = load_file(sampling_last_index_path)+1

    # if grasp==1:verify_saved_grasp_pose(transformation,distance,width,center_point,full_pc)

    # point_data = np.load(sensory_pc_path)
    point_data=full_pc
    label = standard_label_structure(width, distance, transformation, normal, center_point, grasp, suction, score,state)
    dataset.save_labeled_data(point_data,np.asarray(label),str(index).zfill(6))

    if score==1:print(Fore.GREEN,'Save new data, award =', score, Fore.RESET)
    else: print(Fore.YELLOW,'Save new data, award =', score,Fore.RESET)

    save_to_file(index, sampling_last_index_path)

    # tabulated data:
    # [0:3]: center_point
    # [3]: score
    # [4]: grasp
    # [5:21]: rotation_matrix
    # [21]: width
    # [22]: distance
    # [23]: suction
    # [24:27]: pred_normal

def accumulate_mask_layer(array,current_mask,maximum_size,mask_inversion=False):
    global score_threshold
    m1=(array>score_threshold) & (current_mask)
    z=array[m1]
    if z.shape[0]<0 or maximum_size is None:
        save_error_log('Mask has no True element, function: accumulate_mask_layer')
        return m1
    if z.shape[0]<=maximum_size: return m1
    else:
        limit_value=sorted(z,reverse= not mask_inversion)[maximum_size]
        value_mask=array<limit_value if  mask_inversion else array>limit_value
        total_mask=(array>score_threshold) & (value_mask) & (current_mask)
        return total_mask

def initialize_masks(grasp_score_pred, suction_score_pred, data_ ,grasp_max_size,suction_max_size,mask_inversion=False):
        # filter bin points
        x = data_[:, 0]
        y = data_[:, 1]
        z = data_[:, 2]
        x_mask = (x > 0.275+0.01) & (x < 0.585-0.01)
        y_mask = (y > -0.20) & (y < 0.20)
        z_mask = (z > config.z_limits[0]) & (z < config.z_limits[1])
        mask = x_mask & y_mask & z_mask
        # view_npy_open3d(data_)
        # view_npy_open3d(data_[mask])
        # view_npy_open3d(data_[~mask])
        # exit()
        grasp_score_pred_mask = accumulate_mask_layer(grasp_score_pred, mask, grasp_max_size,mask_inversion)

        suction_score_pred_mask = accumulate_mask_layer(suction_score_pred,mask, suction_max_size,mask_inversion)


        return grasp_score_pred_mask, suction_score_pred_mask


def dense_data_preprocessing(data,max_grasp_candidates_,
                             max_suction_candidates_,view_grasp_suction=False,mask_inversion=False):

    global normals
    global poses

    global gripper_D_model
    global gripper_G_model
    global suction_D_model

    target_mask = get_spatial_mask(data)

    normals=estimate_suction_direction(data.cpu().numpy().squeeze())
    normals_torch=torch.from_numpy(normals).to('cuda').float()[None,:,:]
    pc_with_normals=torch.cat([data,normals_torch],dim=-1)

    suction_score_pred,_ = suction_D_model(pc_with_normals.clone())

    # estimate_suction_direction(data.cpu().numpy().squeeze(), view=True, view_mask=target_mask,score=suction_score_pred)

    _ ,_,poses= gripper_G_model(data.clone(),output_theta_phi=False)
    assert torch.any(torch.isinf(poses))==False,f'{poses}'

    # print(poses)
    grasp_score_pred,_=gripper_D_model(pc_with_normals,poses,use_collision_module=True,use_quality_module=True)
    # dense_grasps_visualization(pc_with_normals.clone(), poses.clone(),grasp_score_pred.clone(),target_mask)
    # grasp_score_pred+=.5
    # print(grasp_score_pred.sum())
    # print(grasp_score_pred.mean())
    # print(suction_score_pred.sum())
    # print(suction_score_pred.mean())
    # post processing
    suction_score_pred[suction_score_pred < suction_limit] = -1
    grasp_score_pred[grasp_score_pred<gripper_limit]=-1
    # high_suction_confidence_mask = suction_score_pred> 1.0
    # high_gripper_confidence_mask=grasp_score_pred>1.0


    # grasp_score_pred[high_suction_confidence_mask]=0.
    # suction_score_pred[high_gripper_confidence_mask]=0.

    # print(collision_score)

    # print(grasp_score_pred.mean())
    # collision_score -= torch.min(collision_score)
    # collision_score/=torch.max(collision_score)
    # print(f'score sum={grasp_score_pred.sum()}')

    poses= vectors_to_ratio_metrics(poses)
    # print(torch.max(poses[:,2,:]))
    # print(torch.mean(poses[:,2,:]))
    # exit()

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

def inference_dense_gripper_pose(point_data_npy,center_point,index):
    global poses
    # point_data = torch.from_numpy(point_data_npy).to('cuda')
    # point_data = point_data[None, :, :]
    # poses=dense_gripper_net.dense_gripper_generator_net_(point_data)
    pose=poses[:,:,index]



    print('prediction------------------------------------------------',pose)
    # if np.random.random()>0.3:
    #     pose=torch.rand_like(pose)
    #     print('random pose=------------------------------------------------',pose)

    # gripper_pose_net.gripper_net.eval()
    # theta_phi_output_GAN=gripper_pose_net.gripper_net(depth_image,center_point_)
    # output = theta_phi_output_GAN

    pose_good_grasp=decode_gripper_pose(pose,center_point)
    return pose_good_grasp

def  grasp_processing(index,point_data,isvis):
    # view_npy_open3d(point_data,view_coordinate=True)

    # Get the pose_good_grasp
    center_point = point_data[index]
    # print('center------------',center_point)
    # view_npy_open3d(point_data,view_coordinate=True)
    # print(point_data.shape)
    pose_good_grasp=inference_dense_gripper_pose(point_data, center_point, index)

    # pose_good_grasp=inference_gripper_pose(point_data,center_point,index)
    # vis_scene(pose_good_grasp[:, :].reshape(1, 14),npy=point_data)
    activate_exploration=True if np.random.rand()<exploration_probabilty else False
    # view_npy_open3d(point_data,view_coordinate=True)
    pose_good_grasp,collision_intensity = local_exploration(pose_good_grasp,point_data, exploration_attempts=5,
                                             explore_if_collision=False, view_if_sucess=view_masked_grasp_pose,explore=activate_exploration)
    success=collision_intensity==0
    # collision_intensity=1.0 if success else 0.0
    # view_npy_open3d(point_data,view_coordinate=True)

    # Get related parameters
    T = get_homogenous_matrix(pose_good_grasp)
    distance = pose_good_grasp[0, -1]

    grasp_width = get_grasp_width(pose_good_grasp)
    # gripper_net_processing(point_data, index, pose_good_grasp,collision_intensity,center_point_)

    if not success:
        return False, pose_good_grasp,grasp_width, distance, T, center_point

    if simulation_mode==False:
        pre_grasp_mat, end_effecter_mat = get_pose_matrixes(T, k_end_effector=0.169, k_pre_grasp=0.23)
        save_grasp_data(end_effecter_mat, grasp_width, grasp_data_path)
        save_grasp_data(pre_grasp_mat, grasp_width, pre_grasp_data_path)

    if isvis: vis_scene(pose_good_grasp[:, :].reshape(1, 14),npy=point_data)

    return True, pose_good_grasp,grasp_width, distance, T, center_point

def suction_processing(index,point_data,isvis):

    global normals
    normal=normals[index]
    normal=normal[None,:]

    suction_xyz, pre_grasp_mat, end_effecter_mat, suction_pose, T, pred_approch_vector \
        = get_suction_pose(index, point_data, normal)
    # suction_net_processing(point_data, index)

    if pre_grasp_mat[0, 3] < config.x_min_dis:
        return False, suction_xyz,pred_approch_vector

    save_suction_data(end_effecter_mat, suction_data_path)
    save_suction_data(pre_grasp_mat, pre_suction_data_path)

    if isvis: visualize_suction_pose(suction_xyz, suction_pose, T, end_effecter_mat,npy=point_data)

    return True,suction_xyz,pred_approch_vector
def run_robot( point_data, candidates,grasp_score_pred,suction_score_pred,isvis):
    # view_npy_open3d(point_data,view_coordinate=True)

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
                    grasp_processing(index,point_data,isvis)

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
            get_point(get_last_data=first_loop)
            first_loop=False

        im = cv.imread(texture_image_path)
        img_suction_pre, img_grasp_pre = crop_side_tray_image(im)

        with torch.no_grad():
            down_sampled_pc,full_pc = get_real_data()
            down_sampled_pc = torch.from_numpy(down_sampled_pc).float().unsqueeze(0).cuda(non_blocking=True)
            # view_npy_open3d(data.cpu().squeeze().numpy(), view_coordinate=True)

            candidates, grasp_score_pred, suction_score_pred=dense_data_preprocessing(down_sampled_pc,max_grasp_candidates_=max_grasp_candidates
                                                                               ,max_suction_candidates_=max_suction_candidates
                                                                                   ,view_grasp_suction=view_grasp_suction_points)
            # view_npy_open3d(data.cpu().squeeze().numpy(), view_coordinate=True)

            down_sampled_pc_ = down_sampled_pc.detach().cpu().numpy().squeeze()
            print('Alternative poses = ',len(candidates))
            # results is defined as follows: [alternative_poses,score,index,grasp_or_suction]

            actions,states,data = run_robot( down_sampled_pc_,candidates,grasp_score_pred,suction_score_pred,isvis=isvis)
            if simulation_mode==True: get_point()

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
        get_point()
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
    if state == 'Succeed' or state=='reset' : get_point()
    if report_result==False:return None

    im = cv.imread(texture_image_path)
    img_suction_after, img_grasp_after = crop_side_tray_image(im)

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

