import math
import subprocess
import time
import numpy as np
import torch
from colorama import Fore
from Configurations import config
from Configurations.run_config import simulation_mode, max_grasp_candidates, max_suction_candidates, \
    view_grasp_suction_points, suction_limit, gripper_limit, suction_factor, gripper_factor, view_score_gradient, \
    chances_ref, shuffling_probability, view_action, report_result
from grasp_post_processing import gripper_processing, suction_processing
from lib.ROS_communication import wait_for_feedback, ROS_communication_file
from lib.depth_map import transform_to_camera_frame, point_clouds_to_depth, depth_to_point_clouds
from lib.image_utils import check_image_similarity
from lib.models_utils import initialize_model, initialize_model_state
from lib.report_utils import progress_indicator
from masks import initialize_masks
from models.Grasp_GAN import gripper_sampler_net, gripper_sampler_path
from models.gripper_quality import gripper_quality_net, gripper_quality_model_state_path
from models.scope_net import scope_net_vanilla, suction_scope_model_state_path
from models.suction_quality import suction_quality_net, suction_quality_model_state_path
from models.suction_sampler import suction_sampler_net, suction_sampler_model_state_path
from pose_object import vectors_to_ratio_metrics
from process_perception import get_new_perception, get_side_bins_images
from registration import camera
from visualiztion import visualize_grasp_and_suction_points, view_score

execute_suction_bash = './bash/run_robot_suction.sh'
execute_grasp_bash = './bash/run_robot_grasp.sh'

class mode():
    def __init__(self):
        self.simulation=simulation_mode
        self.report_result=report_result

        self.max_gripper_candidates=max_grasp_candidates
        self.max_suction_candidates=max_suction_candidates

        self.view_grasp_points=view_grasp_suction_points
        self.view_scores=view_score_gradient
        self.view_action=view_action

        self.chances=chances_ref
        self.shuffling_probability=shuffling_probability


class GraspAgent():
    def __init__(self):
        '''set run mode'''
        self.mode=mode()

        '''models'''
        self.gripper_D_model = None
        self.suction_D_model = None
        self.gripper_G_model = None
        self.Suction_G_model = None
        self.suction_scope_model = None
        self.gripper_scope_model = None

        '''modalities'''
        self.point_clouds = None
        self.depth=None

        '''dense candidates'''
        self.gripper_poses=None
        self.normal=None
        self.candidates=None
        self.gripper_scores=None
        self.suction_scores=None
        self.voxel_pc=None

        self.n_candidates=0

        '''execution results'''
        self.actions=[]
        self.states=[]
        self.data=[]

    def initialize_check_points(self):
        pi = progress_indicator('Loading check points  ', max_limit=7)
        '''load check points'''
        pi.step(1)
        self.suction_D_model = initialize_model(suction_quality_net, suction_quality_model_state_path)
        # suction_D_model=initialize_model(affordance_net, affordance_net_model_path)
        pi.step(2)

        self.suction_scope_model = initialize_model_state(scope_net_vanilla(in_size=6), suction_scope_model_state_path)
        pi.step(3)

        self.gripper_D_model = initialize_model(gripper_quality_net, gripper_quality_model_state_path)
        # gripper_D_model=initialize_model(gripper_discriminator, dense_gripper_discriminator_path)
        pi.step(4)

        # gripper_scope_model=initialize_model(gripper_scope_net,gripper_scope_model_state_path)
        pi.step(5)

        self.gripper_G_model = initialize_model(gripper_sampler_net, gripper_sampler_path)
        pi.step(6)

        self.Suction_G_model = initialize_model(suction_sampler_net, suction_sampler_model_state_path)
        pi.step(7)

        '''set evaluation mode'''
        self.gripper_G_model.eval()
        self.gripper_D_model.eval()
        self.suction_D_model.eval()
        self.Suction_G_model.eval()
        self.suction_scope_model.eval()

        pi.end()

    def model_inference(self,point_clouds, mask_inversion=False):
        self.point_clouds=point_clouds

        '''get depth'''
        transformed_pc = transform_to_camera_frame(self.point_clouds)
        self.depth = point_clouds_to_depth(transformed_pc, camera)
        depth_torch = torch.from_numpy(self.depth)[None, None, ...].to('cuda').float()

        '''sample suction'''
        normals_pixels = self.Suction_G_model(depth_torch.clone())

        '''sample gripper'''
        poses_pixels = self.gripper_G_model(depth_torch.clone())

        '''suction scope'''
        voxel_pc, mask = depth_to_point_clouds(self.depth, camera)
        voxel_pc = transform_to_camera_frame(voxel_pc, reverse=True)
        positions = torch.from_numpy(voxel_pc).to('cuda').float()
        normals = normals_pixels.squeeze().permute(1, 2, 0)[mask]
        suction_scope = self.suction_scope_model(torch.cat([positions, normals], dim=-1)).squeeze().detach().cpu().numpy()
        suction_scope = np.clip(suction_scope, 0, 1)
        # suction_scope[suction_scope>=0.3]=1.
        # suction_scope[suction_scope<0.3]=0.0

        '''gripper scope'''
        # gripper_scope=gripper_scope_model(poses_pixels[:,0:3,...].clone())

        '''suction quality'''
        suction_quality_score = self.suction_D_model(depth_torch.clone(), normals_pixels.clone()).squeeze()

        '''gripper quality'''
        gripper_quality_score = self.gripper_D_model(depth_torch.clone(), poses_pixels.clone()).squeeze()

        '''res u net processing'''
        voxel_pc, mask = depth_to_point_clouds(self.depth, camera)
        voxel_pc = transform_to_camera_frame(voxel_pc, reverse=True)
        suction_quality_score = suction_quality_score[mask].detach().cpu().numpy()
        gripper_quality_score = gripper_quality_score[mask].detach().cpu().numpy()
        normals = normals_pixels.squeeze().permute(1, 2, 0)[mask]
        poses = poses_pixels.squeeze().permute(1, 2, 0)[mask]

        '''final scores'''
        # suction_scores=suction_quality_score*suction_scope
        suction_scores = suction_scope
        gripper_scores = gripper_quality_score + 1  # *gripper_scope
        suction_scores = suction_scores.squeeze()
        gripper_scores = gripper_scores.squeeze()

        '''set limits'''
        suction_scores[suction_scores < suction_limit] = -1
        gripper_scores[gripper_scores < gripper_limit] = -1
        suction_scores = suction_scores * suction_factor
        gripper_scores = gripper_scores * gripper_factor

        '''post processing'''
        best_gripper_scores = gripper_scores > 1.0
        suction_scores[best_gripper_scores] = -1

        # if view_sampled_suction:
        #     target_mask = get_spatial_mask(data)
        #     estimate_suction_direction(data.cpu().numpy().squeeze(), view=True, view_mask=target_mask,score=suction_score_pred)
        # if view_sampled_griper:
        #     dense_grasps_visualization(pc_with_normals.clone(), poses.clone(),grasp_score_pred.clone(),target_mask)
        # suction_score_pred[suction_score_pred < suction_limit] = -1
        # grasp_score_pred[grasp_score_pred<gripper_limit]=-1

        self.gripper_poses = vectors_to_ratio_metrics(poses)
        self.normals = normals.detach().cpu().numpy()

        grasp_score_pred_mask, suction_score_pred_mask = initialize_masks(gripper_scores, suction_scores, voxel_pc,
                                                                          self.mode.max_gripper_candidates,
                                                                          self.mode.max_suction_candidates, mask_inversion)

        if self.mode.view_grasp_points: visualize_grasp_and_suction_points(suction_score_pred_mask, grasp_score_pred_mask,
                                                                  voxel_pc)
        if self.mode.view_scores:
            view_score(voxel_pc, grasp_score_pred_mask, gripper_scores)
            view_score(voxel_pc, suction_score_pred_mask, suction_scores)

        index_suction_cls = [i for i in range(len(suction_score_pred_mask)) if suction_score_pred_mask[i]]
        index_grasp_cls = [j for j in range(len(grasp_score_pred_mask)) if grasp_score_pred_mask[j]]

        score_grasp_value_ind = list((item * 1, index_grasp_cls[index], 0) for index, item in
                                     enumerate(gripper_scores[grasp_score_pred_mask]))
        score_suction_value_ind_ = list((item * 1, index_suction_cls[index], 1) for index, item in
                                        enumerate(suction_scores[suction_score_pred_mask]))

        candidates = score_grasp_value_ind + score_suction_value_ind_

        if len(candidates) > 1:
            candidates = sorted(candidates, reverse=True)

        self.candidates=candidates
        self.gripper_scores=gripper_scores
        self.suction_scores=suction_scores
        self.voxel_pc=voxel_pc


        return candidates, gripper_scores, suction_scores, voxel_pc

    def execute(self):
        state_ = ''
        self.n_candidates = len(self.candidates)
        if self.n_candidates== 0:
            self.actions.append(state_)
            self.states.append('No Object')
            self.data.append((0, 0, np.eye(4), [1, 0, 0], [0, 0, 0]))
            return self.actions, self.states, self.data
        # global chances_ref
        chances = self.mode.chances

        if self.n_candidates > 100:
            if shuffling_probability == 0.0:
                chunk_size = math.floor(0.01 * self.n_candidates)
            else:
                chunk_size = math.floor(0.1 * self.n_candidates)
        else:
            chunk_size = 100

        for i in range(0, self.n_candidates, chunk_size):

            end_index = min(i + chunk_size, self.n_candidates)
            sub_candidates = self.candidates[i:end_index]

            for candidate_ in sub_candidates:

                t = time.time()
                print(f'target candidate {candidate_}')
                with open(config.home_dir + ROS_communication_file, 'w') as f:
                    f.write('Wait')
                state_ = 'Wait'
                action_ = ''

                if candidate_[-1] == 0:  # grasp

                    index = int(candidate_[1])

                    success, grasp_width, distance, T, center_point = \
                        gripper_processing(index, self.voxel_pc, self.gripper_poses, self.mode.view_action)

                    print(f'prediction time = {time.time() - t}')

                    if not success:
                        print('Candidate has collision')
                        if simulation_mode:
                            if chances == 0:
                                return self.actions, self.states, self.data
                            chances -= 1
                        continue

                    print('***********************use grasp***********************')
                    print('Score = ', self.gripper_scores[index])

                    if simulation_mode:
                        if chances == 0:
                            return self.actions, self.states, self.data
                        chances -= 1
                        continue

                    action_ = 'grasp'

                    subprocess.run(execute_grasp_bash)
                    # os.system(execute_grasp_bash)
                    chances -= 1

                else:  # suction
                    index = int(candidate_[1])

                    success, suction_xyz, pred_approch_vector = suction_processing(index, self.voxel_pc, self.normals, self.mode.view_action)
                    if not success: continue
                    print('***********************use suction***********************')
                    print('Suction score = ', self.suction_scores[index])

                    if simulation_mode:
                        if chances == 0:
                            return self.actions, self.states, self.data
                        chances -= 1
                        continue

                    action_ = 'suction'
                    subprocess.run(execute_suction_bash)

                    # os.system(execute_suction_bash)
                    chances -= 1

                state_ = wait_for_feedback(state_)
                if action_ == 'grasp':
                    self.actions.append(action_)
                    self.states.append(state_)
                    self.data.append((grasp_width, distance, T, [1, 0, 0], center_point))
                if action_ == 'suction':
                    self.actions.append(action_)
                    self.states.append(state_)
                    self.data.append((0, 0, np.eye(4), pred_approch_vector, suction_xyz))

                if state_ == 'Succeed' or state_ == 'reset' or chances == 0:
                    return self.actions, self.states, self.data

        else:
            print(Fore.RED, 'No feasible pose is found in the current forward output', Fore.RESET)
            return self.actions, self.states, self.data

    def process_feedback(self,action_, state_, data_, img_grasp_pre, img_suction_pre):
        # states
        # 0 grasp action is executed
        # 1 failed to find a planning path
        # 2 reset
        # 3 collision
        if state_ == 'Succeed' or state_ == 'reset': get_new_perception()
        if not self.mode.report_result: return None

        img_suction_after, img_grasp_after = get_side_bins_images()

        if action_ == 'grasp' and state_ == 'Succeed':
            # compare img_grasp_pre and img_grasp_after
            print(action_, ':')

            award = check_image_similarity(img_grasp_pre, img_grasp_after)
            # if award is not None:
            #     save_new_data_sample(full_pc,trail_data[0], trail_data[1], trail_data[2], trail_data[3], trail_data[4], 1,
            #                          0, award,state=0)

        elif action_ == 'suction' and state_ == 'Succeed':

            # compare img_suction_pre and img_suction_after
            print(action_, ':')

            award = check_image_similarity(img_suction_pre, img_suction_after)
            # if award is not None:
            #     save_new_data_sample(full_pc,trail_data[0], trail_data[1], trail_data[2], trail_data[3],
            #                      trail_data[4], 0, 1, award,state=0)

        elif state_ == 'reset' or state_ == 'Failed':
            print(action_)
            s = None
            if state_ == 'Failed': s = 1
            if state_ == 'reset': s = 2
            # if action == 'grasp':
            #     save_new_data_sample(full_pc,trail_data[0], trail_data[1], trail_data[2], trail_data[3], trail_data[4], 1,
            #                          0, 0,state=s)
            # if action == 'suction':
            #     save_new_data_sample(full_pc,trail_data[0], trail_data[1], trail_data[2], trail_data[3],
            #                          trail_data[4], 0, 1, 0,state=s)

        else:
            print('No movement is executed, State: ', state_, ' ， Action： ', action_)

