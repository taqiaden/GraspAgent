import math
import subprocess
import time
import numpy as np
import torch
from colorama import Fore
from Configurations import config
from Configurations.run_config import simulation_mode, view_grasp_suction_points, suction_limit, gripper_limit, \
    suction_factor, gripper_factor, view_score_gradient, \
    chances_ref, shuffling_probability, view_action, report_result, scope_threshold, use_gripper, use_suction
from Online_data_audit.process_feedback import save_grasp_sample
from grasp_post_processing import gripper_processing, suction_processing
from lib.ROS_communication import wait_for_feedback, ROS_communication_file
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from lib.image_utils import check_image_similarity
from lib.models_utils import  initialize_model_state
from lib.report_utils import progress_indicator
from masks import  static_spatial_mask
from models.Grasp_GAN import gripper_sampler_net, gripper_sampler_path
from models.gripper_quality import gripper_quality_net, gripper_quality_model_state_path
from models.scope_net import scope_net_vanilla, suction_scope_model_state_path, gripper_scope_model_state_path
from models.suction_quality import suction_quality_net, suction_quality_model_state_path
from models.suction_sampler import suction_sampler_net, suction_sampler_model_state_path
from pose_object import vectors_to_ratio_metrics
from process_perception import get_new_perception, get_side_bins_images
from registration import camera
from visualiztion import visualize_grasp_and_suction_points, view_score, view_npy_open3d

execute_suction_bash = './bash/run_robot_suction.sh'
execute_grasp_bash = './bash/run_robot_grasp.sh'

class Mode():
    def __init__(self):
        self.simulation=simulation_mode
        self.report_result=report_result

class Setting():
    def __init__(self):
        self.use_gripper=use_gripper
        self.use_suction=use_suction
        self.chances=chances_ref
        self.shuffling_probability=shuffling_probability

class View():
    def __init__(self):
        self.grasp_points=view_grasp_suction_points
        self.scores=view_score_gradient
        self.action=view_action

def prioritize_candidates(gripper_scores, suction_scores, gripper_mask, suction_mask):
    index_suction_cls = [i for i in range(len(suction_mask)) if suction_mask[i]]
    index_grasp_cls = [j for j in range(len(gripper_mask)) if gripper_mask[j]]

    score_grasp_value_ind = list((item * 1, index_grasp_cls[index], 0) for index, item in
                                 enumerate(gripper_scores[gripper_mask]))
    score_suction_value_ind_ = list((item * 1, index_suction_cls[index], 1) for index, item in
                                    enumerate(suction_scores[suction_mask]))

    candidates = score_grasp_value_ind + score_suction_value_ind_

    if len(candidates) > 1:
        candidates = sorted(candidates, reverse=True)

    return candidates

class GraspAgent():
    def __init__(self):
        '''configurations'''
        self.mode=Mode()
        self.view=View()
        self.setting=Setting()

        '''models'''
        self.gripper_D_model = None
        self.suction_D_model = None
        self.gripper_G_model = None
        self.Suction_G_model = None
        self.suction_scope_model = None
        self.gripper_scope_model = None

        '''ctime records'''
        # self.c_time_list=[-1,-1,-1,-1,-1,-1]

        '''modalities'''
        self.point_clouds = None
        self.depth=None
        self.rgb=None

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
        # new_c_time=smbclient.path.getctime(suction_quality_model_state_path)
        # if self.c_time_list[0] != new_c_time:
        self.suction_D_model = initialize_model_state(suction_quality_net(), suction_quality_model_state_path)
            # self.c_time_list[0] = new_c_time
        pi.step(2)

        self.suction_scope_model = initialize_model_state(scope_net_vanilla(in_size=6), suction_scope_model_state_path)
        pi.step(3)

        self.gripper_D_model = initialize_model_state(gripper_quality_net(), gripper_quality_model_state_path)
        pi.step(4)

        self.gripper_scope_model=initialize_model_state(scope_net_vanilla(in_size=6),gripper_scope_model_state_path)
        pi.step(5)

        self.gripper_G_model = initialize_model_state(gripper_sampler_net(), gripper_sampler_path)
        pi.step(6)

        self.Suction_G_model = initialize_model_state(suction_sampler_net(), suction_sampler_model_state_path)
        pi.step(7)

        '''set evaluation mode'''
        self.gripper_G_model.eval()
        self.gripper_D_model.eval()
        self.suction_D_model.eval()
        self.Suction_G_model.eval()
        self.suction_scope_model.eval()
        self.gripper_scope_model.eval()
        pi.end()

    def model_inference(self,depth,rgb):
        self.depth=depth
        self.rgb=rgb
        depth_torch = torch.from_numpy(self.depth)[None, None, ...].to('cuda').float()

        '''depth to point clouds'''
        voxel_pc, mask = depth_to_point_clouds(self.depth, camera)
        voxel_pc = transform_to_camera_frame(voxel_pc, reverse=True)

        '''sample suction'''
        normals_pixels = self.Suction_G_model(depth_torch.clone())
        normals = normals_pixels.squeeze().permute(1, 2, 0)[mask] # [N,3]

        '''sample gripper'''
        poses_pixels = self.gripper_G_model(depth_torch.clone())
        poses = poses_pixels.squeeze().permute(1, 2, 0)[mask]

        '''suction scope'''
        positions = torch.from_numpy(voxel_pc).to('cuda').float()
        approach=normals.clone()
        approach[:,2]*=-1
        suction_scope = self.suction_scope_model(torch.cat([positions, approach], dim=-1)).squeeze().detach().cpu().numpy()
        suction_scope = np.clip(suction_scope, 0, 1)
        suction_scope[suction_scope>=scope_threshold]=1.
        suction_scope[suction_scope<scope_threshold]=0.0

        '''gripper scope'''
        gripper_approach=poses[:,0:3]
        distance=poses[:,-2:-1]*config.distance_scope
        transition=positions+distance*gripper_approach
        gripper_scope=self.suction_scope_model(torch.cat([transition, gripper_approach], dim=-1)).squeeze().detach().cpu().numpy()
        gripper_scope = np.clip(gripper_scope, 0, 1)
        gripper_scope[gripper_scope >= scope_threshold] = 1.
        gripper_scope[gripper_scope < scope_threshold] = 0.0

        '''suction quality'''
        suction_quality_score = self.suction_D_model(depth_torch.clone(), normals_pixels.clone()).squeeze()

        '''gripper quality'''
        gripper_quality_score = self.gripper_D_model(depth_torch.clone(), poses_pixels.clone()).squeeze()

        '''res u net processing'''
        suction_quality_score = suction_quality_score[mask].detach().cpu().numpy()
        gripper_quality_score = gripper_quality_score[mask].detach().cpu().numpy()

        '''final scores'''
        # suction_scores=suction_quality_score * suction_scope
        suction_scores= suction_quality_score
        # gripper_scores = gripper_quality_score #* gripper_scope
        gripper_scores = gripper_scope

        suction_scores = suction_scores.squeeze()
        gripper_scores = gripper_scores.squeeze()

        '''set limits'''
        suction_scores = suction_scores * suction_factor * int(self.setting.use_suction)
        gripper_scores = gripper_scores * gripper_factor * int(self.setting.use_gripper)

        self.gripper_poses = vectors_to_ratio_metrics(poses)
        self.normals = normals.detach().cpu().numpy()

        '''Masks'''
        spatial_mask=static_spatial_mask(voxel_pc)
        gripper_mask=spatial_mask & (gripper_scores>gripper_limit)
        suction_mask=spatial_mask & (suction_scores>suction_limit)

        '''Visualization'''
        if self.view.grasp_points: visualize_grasp_and_suction_points(suction_mask, gripper_mask,voxel_pc)
        if self.view.scores:
            view_score(voxel_pc, gripper_mask, gripper_scores)
            view_score(voxel_pc, suction_mask, suction_scores)

        # if view_sampled_suction:
        #     target_mask = get_spatial_mask(data)
        #     estimate_suction_direction(data.cpu().numpy().squeeze(), view=True, view_mask=target_mask,score=suction_score_pred)
        # if view_sampled_griper:
        #     dense_grasps_visualization(pc_with_normals.clone(), poses.clone(),grasp_score_pred.clone(),target_mask)
        # suction_score_pred[suction_score_pred < suction_limit] = -1
        # grasp_score_pred[grasp_score_pred<gripper_limit]=-1

        '''Release candidates'''
        self.candidates= prioritize_candidates(gripper_scores, suction_scores, gripper_mask, suction_mask)
        self.gripper_scores=gripper_scores
        self.suction_scores=suction_scores
        self.voxel_pc=voxel_pc

        return self.candidates, gripper_scores, suction_scores, voxel_pc

    def execute(self):
        state_ = ''
        self.n_candidates = len(self.candidates)
        T=np.eye(4)
        width=0
        normal=[1,0,0]
        if self.n_candidates== 0:
            self.actions.append(state_)
            self.states.append('No Object')
            self.data.append((width, T, normal, [0, 0, 0]))
            return self.actions, self.states, self.data
        # global chances_ref
        chances = self.setting.chances

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

                '''Gripper grasp'''
                if candidate_[-1] == 0:  # grasp

                    index = int(candidate_[1])

                    success, width, distance, T, target_point = \
                        gripper_processing(index, self.voxel_pc, self.gripper_poses, self.view.action)

                    print(f'prediction time = {time.time() - t}')

                    if not success:
                        if simulation_mode:
                            if chances == 0:
                                return self.actions, self.states, self.data
                            chances -= 1
                        continue

                    print('***********************use grasp***********************')
                    print('Gripper score = ', self.gripper_scores[index])

                    if simulation_mode:
                        if chances == 0:
                            return self.actions, self.states, self.data
                        chances -= 1
                        continue

                    action_ = 'grasp'

                    subprocess.run(execute_grasp_bash)
                    chances -= 1


                else:
                    '''Suction grasp'''
                    index = int(candidate_[1])

                    success, target_point, normal = suction_processing(index, self.voxel_pc, self.normals, self.view.action)


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
                    chances -= 1

                '''get robot feedback'''
                state_ = wait_for_feedback(state_)
                if action_ == 'grasp':
                    self.actions.append(action_)
                    self.states.append(state_)
                    self.data.append((width, T, normal, target_point))
                if action_ == 'suction':
                    self.actions.append(action_)
                    self.states.append(state_)
                    self.data.append((0, np.eye(4), normal, target_point))

                if state_ == 'Succeed' or state_ == 'reset' or chances == 0:
                    return self.actions, self.states, self.data

        else:
            print(Fore.RED, 'No feasible pose is found in the current forward pass', Fore.RESET)
            return self.actions, self.states, self.data

    def process_feedback(self,action_, state_, data_, img_grasp_pre, img_suction_pre):
        if state_ == 'Succeed' or state_ == 'reset': get_new_perception()
        if not self.mode.report_result: return

        img_suction_after, img_grasp_after = get_side_bins_images()

        if action_ == 'grasp' and state_ == 'Succeed':
            award = check_image_similarity(img_grasp_pre, img_grasp_after)
            if award is not None:
                save_grasp_sample(rgb=self.rgb,depth=self.depth,width=data_[0],transformation=data_[1],normal=data_[2],target_point=data_[3]
                                  ,use_gripper=True,use_suction=False,success=award)

        elif action_ == 'suction' and state_ == 'Succeed':
            award = check_image_similarity(img_suction_pre, img_suction_after)
            if award is not None:
                save_grasp_sample(rgb=self.rgb, depth=self.depth, width=data_[0], transformation=data_[1],
                                  normal=data_[2], target_point=data_[3]
                                  , use_gripper=False, use_suction=True, success=award)

        elif state_ == 'reset' or state_ == 'Failed':
            print(action_)

        else:
            print('No movement is executed, State: ', state_, ' ， Action： ', action_)

