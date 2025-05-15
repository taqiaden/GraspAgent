import subprocess
import smbclient.path

from explore_arms_scope import report_failed_gripper_path_plan, report_failed_suction_path_plan
from lib.dataset_utils import online_data2
import numpy as np
import torch
import trimesh
from colorama import Fore
import open3d as o3d
from Configurations.dynamic_config import get_int
from Online_data_audit.data_tracker2 import DataTracker2
from action import Action
from lib.IO_utils import save_pickle, save_data_to_server, save_to_server, load_pickle_from_server, save_image_to_server
from lib.grasp_utils import shift_a_distance
from lib.models_utils import number_of_parameters
from lib.report_utils import wait_indicator as wi
from Configurations.ENV_boundaries import bin_center, dist_allowance, knee_ref_elevation
from Configurations.config import distance_scope, gripper_width_during_shift, ip_address
from Configurations.run_config import simulation_mode, \
    suction_factor, gripper_factor, report_result, \
    enhance_gripper_firmness, single_arm_operation_mode, \
    enable_gripper_grasp, enable_gripper_shift, enable_suction_grasp, enable_suction_shift, \
    activate_segmentation_queries, activate_handover, only_handover, highest_elevation_to_grasp, \
    report_for_handover, report_for_shift, report_for_grasp, \
    zero_out_distance_when_collision, handover_quality_bias, suction_grasp_bias, gripper_grasp_bias, \
    sample_action_with_argmax, quality_exponent
from Online_data_audit.process_feedback import save_grasp_sample, grasp_data_counter_key
from check_points.check_point_conventions import GANWrapper, ModelWrapper
from lib.ROS_communication import deploy_action, read_robot_feedback, set_wait_flag
from lib.bbox import convert_angles_to_transformation_form
from lib.collision_unit import grasp_collision_detection
from lib.custom_print import my_print
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from lib.image_utils import check_image_similarity, view_image
from lib.pc_utils import numpy_to_o3d
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from models.action_net import ActionNet, action_module_key, action_module_key2
from models.scope_net import scope_net_vanilla, gripper_scope_module_key, suction_scope_module_key
from models.policy_net import PolicyNet, policy_module_key
from pose_object import vectors_to_ratio_metrics
from process_perception import get_side_bins_images, trigger_new_perception
from records.training_satatistics import MovingRate
from registration import camera
from training.learning_objectives.shift_affordnace import shift_execution_length
from training.policy_lr import  buffer_file, action_data_tracker_path
from training.ppo_memory import PPOMemory
from visualiztion import view_npy_open3d, vis_scene, dense_grasps_visualization

segmentation_result_file_path = ip_address + r'\taqiaden_hub\segmentation_query//segmentation_mask.npy'

online_data2=online_data2()

pr=my_print()

def get_unflatten_index(flat_index, ori_size):
    res = len(ori_size)*[0]
    for i in range(len(ori_size)-1, -1, -1):
        j = flat_index % ori_size[i]
        flat_index = flat_index // ori_size[i]
        res[i] = j
    return res

def get_shift_end_points(start_points):
    targets=torch.zeros_like(start_points)
    targets[:,0:2]+=torch.from_numpy(bin_center[0:2]).cuda()
    targets[:,2] += start_points[:,2]
    directions = targets - start_points
    end_points = start_points + ((directions * shift_execution_length) / torch.linalg.norm(directions,axis=-1,keepdims=True))
    return end_points

def masked_color(voxel_pc, score, pivot=0.5):
    mask_=score.cpu().numpy()>pivot if torch.is_tensor(score) else score>pivot
    colors = np.zeros_like(voxel_pc)
    colors[mask_]+= [0.5, 0.9, 0.5]
    colors[~mask_] += [0.52, 0.8, 0.92]
    return colors

def view_mask(voxel_pc, score, pivot=0.5):
    means_=voxel_pc.mean(axis=0)
    voxel_pc_t=voxel_pc-means_
    voxel_pc_t[:,1]*=-1
    colors=masked_color(voxel_pc_t, score, pivot=0.5)
    view_npy_open3d(voxel_pc_t, color=colors)

def multi_mask_view(pc, scores_list, pivot=0.5):
    colors=[]
    for i in range(len(scores_list)):
        colors.append(masked_color(pc, scores_list[i], pivot=pivot))

    stacked_pc=np.repeat(pc[np.newaxis,...],len(scores_list),0)
    colors=np.concatenate(colors,axis=0)

    pc_list=[]
    for i in range(len(scores_list)):
        stacked_pc[i,:,0]+=i*0.5
        pc_list.append(stacked_pc[i])

    cat_pc=np.concatenate(pc_list,axis=0)

    view_npy_open3d(cat_pc,color=colors)

class GraspAgent():
    def __init__(self):
        '''models'''
        self.args=None
        self.action_net = None
        self.policy_net = None
        self.suction_arm_reachability_net = None
        self.gripper_arm_reachability_net = None

        '''
        handover_state_
        None: no handover
        0: first attempt of handover
        1: second attempt of handover
        2: object has been released
        3: no object
        '''
        self.last_handover_action=None

        self.buffer=online_data2.load_pickle(buffer_file) if online_data2.file_exist(buffer_file) else PPOMemory()
        self.data_tracker = DataTracker2(name=action_data_tracker_path, list_size=10)
        # self.buffer=PPOMemory()
        # print(self.buffer.episodic_file_ids)
        # print(self.buffer.rewards)
        # print(self.buffer.is_end_of_episode)
        # print(self.buffer.advantages)
        # exit()

        self.gripper_usage_rate=MovingRate('gripper_usage',min_decay=0.01)
        self.suction_usage_rate=MovingRate('suction_usage',min_decay=0.01)
        self.double_grasp_rate=MovingRate('double_grasp',min_decay=0.01)
        self.gripper_grasp_success_rate=MovingRate('gripper_grasp_success',min_decay=0.01)
        self.suction_grasp_success_rate=MovingRate('suction_grasp_success',min_decay=0.01)
        self.shift_rate=MovingRate('shift',min_decay=0.01)
        self.planning_success_rate=MovingRate('planning_success',min_decay=0.01)

        self.segmentation_result_time_stamp=None
        self.buffer_modify_alert=False
        self.data_tracker_modify_alert=False

        '''Modalities'''
        self.point_clouds = None
        self.depth=None
        self.rgb=None

        self.rollback()

        '''track task sequence'''
        self.run_sequence=0

        self.remove_seg_file()

    def rollback(self):
        self.gripper_collision_mask=None
        self.gripper_poses_5 = None
        self.gripper_poses_7 = None
        self.gripper_grasp_mask = None
        self.suction_grasp_mask = None
        self.gripper_shift_mask = None
        self.suction_shift_mask = None
        self.voxel_pc = None
        self.normals = None
        self.q_value = None
        self.clear_policy = None
        self.shift_end_points = None
        self.valid_actions_mask = None
        self.first_action_mask = None
        self.target_object_mask = None
        self.valid_actions_on_target_mask = None
        self.mask_numpy=None
        self.seg_mask=None
        self.seize_policy=None
        self.tmp_occupation_mask=None
        self.preferred_placement_side=False
        self.handover_mask=None

    def print_report(self):
        print(Fore.BLUE)
        print(f'Samples dictionary containes {len(self.data_tracker)} key values pairs')
        print(f'Episodic buffer size = {len(self.buffer)} ')
        print(f'Non episodic buffer size = {len(self.buffer.non_episodic_file_ids)} ')

        latest_id=get_int(grasp_data_counter_key)
        print(f'Latest saved id is {latest_id}')

        self.gripper_usage_rate.view()
        self.suction_usage_rate.view()
        self.double_grasp_rate.view()
        self.gripper_grasp_success_rate.view()
        self.suction_grasp_success_rate.view()
        self.shift_rate.view()
        self.planning_success_rate.view()
        print(Fore.RESET)

    @property
    def gripper_approach(self):
        approach=self.gripper_poses_7[:,0:3].clone()
        approach[:,2]*=-1
        return approach

    @property
    def suction_approach(self):
        return self.normals*-1.

    def publish_segmentation_query(self,args):
        self.args=args
        TEXT_PROMPT = self.args.text_prompt
        if TEXT_PROMPT.strip()!='':
            print(f'Publish segmentation query for ({TEXT_PROMPT})')
            segmentation_query_file_path=ip_address+r'\taqiaden_hub\segmentation_query//text_prompts.txt'
            segmentation_image_path=ip_address+r'\taqiaden_hub\segmentation_query//seg_image.jpg'
            save_to_server(segmentation_query_file_path,TEXT_PROMPT,binary_mode=False)

            '''save image'''
            save_image_to_server(segmentation_image_path,self.rgb)
            view_image(self.rgb)
    def remove_seg_file(self):
        if smbclient.path.isfile(segmentation_result_file_path) or smbclient.path.islink(segmentation_result_file_path):
            smbclient.unlink(segmentation_result_file_path)

    def retrieve_segmentation_mask(self):
        TEXT_PROMPT = self.args.text_prompt
        if TEXT_PROMPT.strip() != '':
            wait = wi('Waiting for segmentation result')
            while True:
                if smbclient.path.exists(segmentation_result_file_path):
                    segmentation_mask=load_pickle_from_server(segmentation_result_file_path,allow_pickle=False)
                    if segmentation_mask.shape==(1,):
                        print(Fore.RED,'The queried object is not found!',Fore.RESET)
                    self.remove_seg_file()
                    self.seg_mask=(torch.from_numpy(segmentation_mask)[None,None,...]>0.5).cuda()
                    break
                else:
                    wait.step(0.5)

    def initialize_check_points(self):
        pi = progress_indicator('Loading check points  ', max_limit=5)

        pi.step(1)

        action_net = GANWrapper(action_module_key2, ActionNet)
        action_net.ini_generator(train=False)
        self.action_net = action_net.generator

        pi.step(2)

        policy_net = ModelWrapper(model=PolicyNet(), module_key=policy_module_key)
        policy_net.ini_model(train=False)
        self.model_time_stamp=policy_net.model_time_stamp()
        self.policy_net = policy_net.model

        pi.step(3)

        gripper_scope = ModelWrapper(model=scope_net_vanilla(in_size=6), module_key=gripper_scope_module_key)
        gripper_scope.ini_model(train=False)
        self.gripper_arm_reachability_net = gripper_scope.model

        pi.step(4)

        suction_scope = ModelWrapper(model=scope_net_vanilla(in_size=6), module_key=suction_scope_module_key)
        suction_scope.ini_model(train=False)
        self.suction_arm_reachability_net=suction_scope.model

        pi.step(5)

        pi.end()

        self.models_report()

    def models_report(self):
        action_model_size=number_of_parameters(self.action_net)
        policy_model_size=number_of_parameters(self.policy_net)
        gripper_arm_reachability_model_size=number_of_parameters(self.policy_net)
        # suction_arm_reachability_model_size=number_of_parameters(self.suction_arm_reachability_net)
        print(f'Models initiated:')
        print(f'Number of parameters:')
        pr.step_f(f'action net : {action_model_size}')
        pr.print(f'policy net : {policy_model_size}')
        pr.print(f'reachability nets : {gripper_arm_reachability_model_size} * 2')
        pr.step_b()


    def get_suction_grasp_reachability(self,positions,normals):
        approach=-normals.clone()
        suction_scope = self.suction_arm_reachability_net(torch.cat([positions, approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        return suction_scope

    def get_suction_shift_reachability(self,positions):
        suction_scope_a = self.suction_arm_reachability_net(torch.cat([positions, self.suction_approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        if self.shift_end_points is None:
            self.shift_end_points = get_shift_end_points(positions)
        suction_scope_b = self.suction_arm_reachability_net(torch.cat([self.shift_end_points, self.suction_approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        result=torch.stack([suction_scope_a,suction_scope_b],dim=-1)
        result,_=torch.min(result,dim=-1)
        return result

    def get_gripper_grasp_reachability(self,positions):
        distance=self.gripper_poses_7[:,-2:-1]*distance_scope
        transition=positions+distance*self.gripper_approach
        gripper_scope=self.gripper_arm_reachability_net(torch.cat([transition, self.gripper_approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        return gripper_scope

    def get_gripper_shift_reachability(self,positions):
        gripper_scope_a=self.gripper_arm_reachability_net(torch.cat([positions, self.gripper_approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        self.shift_end_points = get_shift_end_points(positions)
        gripper_scope_b=self.gripper_arm_reachability_net(torch.cat([self.shift_end_points, self.gripper_approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        result = torch.stack([gripper_scope_a, gripper_scope_b], dim=-1)
        result, _ = torch.min(result, dim=-1)
        return result

    def next_action(self,sample_from_target_actions=False):
        selected_policy = self.seize_policy if sample_from_target_actions else self.clear_policy
        mask_=self.valid_actions_on_target_mask if sample_from_target_actions else self.valid_actions_mask
        if self.tmp_occupation_mask is not None: mask_=mask_ & self.tmp_occupation_mask
        # print('test',selected_policy[mask_].shape,sample_from_target_actions)
        if sample_action_with_argmax:
            flattened_action_index=(selected_policy*mask_).argmax()
        else:
            dist=MaskedCategorical(probs=selected_policy,mask=mask_)
            flattened_action_index= dist.sample()

        if sample_from_target_actions:
            probs=None
            value=None
            flattened_action_index = torch.squeeze(flattened_action_index).item()

            self.valid_actions_on_target_mask[flattened_action_index] = False
        else:
            probs = torch.squeeze(dist.log_prob(flattened_action_index)).item()
            flattened_action_index = torch.squeeze(flattened_action_index).item()
            value = self.q_value[flattened_action_index]
            value = torch.squeeze(value).item()
        if self.last_handover_action is None:
            self.valid_actions_mask[flattened_action_index]=False
        return flattened_action_index, probs, value

    def inputs(self,depth,rgb,args):
        self.depth=depth
        self.rgb=rgb
        self.args=args

    def view_mask_as_2dimage(self):
        # view_image(self.seg_mask[0,0].cpu().numpy().astype(np.float64))
        view_image(self.mask_numpy.astype(np.float64))

    def background_manual_correction(self,background_class,voxel_pc_tensor):
        lower_bound_mask=(self.voxel_pc[:,-1]<0.045) | (self.voxel_pc[:,0]<0.25) | (self.voxel_pc[:,0]>0.6)
        background_class[lower_bound_mask]=1.0
        min_elevation=voxel_pc_tensor[background_class<0.5,-1].min().item()
        background_class[self.voxel_pc[:,-1]<min_elevation+0.005]=1.0
        return background_class

    def gripper_arm_mask_during_handover(self):
        cylindrical_mask=np.linalg.norm(self.voxel_pc[:,[0,2]]-np.array([0.45,0.24])[np.newaxis],axis=-1)<0.025
        rectangular_mask=(np.abs( self.voxel_pc[:,0]-0.45)<0.08) & (np.abs( self.voxel_pc[:,2]-0.24)<0.08) & (self.voxel_pc[:, 1] < -0.09)
        cylindrical_mask=cylindrical_mask & (self.voxel_pc[:, 1] < -0.025)
        handed_object_mask=(self.voxel_pc[:, 1] > -0.1) & (self.voxel_pc[:, 2] > 0.24) \
                         & (self.voxel_pc[:,1]<0.1) & (~cylindrical_mask) & (~rectangular_mask)
        # view_mask(self.voxel_pc, ~cylindrical_mask)
        # view_mask(self.voxel_pc, ~rectangular_mask)
        # view_mask(self.voxel_pc, (self.voxel_pc[:, 1] > -0.17)& (self.voxel_pc[:,1]<0.17))
        # view_mask(self.voxel_pc, (self.voxel_pc[:, 2] > 0.14))

        '''detect object'''
        no_object=self.voxel_pc[handed_object_mask].shape[0]<50
        # print(self.voxel_pc[handed_object_mask].shape[0])
        # view_mask(self.voxel_pc, handed_object_mask)

        return handed_object_mask,no_object

    def suction_arm_mask_during_handover(self):
        # suction_ref_point=np.array([0.45,0.01,0.24])
        cylindrical_mask=np.linalg.norm(self.voxel_pc[:,[0,2]]-np.array([0.45,0.24])[np.newaxis],axis=-1)<0.015
        handed_object_mask=(self.voxel_pc[:,1]<0.09) & (self.voxel_pc[:,2]>0.24) & (~cylindrical_mask) \
                          & (self.voxel_pc[:,1]>-0.1)

        '''detect object'''
        no_object=self.voxel_pc[handed_object_mask].shape[0]<50
        # view_mask(self.voxel_pc, handed_object_mask)
        # view_npy_open3d(self.voxel_pc[(handed_object_mask)])

        return   handed_object_mask,no_object

    def models_inference(self):
        pr.title('Inference')
        depth_torch = torch.from_numpy(self.depth)[None, None, ...].to('cuda').float()
        rgb_torch = torch.from_numpy(self.rgb).permute(2, 0, 1)[None, ...].to('cuda').float()

        '''action net output'''
        gripper_pose, suction_direction, griper_collision_classifier, suction_seal_classifier, shift_appealing \
            , background_class, action_depth_features = self.action_net(depth_torch.clone(), clip=True)

        '''depth to point clouds'''
        self.voxel_pc, mask = depth_to_point_clouds(self.depth, camera)
        self.voxel_pc = transform_to_camera_frame(self.voxel_pc, reverse=True)
        voxel_pc_tensor = torch.from_numpy(self.voxel_pc).to('cuda').float()

        '''pixel-wise to point-wise sampling'''
        self.normals = suction_direction.squeeze().permute(1, 2, 0)[mask]  # [N,3]
        self.gripper_poses_7 = gripper_pose.squeeze().permute(1, 2, 0)[mask]

        '''target mask'''
        if activate_segmentation_queries and self.seg_mask.shape == background_class.shape:
            self.target_object_mask = (background_class <= 0.5) & self.seg_mask
        else:
            self.target_object_mask = (background_class <= 0.5)
        self.target_object_mask[0, 0][~mask] *= False
        self.mask_numpy = self.target_object_mask.squeeze().cpu().numpy()

        '''policy net output'''
        griper_grasp_score, suction_grasp_score, \
        q_value, clear_policy, handover_scores= \
            self.policy_net(rgb_torch, depth_torch, gripper_pose, suction_direction, self.target_object_mask.float())
        return griper_grasp_score, suction_grasp_score, \
        q_value, clear_policy, handover_scores,\
            griper_collision_classifier,suction_seal_classifier,\
            background_class,shift_appealing,mask,voxel_pc_tensor

    def grasp_reachablity(self,voxel_pc_tensor):
        '''grasp reachability'''
        suction_grasp_scope = self.get_suction_grasp_reachability(voxel_pc_tensor, self.normals)
        gripper_grasp_scope = self.get_gripper_grasp_reachability(voxel_pc_tensor)

        return gripper_grasp_scope,suction_grasp_scope
    def shift_reachablity(self,voxel_pc_tensor):
        '''shift reachability'''
        gripper_shift_scope = self.get_gripper_shift_reachability(voxel_pc_tensor)
        suction_shift_scope = self.get_suction_shift_reachability(voxel_pc_tensor)

        '''grasp/shift with the arm of best reachability'''
        gripper_shift_scope[suction_shift_scope > gripper_shift_scope] *= 0.
        suction_shift_scope[gripper_shift_scope > suction_shift_scope] *= 0.

        return gripper_shift_scope, suction_shift_scope

    def main_dense_processing(self,shift_appealing_mask,gripper_shift_reachablity_mask,suction_shift_reachablity_mask,
                                griper_grasp_score,suction_grasp_score,handover_scores):
        '''shift actions'''
        self.gripper_shift_mask = (shift_appealing_mask * gripper_shift_reachablity_mask * enable_gripper_shift)
        self.suction_shift_mask = (shift_appealing_mask * suction_shift_reachablity_mask * enable_suction_shift)

        # view_mask(voxel_pc_,(shift_appealing>0.5))
        # view_mask(voxel_pc_,(suction_shift_scope>0.5))
        # view_mask(voxel_pc_,self.suction_shift_mask)

        '''change gripper pose represntation'''
        self.gripper_poses_5 = vectors_to_ratio_metrics(self.gripper_poses_7.clone())  # #3 angles, width, dist

        '''initialize the seize policy'''
        self.seize_policy = torch.zeros_like(self.clear_policy)
        self.seize_policy[:, 0] = griper_grasp_score
        self.seize_policy[:, 1] = suction_grasp_score

        '''initialize valid actions mask'''
        self.valid_actions_mask = torch.zeros_like(self.q_value, dtype=torch.bool)
        self.valid_actions_mask[:, 0].masked_fill_(self.gripper_grasp_mask, True)
        self.valid_actions_mask[:, 1].masked_fill_(self.suction_grasp_mask, True)
        self.valid_actions_mask[:, 2].masked_fill_(self.gripper_shift_mask, True)
        self.valid_actions_mask[:, 3].masked_fill_(self.suction_shift_mask, True)

        '''initialize valid actions mask on target'''
        self.valid_actions_on_target_mask = torch.zeros_like(self.q_value, dtype=torch.bool)
        gripper_actions_mask_on_target = self.gripper_grasp_mask & self.target_object_mask
        suction_actions_mask_on_target = self.suction_grasp_mask & self.target_object_mask
        self.valid_actions_on_target_mask[:, 0].masked_fill_(gripper_actions_mask_on_target, True)
        self.valid_actions_on_target_mask[:, 1].masked_fill_(suction_actions_mask_on_target, True)

        '''handover processing'''
        if activate_handover: self.handover_processing(handover_scores)

        self.valid_actions_mask = self.valid_actions_mask.reshape(-1)

        '''to numpy'''
        self.normals=self.normals.cpu().numpy()

        '''flatten'''
        self.q_value = self.q_value.reshape(-1)
        self.clear_policy = self.clear_policy.reshape(-1)
        self.seize_policy=self.seize_policy.reshape(-1)

        self.valid_actions_on_target_mask = self.valid_actions_on_target_mask.reshape(-1)

    def handover_from_gripper_dense_processing(self,suction_grasp_score):
        '''mask gripper arm body'''
        handed_object_mask,no_object = self.gripper_arm_mask_during_handover()
        if no_object:
            '''no object for handover'''
            self.last_handover_action.handover_state = 3  # drop
            return
        self.suction_grasp_mask = self.suction_grasp_mask * torch.from_numpy(handed_object_mask).cuda()
        self.gripper_grasp_mask *= False

        '''initialize the seize policy'''
        self.seize_policy = torch.zeros_like(self.clear_policy)
        self.seize_policy[:, 1] = suction_grasp_score

        '''initialize valid actions mask on target'''
        self.valid_actions_on_target_mask = torch.zeros_like(self.q_value, dtype=torch.bool)
        suction_actions_mask_on_target = self.suction_grasp_mask  # & self.target_object_mask
        self.valid_actions_on_target_mask[:, 1].masked_fill_(suction_actions_mask_on_target, True)

        '''to numpy'''
        self.normals=self.normals.cpu().numpy()

        '''flatten'''
        self.seize_policy=self.seize_policy.reshape(-1)
        self.valid_actions_on_target_mask = self.valid_actions_on_target_mask.reshape(-1)

    def handover_from_suction_dense_processing(self,griper_grasp_score):
        '''mask suction arm body'''
        handed_object_mask,no_object = self.suction_arm_mask_during_handover()
        if no_object:
            '''no object for handover'''
            self.last_handover_action.handover_state = 3
            return
        self.gripper_grasp_mask = self.gripper_grasp_mask * torch.from_numpy(handed_object_mask).cuda()
        self.suction_grasp_mask *= False

        '''change gripper pose represntation'''
        self.gripper_poses_5 = vectors_to_ratio_metrics(self.gripper_poses_7.clone())  # #3 angles, width, dist

        '''initialize the seize policy'''
        self.seize_policy = torch.zeros_like(self.clear_policy)
        self.seize_policy[:, 0] = griper_grasp_score


        '''initialize valid actions mask on target'''
        self.valid_actions_on_target_mask = torch.zeros_like(self.q_value, dtype=torch.bool)
        gripper_actions_mask_on_target = self.gripper_grasp_mask  # & self.target_object_mask
        self.valid_actions_on_target_mask[:, 0].masked_fill_(gripper_actions_mask_on_target, True)

        '''to numpy'''
        self.normals=self.normals.cpu().numpy()

        '''flatten'''
        self.seize_policy=self.seize_policy.reshape(-1)
        self.valid_actions_on_target_mask = self.valid_actions_on_target_mask.reshape(-1)

    def sample_masked_actions(self):
        '''Action and Policy network inference'''
        griper_grasp_score, suction_grasp_score, \
        q_value, clear_policy, handover_scores, \
        griper_collision_classifier, suction_seal_classifier, \
        background_class, shift_appealing,mask,voxel_pc_tensor=\
        self.models_inference()

        '''Reachability inference'''
        gripper_grasp_scope,suction_grasp_scope=self.grasp_reachablity(voxel_pc_tensor)
        gripper_shift_scope, suction_shift_scope=self.shift_reachablity( voxel_pc_tensor)

        '''add exponent term'''
        if quality_exponent!=1.:
            griper_grasp_score=griper_grasp_score**quality_exponent
            suction_grasp_score = suction_grasp_score ** quality_exponent
            handover_scores = handover_scores ** quality_exponent

        '''clip quality scores'''
        griper_grasp_score=torch.clip(griper_grasp_score,0.,1.)
        suction_grasp_score=torch.clip(suction_grasp_score,0.,1.)
        handover_scores=torch.clip(handover_scores,0.,1.)

        '''bias quality scores'''
        griper_grasp_score += gripper_grasp_bias
        suction_grasp_score += suction_grasp_bias
        handover_scores += handover_quality_bias

        '''pixel-wise to point-wise'''
        griper_object_collision_classifier=griper_collision_classifier[0,0][mask]
        griper_bin_collision_classifier=griper_collision_classifier[0,1][mask]
        griper_grasp_score=griper_grasp_score.squeeze()[mask]
        suction_seal_classifier=suction_seal_classifier.squeeze()[mask]
        suction_grasp_score=suction_grasp_score.squeeze()[mask]
        background_class=background_class.squeeze()[mask]
        shift_appealing=shift_appealing.squeeze()[mask]
        # view_image(self.target_object_mask[0,0].cpu().numpy().astype(np.float64))
        self.target_object_mask=self.target_object_mask.squeeze()[mask]
        self.q_value=q_value.squeeze().permute(1,2,0)[mask]
        self.clear_policy=clear_policy.squeeze().permute(1,2,0)[mask]
        handover_scores=handover_scores.squeeze().permute(1,2,0)[mask]

        # view_mask(self.voxel_pc, suction_seal_classifier>0.5 )


        '''correct background mask'''
        # if self.last_handover_action is None:
        #     background_class=self.background_manual_correction(background_class,voxel_pc_tensor)
        # view_mask(self.voxel_pc, background_class < 0.5)
        '''actions masks'''
        object_mask=background_class<0.5
        # view_mask(self.voxel_pc, object_mask )
        if self.last_handover_action is None:
            '''set highest grasp elevation'''
            object_mask=object_mask & (voxel_pc_tensor[:,2]<highest_elevation_to_grasp)
        gripper_grasp_reachablity_mask=gripper_grasp_scope>0.5
        self.gripper_collision_mask=(griper_object_collision_classifier<0.5) & (griper_bin_collision_classifier<0.5)
        gripper_quality_mask=griper_grasp_score*gripper_factor>0.5
        suction_grasp_reachablity_mask=suction_grasp_scope>0.5
        seal_quality_mask=suction_seal_classifier>0.5
        suction_quality_mask=suction_grasp_score*suction_factor>0.5
        shift_appealing_mask=shift_appealing>0.5
        gripper_shift_reachablity_mask=gripper_shift_scope>0.5
        suction_shift_reachablity_mask=suction_shift_scope>0.5


        if self.last_handover_action is not None :
            self.gripper_grasp_mask = (object_mask * gripper_grasp_reachablity_mask
                                       * self.gripper_collision_mask
                                       * gripper_quality_mask )
            self.suction_grasp_mask = (object_mask * suction_grasp_reachablity_mask
                                       * seal_quality_mask
                                       * suction_quality_mask )
            if self.last_handover_action.gripper_at_home_position==False:
                print('Process handover from gripper to suction')
                # proceede with handover sequence
                self.handover_from_gripper_dense_processing(suction_grasp_score)
            else:
                print('Process handover from suction to gripper')
                # proceede with handover sequence
                self.handover_from_suction_dense_processing(griper_grasp_score)
        else:
            self.gripper_grasp_mask = (object_mask * gripper_grasp_reachablity_mask
                                       * self.gripper_collision_mask
                                       * gripper_quality_mask * enable_gripper_grasp)
            self.suction_grasp_mask = (object_mask * suction_grasp_reachablity_mask
                                       * seal_quality_mask
                                       * suction_quality_mask * enable_suction_grasp)

            '''Arms at home position'''
            # proceede with normal inference
            self.main_dense_processing(shift_appealing_mask,gripper_shift_reachablity_mask,suction_shift_reachablity_mask,
                                griper_grasp_score,suction_grasp_score,handover_scores)



    def handover_processing(self,handover_scores):
        self.handover_mask=handover_scores>0.5
        if only_handover:
            self.valid_actions_on_target_mask[:, 0:2].masked_fill_(~self.handover_mask,False)

        elif self.args.placement_bin=='g':
            '''place at gripper side container'''
            self.handover_mask[:,0]=False
            if activate_handover==False:self.handover_mask[:,1]=False
            # masked all suctions on target that can not allow a handover to the gripper arm
            self.valid_actions_on_target_mask[:, 1].masked_fill_(~self.handover_mask[:,1],False)
        elif self.args.placement_bin=='s':
            ''''place at the suction side container'''
            self.handover_mask[:,1]=False
            if activate_handover==False:self.handover_mask[:,0]=False
            # masked all gripper grasps on target that can not allow a handover to the suction arm
            self.valid_actions_on_target_mask[:, 0].masked_fill_(~self.handover_mask[:,0],False)
        else:
            self.handover_mask.fill_(False)

    def report_current_scene_metrics(self):
        print(Fore.CYAN)

        if self.last_handover_action is None:
            self.valid_actions_mask=self.valid_actions_mask.reshape(-1,4)

            '''clear actions'''
            n_grasps = torch.count_nonzero(self.valid_actions_mask[:, 0:2])
            n_shifts = torch.count_nonzero(self.valid_actions_mask[:, 2:4])
            print(f'Total action space includes {n_grasps} grasps and {n_shifts} shifts')
            print(
                f'Total available grasps are  {self.valid_actions_mask[:, 0].sum()} using gripper and {self.valid_actions_mask[:, 1].sum()} using suction')
            print(
                f'Total available shifts are  {self.valid_actions_mask[:, 2].sum()} using gripper and {self.valid_actions_mask[:, 3].sum()} using suction')
            self.valid_actions_mask = self.valid_actions_mask.reshape(-1)
        else:
            if self.last_handover_action.handover_state!=3:
                self.valid_actions_on_target_mask=self.valid_actions_on_target_mask.reshape(-1,4)
                '''on target'''
                print(
                    f'Available grasps on the target object/s is {(self.valid_actions_on_target_mask[:, 0]).sum()} for gripper and {self.valid_actions_on_target_mask[:, 1].sum()} for suction')
                self.valid_actions_on_target_mask=self.valid_actions_on_target_mask.reshape(-1)

        print(Fore.RESET)

    def view_predicted_normals(self):
        view_npy_open3d(pc=self.voxel_pc,normals=self.normals)

    def dense_view(self,view_gripper_sampling=False):
        self.view_valid_actions_mask()
        if view_gripper_sampling:
            dense_grasps_visualization(self.voxel_pc, self.gripper_poses_7,
                                       view_mask= self.target_object_mask & self.gripper_collision_mask,view_all=False)

        # multi_mask_view(self.voxel_pc,[self.gripper_grasp_mask,self.suction_grasp_mask,self.gripper_shift_mask,self.suction_shift_mask])
        # view_mask(self.voxel_pc, self.gripper_grasp_mask, pivot=0.5)
        # view_mask(self.voxel_pc, self.suction_grasp_mask, pivot=0.5)
        # view_mask(self.voxel_pc, self.gripper_shift_mask, pivot=0.5)
        # view_mask(self.voxel_pc, self.suction_shift_mask, pivot=0.5)

    def actions_view(self,first_action_obj:Action,second_action_obj):
        if self.last_handover_action is not None and first_action_obj.handover_state in {1,3}: return

        scene_list = []

        if first_action_obj.is_executable and second_action_obj.is_executable:
            '''dual pose view'''
            first_pose_mesh=first_action_obj.pose_mesh2()
            if first_pose_mesh is not None: scene_list+=first_pose_mesh

            second_pose_mesh=second_action_obj.pose_mesh2()
            if second_pose_mesh is not None: scene_list+=second_pose_mesh
        else:
            '''single pose view'''
            first_pose_mesh = first_action_obj.pose_mesh()
            if first_pose_mesh is not None: scene_list += [first_pose_mesh]

            second_pose_mesh = second_action_obj.pose_mesh()
            if second_pose_mesh is not None: scene_list += [second_pose_mesh]

        masked_colors = np.ones_like(self.voxel_pc) * [0.52, 0.8, 0.92]
        if self.first_action_mask is not None:
            masked_colors[self.first_action_mask] /=1.1
        pcd = numpy_to_o3d(pc=self.voxel_pc, color=masked_colors)
        scene_list.append(pcd)

        o3d.visualization.draw_geometries(scene_list)

    def increase_gripper_penetration_distance(self,T_d,width,distance,step_factor=0.5):
        step = dist_allowance *step_factor
        n = max(int((distance_scope - distance) / step), 10)
        for i in range(n):
            T_d_new = shift_a_distance(T_d, step).copy()
            collision_intensity2 = grasp_collision_detection(T_d_new, width, self.voxel_pc, visualize=False,
                                                             with_allowance=False)
            if collision_intensity2 == 0:
                T_d = T_d_new
            else:
                break
        return T_d

    def gripper_grasp_processing(self,action_obj,  view=False):
        target_point = self.voxel_pc[action_obj.point_index]
        relative_pose_5 = self.gripper_poses_5[action_obj.point_index]
        T_d, width, distance = convert_angles_to_transformation_form(relative_pose_5, target_point)




        collision_intensity = grasp_collision_detection(T_d, width, self.voxel_pc, visualize=False)
        if collision_intensity==0 and enhance_gripper_firmness:
            T_d=self.increase_gripper_penetration_distance(T_d,width,distance,step_factor=0.5)
        elif zero_out_distance_when_collision:
            '''apply zero penetration distance'''
            T_d = shift_a_distance(T_d, - distance).copy()
            collision_intensity = grasp_collision_detection(T_d, width, self.voxel_pc, visualize=False)
            if collision_intensity==0 and enhance_gripper_firmness:
                T_d=self.increase_gripper_penetration_distance(T_d,width,distance,step_factor=0.25)

        # T_d, distance, width, collision_intensity = local_exploration(T_d, width, distance, self.voxel_pc,
        #                                                               exploration_attempts=5,
        #                                                               explore_if_collision=False,
        #                                                               view_if_sucess=view_masked_grasp_pose,
        #                                                               explore=activate_exploration)
        action_obj.is_executable= collision_intensity == 0
        action_obj.width=width
        action_obj.transformation=T_d


        if view: vis_scene(T_d, width, npy=self.voxel_pc)

    def gripper_shift_processing(self,action_obj):
        normal = self.normals[action_obj.point_index]
        target_point = self.voxel_pc[action_obj.point_index]

        v0 = np.array([1, 0, 0])
        a = trimesh.transformations.angle_between_vectors(v0, -normal)
        b = trimesh.transformations.vector_product(v0, -normal)
        T_d = trimesh.transformations.rotation_matrix(a, b)
        T_d[:3, 3] = target_point.T

        action_obj.width=gripper_width_during_shift
        action_obj.transformation=T_d

        has_collision = grasp_collision_detection(T_d, gripper_width_during_shift, self.voxel_pc, visualize=False,allowance=0.01)
        # if has_collision:
        #     grasp_collision_detection(T_d, gripper_width_during_shift, self.voxel_pc, visualize=True,allowance=0.01)

        action_obj.is_executable=not has_collision

    def suction_processing(self,action_obj):
        normal = self.normals[action_obj.point_index]
        target_point = self.voxel_pc[action_obj.point_index]

        v0 = np.array([1, 0, 0])
        a = trimesh.transformations.angle_between_vectors(v0, -normal)
        b = trimesh.transformations.vector_product(v0, -normal)
        T = trimesh.transformations.rotation_matrix(a, b)
        T[:3, 3] = target_point.T

        action_obj.transformation=T


        has_collision = grasp_collision_detection(T, gripper_width_during_shift, self.voxel_pc, visualize=False,allowance=0.01)
        # if has_collision:
        #     grasp_collision_detection(T, gripper_width_during_shift, self.voxel_pc, visualize=True,allowance=0.01)

        action_obj.is_executable=not has_collision

    def process_action(self,action_obj):
        if action_obj.is_grasp:
            if action_obj.use_gripper_arm:
                self.gripper_grasp_processing(action_obj )
            else:
                self.suction_processing(action_obj)
        else:
            '''shift action'''
            action_obj.shift_end_point=self.shift_end_points[action_obj.point_index]
            if action_obj.use_gripper_arm:
                self.gripper_shift_processing(action_obj)
            else:
                self.suction_processing(action_obj)

    def get_dense_knee_extremes(self,approach_vectors):
        res_elevation = knee_ref_elevation - self.voxel_pc[:,2]
        arm_knee_margin = res_elevation / (- approach_vectors[:,2] )
        arm_knee_margin=arm_knee_margin.reshape(-1,1)
        dense_extreme = (self.voxel_pc - approach_vectors * arm_knee_margin)
        assert not np.isnan(dense_extreme).any(), f'{dense_extreme}'
        return dense_extreme

    def mask_arm_occupancy(self, action_obj):
        tmp_occupation_mask=torch.ones_like(self.valid_actions_mask)
        tmp_occupation_mask=tmp_occupation_mask.reshape(-1,4)

        '''mask occupied arm'''
        if action_obj.use_gripper_arm:
            tmp_occupation_mask[:, [0, 2]]=False
        else:
            tmp_occupation_mask[:, [1, 3]]=False

        '''mask occupied space'''
        minimum_safety_margin=0.07
        knee_threeshold=0.2
        x_dist = np.abs(self.voxel_pc[:, 0] - action_obj.target_point[0])
        y_dist = np.abs(self.voxel_pc[:, 1] - action_obj.target_point[1])
        dist_mask=(x_dist<minimum_safety_margin) & (y_dist<minimum_safety_margin)

        arm_knee_extreme=action_obj.knee_extreme_point[1]

        if action_obj.use_gripper_arm:
            minimum_margin = action_obj.target_point[1] - minimum_safety_margin
            second_arm_extremes=self.get_dense_knee_extremes(self.suction_approach)
            occupancy_mask = (self.voxel_pc[:,1] < minimum_margin)\
                             | (arm_knee_extreme>(second_arm_extremes[:,1]-knee_threeshold))\
                             | dist_mask
        else:
            minimum_margin = action_obj.target_point[1] + minimum_safety_margin
            second_arm_extremes=self.get_dense_knee_extremes(self.gripper_approach.cpu().numpy())
            assert not np.isnan(second_arm_extremes).any(), f'{second_arm_extremes}'

            occupancy_mask = (self.voxel_pc[:,1] > minimum_margin)\
                             | (arm_knee_extreme<(second_arm_extremes[:,1]+knee_threeshold))\
                             |  dist_mask

        self.first_action_mask=occupancy_mask
        tmp_occupation_mask[occupancy_mask]=False
        tmp_occupation_mask=tmp_occupation_mask.reshape(-1)
        return tmp_occupation_mask, second_arm_extremes

    def view_valid_actions_mask(self):
        four_pc_stack = np.stack([self.voxel_pc, self.voxel_pc, self.voxel_pc, self.voxel_pc])
        four_pc_stack[1, :, 0] += 0.5
        four_pc_stack[2, :, 0] += 1.0
        four_pc_stack[3, :, 0] += 1.5

        colors = np.ones_like(four_pc_stack) * [0.5, 0.9, 0.5]
        self.valid_actions_mask=self.valid_actions_mask.reshape(-1,4)
        for i in range(4):
            mask_i = (self.valid_actions_mask[:, i] > 0.5).cpu().numpy()
            (colors[i])[~mask_i] *= 0.
            (colors[i])[~mask_i] += [0.9, 0.9, 0.9]

        self.valid_actions_mask=self.valid_actions_mask.reshape(-1)

        four_pc_stack = np.concatenate([four_pc_stack[0], four_pc_stack[1], four_pc_stack[2], four_pc_stack[3]], axis=0)
        colors = np.concatenate([colors[0], colors[1], colors[2], colors[3]], axis=0)
        view_npy_open3d(four_pc_stack, color=colors)

    def pick_for_handover(self):
        if self.last_handover_action.handover_state ==3: # rotate or drop
            print('No object to handover. Action: Go to home position')
            return self.last_handover_action
        else:
            available_actions_on_target = torch.count_nonzero(self.valid_actions_on_target_mask).item()
            for i in range(available_actions_on_target):
                flattened_action_index, probs, value = self.next_action(
                    sample_from_target_actions=True)
                unflatten_index = get_unflatten_index(flattened_action_index, ori_size=(self.voxel_pc.shape[0], 4))
                action_obj = Action(point_index=unflatten_index[0], action_index=unflatten_index[1], probs=probs,
                                    value=value)
                self.process_action(action_obj)
                if action_obj.is_executable:
                    action_obj.handover_state = 2
                    self.last_handover_action.handover_state = 2
                    return action_obj

            '''rotate if it is the first attempt otherwise release and go home'''
            if self.last_handover_action.handover_state == 0:
                print('No feasible access to handover. Action: rotate')
                self.last_handover_action.handover_state = 1
            else:
                print('No feasible access to handover. Action: release and go home')
                self.last_handover_action.handover_state = 3
            return self.last_handover_action

    def pick_action(self):
        # TODO: prioritize direct grasp to handover when the placement container is specified
        pr.title('pick action/s')
        first_action_obj=Action()
        second_action_obj=Action()

        if self.last_handover_action is not None:
            '''complete the handover steps'''
            first_action_obj = self.pick_for_handover()
        else:
            self.first_action_mask=None
            self.tmp_occupation_mask=torch.ones_like(self.valid_actions_mask)

            '''first action'''
            total_available_actions=torch.count_nonzero(self.valid_actions_mask).item()
            available_actions_on_target=torch.count_nonzero(self.valid_actions_on_target_mask).item()
            visible_target=torch.any(self.target_object_mask==True)
            for i in range(total_available_actions):
                flattened_action_index, probs, value=self.next_action(sample_from_target_actions=i<available_actions_on_target)
                unflatten_index = get_unflatten_index(flattened_action_index, ori_size=(self.voxel_pc.shape[0],4))
                action_obj=Action(point_index=unflatten_index[0],action_index=unflatten_index[1], probs=probs, value=value)
                self.process_action(action_obj)
                if action_obj.is_executable:
                    action_obj.target_point = self.voxel_pc[action_obj.point_index]
                    if i<available_actions_on_target:first_action_obj.policy_index=1
                    elif visible_target:
                        first_action_obj.policy_index=0
                    else: first_action_obj.policy_index=2
                    if action_obj.is_grasp:
                        self.tmp_occupation_mask,second_arm_extremes=self.mask_arm_occupancy(action_obj)
                        if self.handover_mask is not None and self.handover_mask[action_obj.point_index, action_obj.arm_index]:
                            action_obj.handover_state=0
                            self.last_handover_action=action_obj
                    elif first_action_obj.is_shift:
                        first_action_obj.contact_with_container=1 if self.obj

                    first_action_obj=action_obj
                    break

            if not first_action_obj.is_executable: exit('No executable action found ...')
            first_action_obj.target_point=self.voxel_pc[first_action_obj.point_index]
            first_action_obj.print()

            if first_action_obj.is_shift or single_arm_operation_mode or first_action_obj.policy_index==0 or first_action_obj.handover_state is not None:
                return first_action_obj, second_action_obj

            '''second action'''
            available_actions_on_target = torch.count_nonzero(self.valid_actions_on_target_mask & self.tmp_occupation_mask).item()
            for i in range(available_actions_on_target):
                flattened_action_index, probs, value = self.next_action(sample_from_target_actions=True)
                unflatten_index = get_unflatten_index(flattened_action_index, ori_size=(self.voxel_pc.shape[0], 4))
                action_obj = Action(point_index=unflatten_index[0],action_index=unflatten_index[1], probs=probs, value=value)
                self.process_action(action_obj)
                if action_obj.is_executable:
                    second_action_obj=action_obj
                    if i < available_actions_on_target: second_action_obj.policy_index = 1
                    elif visible_target: first_action_obj.policy_index=0
                    else: first_action_obj.policy_index=2
                    break

            if second_action_obj.is_executable:
                second_action_obj.target_point=self.voxel_pc[second_action_obj.point_index]
                first_action_obj.is_synchronous=True
                second_action_obj.is_synchronous=True
                second_action_obj.print()
                # print('second arm extreme from dense extrems',second_arm_extremes[second_action_obj.point_index])

        return first_action_obj,second_action_obj

    def wait_robot_feedback(self,first_action_obj,second_action_obj):
        # wait until grasp or suction finished
        robot_feedback_ = 'Wait'
        wait = wi('Waiting for robot feedback')
        print()
        counter=0
        while robot_feedback_ == 'Wait' or robot_feedback_.strip()=='':
            if counter == 0:
                if self.buffer_modify_alert:
                    '''reduce buffer size'''
                    self.buffer.pop()
                    print('buffer pop')
            elif counter == 1:
                if self.buffer_modify_alert:
                    '''dump the buffer as pickl'''
                    # save_pickle(buffer_file,self.buffer)
                    online_data2.save_pickle(buffer_file, self.buffer)
                    self.buffer_modify_alert=False
                    print('save buffer')
            elif counter == 2:
                if self.data_tracker_modify_alert:
                    '''save data tracker'''
                    self.data_tracker.save()
                    self.data_tracker_modify_alert=False
                    print('save data tracker')
            elif counter == 3:
                policy_net = ModelWrapper(model=PolicyNet(), module_key=policy_module_key)
                new_time_stamp=policy_net.model_time_stamp()
                if new_time_stamp != self.model_time_stamp:
                    policy_net.ini_model(train=False)
                    self.model_time_stamp=new_time_stamp
                    print('Update policy')

            elif counter==4:
                self.gripper_usage_rate.save()
                self.suction_usage_rate.save()
                self.double_grasp_rate.save()
                self.gripper_grasp_success_rate.save()
                self.suction_grasp_success_rate.save()
                self.shift_rate.save()
                self.planning_success_rate.save()

            else:
                wait.step(0.5)
            robot_feedback_ = read_robot_feedback()
            counter+=1
        else:
            wait.end()
            print('Robot returned msg: ' + robot_feedback_)
        first_action_obj.robot_feedback = robot_feedback_
        second_action_obj.robot_feedback = robot_feedback_
        return first_action_obj,second_action_obj

    def swap_actions(self,first_action_obj:Action,second_action_obj:Action):
        return second_action_obj,first_action_obj
    def completion_check_for_dual_grasp(self,first_action_obj:Action,second_action_obj:Action):
        if first_action_obj.robot_feedback=='Failed_l': # left is the suction arm
            if first_action_obj.use_suction_arm:
                first_action_obj,second_action_obj,self.swap_actions(first_action_obj,second_action_obj)
            second_action_obj.clear()

            return True, first_action_obj, second_action_obj
        elif first_action_obj.robot_feedback=='Failed_r': # right is the gripper arm
            if first_action_obj.use_gripper_arm:
                first_action_obj,second_action_obj,self.swap_actions(first_action_obj,second_action_obj)
            second_action_obj.clear()

            return True, first_action_obj, second_action_obj
        else:
            return  False, first_action_obj,second_action_obj


    def deploy_action_metrics(self,first_action_obj:Action,second_action_obj:Action):
        pr.print('Deploy action commands')
        if not first_action_obj.is_valid: return first_action_obj, second_action_obj
        deploy_action(first_action_obj)
        if second_action_obj.is_valid: deploy_action(second_action_obj)

    def run_robot(self,  first_action_obj:Action,second_action_obj:Action):
        pr.title('execute action')


        if not simulation_mode:
            pr.print('Run robot')
            set_wait_flag()
            if second_action_obj.is_valid and (first_action_obj.is_grasp and second_action_obj.is_grasp):
                '''dual grasp'''
                subprocess.run(["bash", './bash/pass_command.sh', "2"])
            elif first_action_obj.is_grasp and first_action_obj.handover_state is None:
                '''single grasp'''
                if first_action_obj.use_gripper_arm:
                    subprocess.run(["bash", './bash/pass_command.sh', "0"])
                else:
                    '''suction'''
                    subprocess.run(["bash", './bash/pass_command.sh', "1"])
            elif first_action_obj.is_shift:
                '''shift'''
                if first_action_obj.use_gripper_arm:
                    subprocess.run(["bash", './bash/pass_command.sh', "3"])
                else:
                    '''suction'''
                    subprocess.run(["bash", './bash/pass_command.sh', "4"])
            elif first_action_obj.handover_state is not None:

                if first_action_obj.use_gripper_arm:
                    '''gripper'''
                    if first_action_obj.handover_state==0:
                        '''hand'''
                        subprocess.run(["bash", './bash/pass_command.sh', "8"])
                    elif first_action_obj.handover_state==1:
                        '''rotate'''
                        subprocess.run(["bash", './bash/pass_command.sh', "12"])
                    elif first_action_obj.handover_state==2:
                        '''grasp'''
                        subprocess.run(["bash", './bash/pass_command.sh', "10"])
                    elif first_action_obj.handover_state==3:
                        '''drop'''
                        subprocess.run(["bash", './bash/pass_command.sh', "14"])

                else:
                    '''suction'''
                    if first_action_obj.handover_state == 0:
                        '''hand'''
                        subprocess.run(["bash", './bash/pass_command.sh', "9"])
                    elif first_action_obj.handover_state == 1:
                        '''rotate'''
                        subprocess.run(["bash", './bash/pass_command.sh', "13"])
                    elif first_action_obj.handover_state == 2:
                        '''grasp'''
                        subprocess.run(["bash", './bash/pass_command.sh', "11"])
                    elif first_action_obj.handover_state == 3:
                        '''drop'''
                        subprocess.run(["bash", './bash/pass_command.sh', "15"])

        return first_action_obj,second_action_obj

    def update_gripper_grasp_success_rate(self,gripper_action):
        if gripper_action.grasp_result is None:
            print(Fore.LIGHTCYAN_EX, 'Unable to detect the grasp result for the gripper', Fore.RESET)
        elif gripper_action.grasp_result:
            print(Fore.GREEN, 'A new object is detected at the gripper side of the bin', Fore.RESET)
            self.gripper_grasp_success_rate.update(1)
        else:
            print(Fore.YELLOW, 'No object is detected at to the gripper side of the bin', Fore.RESET)
            self.gripper_grasp_success_rate.update(0)

    def update_suction_grasp_success_rate(self,suction_action):
        if suction_action.grasp_result is None:
            print(Fore.LIGHTCYAN_EX, 'Unable to detect the grasp result for the suction', Fore.RESET)
        elif suction_action.grasp_result:
            print(Fore.GREEN, 'A new object is detected at the suction side of the bin', Fore.RESET)
            self.suction_grasp_success_rate.update(1)
        else:
            print(Fore.YELLOW, 'No object is detected at to the suction side of the bin', Fore.RESET)
            self.suction_grasp_success_rate.update(0)

    def report_handover(self,first_action_obj,second_action_obj,img_grasp_pre, img_suction_pre,img_suction_after, img_grasp_after):
        if first_action_obj.robot_feedback == 'Succeed':
            self.planning_success_rate.update(1)
            first_action_obj.executed = True

            '''report at the end of handover sequence'''
            if first_action_obj.handover_state in {2, 3}:
                if first_action_obj.use_gripper_arm:
                    '''handed to the gripper'''
                    first_action_obj.handover_result = check_image_similarity(img_grasp_pre, img_grasp_after)
                    gripper_action = first_action_obj
                    suction_action = second_action_obj
                else:
                    '''handed to the suction'''
                    first_action_obj.grasp_result = check_image_similarity(img_suction_pre, img_suction_after)
                    gripper_action = second_action_obj
                    suction_action = first_action_obj

                gripper_action, suction_action = save_grasp_sample(self.rgb, self.depth, self.mask_numpy,
                                                                   gripper_action, suction_action,
                                                                   self.run_sequence)

                '''update tracker'''
                if gripper_action.is_executable:
                    self.data_tracker.push(gripper_action)
                    self.data_tracker_modify_alert = True

                if suction_action.is_executable:
                    self.data_tracker.push(suction_action)
                    self.data_tracker_modify_alert = True

        else:
            self.planning_success_rate.update(0)
            first_action_obj.executed = False

    def report_result(self,first_action_obj,second_action_obj,img_grasp_pre, img_suction_pre,img_main_pre):

        img_suction_after, img_grasp_after, img_main_after = get_side_bins_images()

        if first_action_obj.policy_index in {1, 2}:
            self.run_sequence = 0

        if first_action_obj.handover_state is not None:
            '''process feedback for a handover action'''
            if report_for_handover:
                self.report_handover(first_action_obj, second_action_obj,img_grasp_pre, img_suction_pre,img_suction_after, img_grasp_after)
        else:
            '''save feedback to data pool'''
            if first_action_obj.robot_feedback == 'Succeed':
                if first_action_obj.is_shift:
                    if not report_for_shift: return
                    if enable_gripper_grasp==False or enable_suction_grasp==False: return # do not update the buffer when any arm grasp function is disabled
                    first_action_obj.shift_result=check_image_similarity(img_main_pre, img_main_after)
                    self.shift_rate.update(1)
                else:
                    if not report_for_grasp: return
                    self.shift_rate.update(0)

                self.planning_success_rate.update(1)

                first_action_obj.executed = True
                second_action_obj.executed = True

                if first_action_obj.use_gripper_arm:
                    gripper_action=first_action_obj
                    suction_action=second_action_obj
                else:
                    gripper_action = second_action_obj
                    suction_action = first_action_obj

                '''check changes in side bins'''
                if gripper_action.is_grasp:
                    gripper_action.grasp_result=check_image_similarity(img_grasp_pre, img_grasp_after)
                    self.update_gripper_grasp_success_rate(gripper_action)

                if suction_action.is_grasp:
                    suction_action.grasp_result=check_image_similarity(img_suction_pre, img_suction_after)
                    self.update_suction_grasp_success_rate(suction_action)

                '''save action instance'''
                # assert gripper_action.result is not None or suction_action.result is not None
                gripper_action,suction_action=save_grasp_sample(self.rgb, self.depth,self.mask_numpy, gripper_action, suction_action,self.run_sequence)

                '''update buffer and tracker'''
                if gripper_action.is_executable:
                    self.gripper_usage_rate.update(1)
                    if suction_action.is_executable:
                        self.double_grasp_rate.update(1)
                    else:
                        self.double_grasp_rate.update(0)

                    self.buffer.push(gripper_action)
                    self.data_tracker.push(gripper_action)
                    self.buffer_modify_alert=True
                    self.data_tracker_modify_alert=True
                else:
                    self.gripper_usage_rate.update(0)

                if suction_action.is_executable:
                    self.suction_usage_rate.update(1)

                    self.buffer.push(suction_action)
                    self.data_tracker.push(suction_action)
                    self.buffer_modify_alert=True
                    self.data_tracker_modify_alert=True
                else:
                    self.suction_usage_rate.update(0)

            else:
                self.planning_success_rate.update(0)

                first_action_obj.executed=False
                second_action_obj.executed=False

        self.run_sequence += 1

    def process_feedback(self,first_action_obj:Action,second_action_obj:Action, img_grasp_pre, img_suction_pre,img_main_pre):
        pr.title('Process feedback')
        new_state_available=False
        if first_action_obj.robot_feedback == 'Succeed' or first_action_obj.robot_feedback == 'reset':
            trigger_new_perception()
            new_state_available=True

            '''handover post processing'''
            if first_action_obj.handover_state is not None:
                if first_action_obj.handover_state==0 or first_action_obj.handover_state==1 :
                    if first_action_obj.robot_feedback == 'Succeed':
                        first_action_obj.set_activated_arm_position(at_home=False)
                else:
                    first_action_obj.set_activated_arm_position(at_home=True)

                self.last_handover_action==first_action_obj
        elif first_action_obj.robot_feedback == 'Failed':
            if first_action_obj.is_grasp and second_action_obj.is_executable is None:
                print('report failed path planning')
                if first_action_obj.use_gripper_arm:report_failed_gripper_path_plan(first_action_obj.transformation)
                else: report_failed_suction_path_plan(first_action_obj.transformation)

        if report_result:
            self.report_result(first_action_obj,second_action_obj,img_grasp_pre, img_suction_pre,img_main_pre)

        if self.last_handover_action is not None :
            if (self.last_handover_action.handover_state in {2,3}) \
                    | (first_action_obj.robot_feedback != 'Succeed'):
                '''End of handover action sequence'''
                self.last_handover_action=None
                subprocess.run(["bash", './bash/pass_command.sh', "5"])

        return new_state_available

'''conventions'''
# normal is a vector emerge out of the surface
# approach direction is a vector pointing to the surface
# approach = -1 * normal
# The first three parameters of the gripper pose are approach[0] and approach [1] and -1* approach[2]
# the suction sampler outputs the normal direction
# T_0 refers to a gripper head transformation matrix with zero penetration while T_d embeds the distance term
# for any sequence we will always give the gripper the first index followed by the suction, e.g. if gripper grasp score locate at the (i) channel then the suction is located at (i+1) channel
# if the action is shift, it is always saved in the first action object and no action is assigned to the second action object
# executing both arms at the same time is only allowed when both actions are grasp
# a single run may include one action or two actions (moving both arms)
# After execution, the robot rises three flags:
    # succeed: the plan has been executed completely
    # failed: Unable to execute part or full of the path
    # reset: path plan is found but execution terminated due to an error
