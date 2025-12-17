import subprocess
import smbclient.path
from analytical_suction_sampler import estimate_suction_direction
from explore_arms_scope import report_failed_gripper_path_plan, report_failed_suction_path_plan
from lib.bbox import rotation_matrix_from_normal_target
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from colorama import Fore
import open3d as o3d
from Configurations.dynamic_config import get_int
from Online_data_audit.data_tracker2 import DataTracker2
from action import Action
from lib.IO_utils import  save_to_server, load_pickle_from_server, save_image_to_server
from lib.grasp_utils import shift_a_distance
from lib.models_utils import number_of_parameters
from lib.report_utils import wait_indicator as wi
from Configurations.ENV_boundaries import bin_center, dist_allowance, knee_ref_elevation
from Configurations.config import distance_scope, gripper_width_during_shift, ip_address
from Configurations.run_config import simulation_mode, \
    suction_factor, gripper_factor, report_result, \
    enhance_gripper_firmness, single_arm_operation_mode, \
    enable_gripper_grasp, enable_gripper_shift, enable_suction_grasp, enable_suction_shift, \
    activate_segmentation_queries, activate_handover, only_handover, \
    report_for_handover, report_for_shift, report_for_grasp, \
    zero_out_distance_when_collision, handover_quality_bias, suction_grasp_bias, gripper_grasp_bias, \
    sample_action_with_argmax, quality_exponent, activate_shift_from_beneath_objects, bin_collision_threshold, \
    objects_collision_threshold, suction_reachablity_threshold, gripper_reachablity_threshold, \
    activate_gripper_quality_mask, activate_suction_quality_mask, use_analytical_normal_estimation, \
    activate_grasp_quality_check, activate_preferred_placement_side, minimum_penetration_distance_ratio
from Online_data_audit.process_feedback import save_grasp_sample, grasp_data_counter_key
from check_points.check_point_conventions import GANWrapper, ModelWrapper, ActorCriticWrapper
from lib.ROS_communication import deploy_action, read_robot_feedback, set_wait_flag
from lib.collision_unit import grasp_collision_detection
from lib.custom_print import my_print
from lib.depth_map import transform_to_camera_frame_torch, depth_to_point_clouds
from lib.image_utils import check_image_similarity, view_image
from lib.pc_utils import numpy_to_o3d
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from models.Grasp_handover_policy_net import GraspHandoverPolicyNet, grasp_handover_policy_module_key
from models.action_net import ActionNet, action_module_with_GAN_key
from models.scope_net import scope_net_vanilla, gripper_scope_module_key, suction_scope_module_key
from models.shift_policy_net import shift_policy_module_key, ShiftPolicyCriticNet, ShiftPolicyActorNet
from pose_object import vectors_to_ratio_metrics, pose_7_to_transformation
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

def angle_between_vectors(v1, v2, eps=1e-8):
    # v1, v2: (..., 3)
    v1_n = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
    v2_n = v2 / (v2.norm(dim=-1, keepdim=True) + eps)
    dot = torch.sum(v1_n * v2_n, dim=-1).clamp(-1.0, 1.0)
    return torch.acos(dot)

def vector_product(v1, v2):
    return torch.cross(v1, v2, dim=-1)

def rotation_matrix(angle, axis):
    """
    angle: scalar tensor
    axis: (3,) tensor
    returns: (4,4) rotation matrix
    """
    axis = axis / (axis.norm() + 1e-8)
    x, y, z = axis
    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1 - c

    R = torch.tensor([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,     y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]
    ], dtype=angle.dtype, device=angle.device)

    T = torch.eye(4, dtype=angle.dtype, device=angle.device)
    T[:3, :3] = R
    return T

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
    directions=F.normalize(directions,p=2,dim=-1,eps=1e-8)
    return end_points,directions

def masked_color(voxel_pc, score, pivot=0.5):
    mask_=score.cpu().numpy()>pivot if torch.is_tensor(score) else score>pivot
    colors = np.zeros_like(voxel_pc)
    colors[mask_]+= [0.5, 0.9, 0.5]
    colors[~mask_] += [0.52, 0.8, 0.92]
    return colors

def view_mask(voxel_pc, score, pivot=0.5):
    means_=voxel_pc.mean(axis=0)
    voxel_pc_t=(voxel_pc-means_).cpu().numpy()
    # voxel_pc_t[:,1]*=-1
    colors=masked_color(voxel_pc_t, score, pivot=pivot)
    view_npy_open3d(voxel_pc_t, color=colors,view_coordinate=False)

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
        self.grasp_handover_policy_net=None
        self.shift_policy_actor=None
        self.shift_policy_critic=None
        self.suction_arm_reachability_net = None
        self.gripper_arm_reachability_net = None

        self.priority_id=None

        self.shift_model_time_stamp=None
        self.grasp_handover_model_time_stamp=None

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

        self.gripper_usage_rate=MovingRate('gripper_usage',decay_rate=0.01)
        self.suction_usage_rate=MovingRate('suction_usage',decay_rate=0.01)
        self.double_grasp_rate=MovingRate('double_grasp',decay_rate=0.01)
        self.gripper_grasp_success_rate=MovingRate('gripper_grasp_success',decay_rate=0.01)
        self.suction_grasp_success_rate=MovingRate('suction_grasp_success',decay_rate=0.01)
        self.shift_rate=MovingRate('shift',decay_rate=0.01)
        self.planning_success_rate=MovingRate('planning_success',decay_rate=0.01)

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

        self.forward_counter=-1
        # self.actions_sequence=[1,0]
        self.actions_sequence=[]

        self.remove_seg_file()

    def rollback(self):
        self.gripper_collision_mask=None
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
        self.objects_mask=None

        self.shift_directions=None

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
            # view_image(self.rgb)
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

                    self.remove_seg_file()
                    if segmentation_mask.shape==(1,):
                        print(Fore.RED,'The queried object is not found!',Fore.RESET)
                        self.seg_mask=torch.zeros_like(self.depth,dtype=torch.bool)
                    else:
                        self.seg_mask=(torch.from_numpy(segmentation_mask)[None,None,...]>0.5).cuda()

                    break
                else:
                    wait.step(0.5)

    def ini_shift_policy(self):
        shift_model_wrapper = ActorCriticWrapper(module_key=shift_policy_module_key,actor=ShiftPolicyActorNet,critic=ShiftPolicyCriticNet)
        new_time_stamp = shift_model_wrapper.model_time_stamp()
        if self.shift_model_time_stamp is not None or new_time_stamp != self.shift_model_time_stamp:
            shift_model_wrapper.ini_models(train=False)
            self.shift_policy_actor=shift_model_wrapper.actor
            self.shift_policy_critic=shift_model_wrapper.critic
            self.shift_model_time_stamp=shift_model_wrapper.model_time_stamp()

    def ini_grasp_handover_policy(self):
        grasp_handover_model_wrapper = ModelWrapper(model=GraspHandoverPolicyNet(),
                                                    module_key=grasp_handover_policy_module_key)
        new_time_stamp = grasp_handover_model_wrapper.model_time_stamp()
        if self.grasp_handover_model_time_stamp is not None or new_time_stamp != self.grasp_handover_model_time_stamp:
            grasp_handover_model_wrapper.ini_model(train=False)
            self.grasp_handover_policy_net = grasp_handover_model_wrapper.model
            self.grasp_handover_model_time_stamp = grasp_handover_model_wrapper.model_time_stamp()

    def initialize_check_points(self):
        pi = progress_indicator('Loading check points  ', max_limit=6)

        pi.step(1)

        actions_net = GANWrapper(action_module_with_GAN_key, ActionNet)
        actions_net.ini_generator(train=False)

        self.action_net = actions_net.generator

        pi.step(2)
        self.ini_shift_policy()

        pi.step(3)
        self.ini_grasp_handover_policy()

        pi.step(4)

        gripper_scope = ModelWrapper(model=scope_net_vanilla(in_size=6), module_key=gripper_scope_module_key)
        gripper_scope.ini_model(train=False)
        self.gripper_arm_reachability_net = gripper_scope.model

        pi.step(5)

        suction_scope = ModelWrapper(model=scope_net_vanilla(in_size=6), module_key=suction_scope_module_key)
        suction_scope.ini_model(train=False)
        self.suction_arm_reachability_net=suction_scope.model

        pi.step(6)

        pi.end()

        self.models_report()

    def models_report(self):
        action_model_size=number_of_parameters(self.action_net)
        grasp_handover_policy_model_size=number_of_parameters(self.grasp_handover_policy_net)
        shift_policy_model_size=number_of_parameters(self.shift_policy_actor)

        # suction_arm_reachability_model_size=number_of_parameters(self.suction_arm_reachability_net)
        print(f'Models initiated:')
        print(f'Number of parameters:')
        pr.step_f(f'action net : {action_model_size}')
        pr.print(f'Shift policy net : {shift_policy_model_size}*2')
        pr.print(f'grasp_handover policy net : {grasp_handover_policy_model_size}')

        pr.step_b()


    def get_suction_grasp_reachability(self,positions,normals):
        approach=-normals.clone()
        suction_scope = self.suction_arm_reachability_net(torch.cat([positions, approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        return suction_scope

    def get_suction_shift_reachability(self,positions):
        suction_scope_a = self.suction_arm_reachability_net(torch.cat([positions, self.suction_approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        if self.shift_end_points is None:
            self.shift_end_points,self.shift_directions = get_shift_end_points(positions)
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
        self.shift_end_points,self.shift_directions = get_shift_end_points(positions)
        gripper_scope_b=self.gripper_arm_reachability_net(torch.cat([self.shift_end_points, self.gripper_approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        result = torch.stack([gripper_scope_a, gripper_scope_b], dim=-1)
        result, _ = torch.min(result, dim=-1)
        return result

    def next_action(self,sample_from_target_actions=False,eps=0.001):
        selected_policy = self.seize_policy if sample_from_target_actions else self.clear_policy
        selected_policy=torch.clip(selected_policy,eps)
        mask_=self.valid_actions_on_target_mask if sample_from_target_actions else self.valid_actions_mask
        if self.tmp_occupation_mask is not None: mask_=mask_ & self.tmp_occupation_mask
        # print('test',selected_policy[mask_].shape,sample_from_target_actions)
        assert torch.any(mask_)
        if sample_action_with_argmax:
            flattened_action_index=(selected_policy*mask_).argmax()
        else:
            dist=MaskedCategorical(probs=selected_policy,mask=mask_)
            flattened_action_index= dist.sample()

        if sample_from_target_actions:
            probs=(selected_policy[flattened_action_index.item()]).item()

            value=None

            flattened_action_index = torch.squeeze(flattened_action_index).detach().cpu().item()

            self.valid_actions_on_target_mask[flattened_action_index] = False
        else:
            probs = (dist.log_prob(flattened_action_index)).item()
            flattened_action_index = torch.squeeze(flattened_action_index).item()
            value = self.q_value[flattened_action_index]
            value = torch.squeeze(value).item()
        if self.last_handover_action is None:
            self.valid_actions_mask[flattened_action_index]=False
        return flattened_action_index, probs, value

    def inputs(self,depth,rgb,args):
        self.depth=torch.from_numpy(depth)[None, None, ...].to('cuda').float()
        self.rgb=rgb
        self.args=args

    def view_mask_as_2dimage(self):
        view_image(self.seg_mask[0,0].cpu().numpy().astype(np.float64))
        # view_image(self.mask_numpy.astype(np.float64))

    def background_manual_correction(self,background_class):
        min_elevation=self.voxel_pc[background_class<0.5,-1].min().item()

        floor_correction_mask=(self.voxel_pc[:,-1]<max(min_elevation+0.005,0.042))
        ceil_mask=self.voxel_pc[:,-1]>0.2
        background_class[floor_correction_mask|ceil_mask]=1.0
        x_correction_mask=(self.voxel_pc[:,0]<0.3) | (self.voxel_pc[:,0]>0.6)
        y_correction_mask=(self.voxel_pc[:,1]<-0.20) | (self.voxel_pc[:,1]>0.22)

        background_class[x_correction_mask|y_correction_mask]=1.0
        # background_class[y_correction_mask]=1.0

        return background_class

    def gripper_arm_mask_during_handover(self):
        elevation=0.3
        cylindrical_mask=torch.linalg.norm(self.voxel_pc[:,[0,2]]-torch.tensor([0.45,elevation-0.03],device=self.voxel_pc.device)[None],dim=-1)<0.03
        arm_rectangular_mask=(torch.abs( self.voxel_pc[:,0]-0.45)<0.08) & (torch.abs(self.voxel_pc[:,2]-elevation)<0.08) & (self.voxel_pc[:, 1] < -0.09)
        cylindrical_mask=cylindrical_mask & (self.voxel_pc[:, 1] < -0.03)
        object_rectangular_mask=(self.voxel_pc[:, 2] < (elevation+0.05)) &(self.voxel_pc[:, 2] > (elevation-0.15))
        object_rectangular_mask=object_rectangular_mask & (self.voxel_pc[:, 1] > -0.03) & (self.voxel_pc[:,1]<0.12)
        object_rectangular_mask=object_rectangular_mask & (self.voxel_pc[:, 0] > 0.3) & (self.voxel_pc[:,0]<0.6)

        handed_object_mask=  object_rectangular_mask\
                         & (~cylindrical_mask) & (~arm_rectangular_mask)
        # view_mask(self.voxel_pc, object_rectangular_mask)
        # view_mask(self.voxel_pc, ~cylindrical_mask)
        # view_mask(self.voxel_pc, ~arm_rectangular_mask)
        # view_mask(self.voxel_pc,handed_object_mask)


        '''detect object'''
        no_object=self.voxel_pc[handed_object_mask].shape[0]<1
        # print(self.voxel_pc[handed_object_mask].shape[0])
        # view_mask(self.voxel_pc, handed_object_mask)

        return handed_object_mask,no_object

    def suction_arm_mask_during_handover(self):
        # suction_ref_point=np.array([0.45,0.01,0.24])
        elevation = 0.3
        # cylindrical_mask=np.linalg.norm(self.voxel_pc[:,[0,2]]-np.array([0.45,elevation-0.05])[np.newaxis],axis=-1)<0.015

        cylindrical_mask=torch.linalg.norm(self.voxel_pc[:,[0,2]]-torch.tensor([0.45,elevation-0.05],device=self.voxel_pc.device)[None],dim=-1)<0.015
        handed_object_mask=(self.voxel_pc[:,1]<0.0) & (self.voxel_pc[:,2]>(elevation-0.1)) & (~cylindrical_mask) \
                          & (self.voxel_pc[:,1]>-0.2)

        # view_mask(self.voxel_pc, ~cylindrical_mask)
        # view_mask(self.voxel_pc, (self.voxel_pc[:,1]<0.0))
        # view_mask(self.voxel_pc, (self.voxel_pc[:,2]>(elevation-0.1)))
        # view_mask(self.voxel_pc, (self.voxel_pc[:,1]>-0.2))
        # view_mask(self.voxel_pc, handed_object_mask)

        '''detect object'''
        no_object=self.voxel_pc[handed_object_mask].shape[0]<1
        # view_mask(self.voxel_pc, handed_object_mask)
        # view_npy_open3d(self.voxel_pc[(handed_object_mask)])

        return   handed_object_mask,no_object

    def models_inference(self):
        pr.title('Inference')

        '''depth to point clouds'''
        self.voxel_pc, mask = depth_to_point_clouds(self.depth[0,0], camera)
        self.voxel_pc = transform_to_camera_frame_torch(self.voxel_pc, reverse=True)

        if self.last_handover_action is not None:self.depth=self.alter_depth_ref(self.depth)

        rgb_torch = torch.from_numpy(self.rgb).permute(2, 0, 1)[None, ...].to('cuda').float()

        # if self.last_handover_action is not None and self.last_handover_action.suction_at_home_position==False:
        #     approach=torch.zeros_like(self.depth).repeat(1,3,1,1)
        #     approach[:,2,...]+=0.966
        #     approach[:,1,...]+=0.2588
        # else:
        #     approach=None

        '''action net output'''
        gripper_pose, suction_direction, griper_collision_classifier, suction_seal_classifier, shift_appealing \
            , background_class = self.action_net(self.depth.clone(),mask[None,None])

        '''pixel-wise to point-wise sampling'''
        if use_analytical_normal_estimation:
            self.normals = estimate_suction_direction(self.voxel_pc.cpu().numpy(),
                                                view=False)
            self.normals=torch.from_numpy(self.normals).to(suction_direction.device).float()
        else:
            self.normals = suction_direction.squeeze().permute(1, 2, 0)[mask]  # [N,3]

        self.gripper_poses_7 = gripper_pose.squeeze().permute(1, 2, 0)[mask]
        self.gripper_poses_7[:,5:]=torch.clip(self.gripper_poses_7[:,5:],minimum_penetration_distance_ratio,1)

        '''target mask'''

        if self.last_handover_action is  None and activate_segmentation_queries and self.seg_mask.shape == background_class.shape:
            self.target_object_mask = (background_class <= 0.5) & self.seg_mask
            # view_mask(self.voxel_pc, self.seg_mask[0, 0][mask])
            # view_image((background_class <= 0.5)[0,0].cpu().numpy().astype(np.float64))

            # view_image(self.seg_mask[0,0].cpu().numpy().astype(np.float64))
            # print(self.target_object_mask.shape)
            # view_mask(self.voxel_pc, self.target_object_mask[0, 0][mask])
        else:
            self.target_object_mask = (background_class <= 0.5)
        self.target_object_mask[0, 0][~mask] *= False
        self.mask_numpy = self.target_object_mask.squeeze().cpu().numpy()

        '''shift mask'''
        shift_mask_ = (shift_appealing > 0.5)
        shift_mask_[0, 0][~mask] *= False

        '''policy net output'''
        clear_policy_props= self.shift_policy_actor(rgb_torch, self.depth.clone(), self.target_object_mask.float(),shift_mask=shift_mask_)
        q_value= self.shift_policy_critic(rgb_torch, self.depth.clone(), self.target_object_mask.float())
        griper_grasp_score, suction_grasp_score, handover_scores = \
            self.grasp_handover_policy_net(rgb_torch, self.depth.clone(), gripper_pose, suction_direction)



        '''alter policy dim'''
        q_value=q_value.repeat(1,4,1,1)
        q_value[:,0:2,...]*=q_value.min()
        clear_policy_props=clear_policy_props.repeat(1,4,1,1)
        clear_policy_props[:,0:2,...]*=0.



        return griper_grasp_score, suction_grasp_score, \
        q_value, clear_policy_props, handover_scores,\
            griper_collision_classifier,suction_seal_classifier,\
            background_class,shift_appealing,mask

    def grasp_reachablity(self):
        '''grasp reachability'''
        suction_grasp_scope = self.get_suction_grasp_reachability(self.voxel_pc, self.normals)
        gripper_grasp_scope = self.get_gripper_grasp_reachability(self.voxel_pc)

        return gripper_grasp_scope,suction_grasp_scope
    def shift_reachablity(self):
        '''shift reachability'''
        gripper_shift_scope = self.get_gripper_shift_reachability(self.voxel_pc)
        suction_shift_scope = self.get_suction_shift_reachability(self.voxel_pc)

        '''grasp/shift with the arm of best reachability'''
        gripper_shift_scope[suction_shift_scope > gripper_shift_scope] *= 0.
        suction_shift_scope[gripper_shift_scope > suction_shift_scope] *= 0.

        return gripper_shift_scope, suction_shift_scope

    def main_dense_processing(self,shift_appealing_mask,gripper_shift_reachablity_mask,suction_shift_reachablity_mask,
                                griper_grasp_score,suction_grasp_score,handover_scores):
        '''shift actions'''
        self.gripper_shift_mask = (shift_appealing_mask * gripper_shift_reachablity_mask * enable_gripper_shift)
        self.suction_shift_mask = (shift_appealing_mask * suction_shift_reachablity_mask * enable_suction_shift)
        # view_mask(self.voxel_pc,(self.suction_shift_mask>0.5))

        if not activate_shift_from_beneath_objects:
            self.gripper_shift_mask[~self.objects_mask]=False
            self.suction_shift_mask[~self.objects_mask]=False

        # view_mask(self.voxel_pc,(self.gripper_shift_mask>0.5))
        # view_mask(self.voxel_pc,(self.suction_shift_mask>0.5))
        # view_mask(voxel_pc_,self.suction_shift_mask)


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

        # view_mask(self.voxel_pc, self.suction_grasp_mask)
        # view_mask(self.voxel_pc, self.target_object_mask)

        if activate_preferred_placement_side and self.args.placement_bin=='g':
            self.valid_actions_on_target_mask[:, 1].fill_( False)
        else:
            self.valid_actions_on_target_mask[:, 1].masked_fill_(suction_actions_mask_on_target, True)

        if activate_preferred_placement_side and self.args.placement_bin=='s':
            self.valid_actions_on_target_mask[:, 0].fill_( False)
        else:
            self.valid_actions_on_target_mask[:, 0].masked_fill_(gripper_actions_mask_on_target, True)

        # view_mask(self.voxel_pc,gripper_actions_mask_on_target)

        '''handover processing'''
        if activate_handover: self.handover_processing(handover_scores)
        if activate_preferred_placement_side:
            if self.args.placement_bin == 'g':
                self.priority_id=0
            elif self.args.placement_bin == 's':
                self.priority_id=1

        self.valid_actions_mask = self.valid_actions_mask.reshape(-1)

        '''to numpy'''
        self.normals=self.normals#.cpu().numpy()

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
        self.suction_grasp_mask = self.suction_grasp_mask * handed_object_mask
        self.gripper_grasp_mask *= False

        '''initialize the seize policy'''
        self.seize_policy = torch.zeros_like(self.clear_policy)
        self.seize_policy[:, 1] = suction_grasp_score

        '''initialize valid actions mask on target'''
        self.valid_actions_on_target_mask = torch.zeros_like(self.q_value, dtype=torch.bool)
        suction_actions_mask_on_target = self.suction_grasp_mask  # & self.target_object_mask
        self.valid_actions_on_target_mask[:, 1].masked_fill_(suction_actions_mask_on_target, True)

        '''to numpy'''
        self.normals=self.normals#.cpu().numpy()

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
        self.gripper_grasp_mask = self.gripper_grasp_mask * handed_object_mask
        self.suction_grasp_mask *= False

        '''initialize the seize policy'''
        self.seize_policy = torch.zeros_like(self.clear_policy)
        self.seize_policy[:, 0] = griper_grasp_score

        '''initialize valid actions mask on target'''
        self.valid_actions_on_target_mask = torch.zeros_like(self.q_value, dtype=torch.bool)
        gripper_actions_mask_on_target = self.gripper_grasp_mask  # & self.target_object_mask
        self.valid_actions_on_target_mask[:, 0].masked_fill_(gripper_actions_mask_on_target, True)

        '''to numpy'''
        self.normals=self.normals#.cpu().numpy()

        '''flatten'''
        self.seize_policy=self.seize_policy.reshape(-1)
        self.valid_actions_on_target_mask = self.valid_actions_on_target_mask.reshape(-1)
    def process_grasp_masks(self,gripper_grasp_reachablity_mask,gripper_quality_mask,suction_grasp_reachablity_mask
                            ,seal_quality_mask,suction_quality_mask):
        self.gripper_grasp_mask = (self.objects_mask * gripper_grasp_reachablity_mask
                                   * self.gripper_collision_mask)

        if activate_gripper_quality_mask: self.gripper_grasp_mask *= gripper_quality_mask

        self.suction_grasp_mask = (self.objects_mask * suction_grasp_reachablity_mask
                                   * seal_quality_mask)

        # view_mask(self.voxel_pc, self.gripper_grasp_mask)
        # view_mask(self.voxel_pc, self.objects_mask)
        # view_mask(self.voxel_pc, self.gripper_collision_mask)
        # view_mask(self.voxel_pc, gripper_grasp_reachablity_mask)
        # view_mask(self.voxel_pc, self.suction_grasp_mask)
        # view_mask(self.voxel_pc, self.objects_mask)
        # view_mask(self.voxel_pc, suction_grasp_reachablity_mask)
        # view_mask(self.voxel_pc, seal_quality_mask)
        # view_mask(self.voxel_pc, suction_quality_mask)

        if activate_suction_quality_mask: self.suction_grasp_mask *= suction_quality_mask

    def alter_depth_ref(self,depth,objects_mask=None):
        # pc, mask = self.get_point_clouds(depth)

        if objects_mask is None:
            shift_entities_mask = (depth > 0.0001)
        else:
            shift_entities_mask = objects_mask & (depth > 0.0001)

        new_depth = depth.clone().detach()
        new_depth[shift_entities_mask] += 0.14  * camera.scale

        # pcs[k], masks[k] = depth_to_point_clouds(depth[k, 0].cpu().numpy(), camera)
        # pcs[k] = transform_to_camera_frame(pcs[k], reverse=True)

        return new_depth#,pc, mask

    def  sample_masked_actions(self):
        self.forward_counter+=1
        '''Action and Policy network inference'''
        griper_grasp_score, suction_grasp_score, \
        q_value, clear_policy, handover_scores, \
        griper_collision_classifier, suction_seal_classifier, \
        background_class, shift_appealing,mask=\
        self.models_inference()

        '''Reachability inference'''
        gripper_grasp_scope,suction_grasp_scope=self.grasp_reachablity()
        gripper_shift_scope, suction_shift_scope=self.shift_reachablity()

        '''clip quality scores'''
        griper_grasp_score=torch.clamp(griper_grasp_score,0.,1.)
        suction_grasp_score=torch.clamp(suction_grasp_score,0.,1.)
        handover_scores=torch.clamp(handover_scores,0.,1.)

        '''add exponent term'''
        if quality_exponent!=1.:
            griper_grasp_score=griper_grasp_score ** quality_exponent
            suction_grasp_score = suction_grasp_score ** quality_exponent
            handover_scores = handover_scores ** quality_exponent

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
        # view_mask(self.voxel_pc, self.target_object_mask )

        '''correct background mask'''
        # if self.last_handover_action is None:
        #     # view_mask(self.voxel_pc, background_class < 0.5)
        #     background_class=self.background_manual_correction(background_class)
        #     # view_image(self.target_object_mask[0,0].cpu().numpy().astype(np.float64))

        '''actions masks'''
        self.objects_mask=background_class<0.5
        # view_mask(self.voxel_pc, object_mask )
        gripper_grasp_reachablity_mask=gripper_grasp_scope>gripper_reachablity_threshold
        self.gripper_collision_mask=(griper_object_collision_classifier<objects_collision_threshold) & (griper_bin_collision_classifier<bin_collision_threshold)
        gripper_quality_mask=griper_grasp_score*gripper_factor>(0.5** quality_exponent)
        suction_grasp_reachablity_mask=suction_grasp_scope>suction_reachablity_threshold
        seal_quality_mask=suction_seal_classifier>0.5
        suction_quality_mask=suction_grasp_score*suction_factor>(0.5** quality_exponent)
        shift_appealing_mask=shift_appealing>0.5
        gripper_shift_reachablity_mask=gripper_shift_scope>gripper_reachablity_threshold
        suction_shift_reachablity_mask=suction_shift_scope>suction_reachablity_threshold

        # view_mask(self.voxel_pc, gripper_quality_mask)
        '''Robust score processing'''
        suction_grasp_score[~seal_quality_mask]=torch.clip(suction_grasp_score[~seal_quality_mask],max=0.01)
        robust_suction_grasp_score = suction_grasp_score  * (1 - griper_grasp_score ** 2)
        robust_griper_grasp_score=griper_grasp_score*(1-suction_grasp_score**2)#*dist
        suction_grasp_score=robust_suction_grasp_score
        griper_grasp_score=robust_griper_grasp_score

        if self.last_handover_action is not None :
            self.process_grasp_masks( gripper_grasp_reachablity_mask, gripper_quality_mask,
                                suction_grasp_reachablity_mask
                                , seal_quality_mask, suction_quality_mask)

            # view_mask(self.voxel_pc, (self.suction_grasp_mask > 0.5))

            if self.last_handover_action.gripper_at_home_position==False:
                print('Process handover from gripper to suction')
                # proceede with handover sequence
                self.handover_from_gripper_dense_processing(suction_grasp_score)
            else:
                print('Process handover from suction to gripper')
                # proceede with handover sequence
                self.handover_from_suction_dense_processing(griper_grasp_score)
        else:
            self.process_grasp_masks(gripper_grasp_reachablity_mask, gripper_quality_mask,
                                     suction_grasp_reachablity_mask
                                     , seal_quality_mask, suction_quality_mask)
            self.gripper_grasp_mask *= enable_gripper_grasp
            self.suction_grasp_mask *= enable_suction_grasp

            # view_mask(self.voxel_pc, self.objects_mask)
            # view_mask(self.voxel_pc, (griper_object_collision_classifier < objects_collision_threshold))
            # view_mask(self.voxel_pc, (griper_bin_collision_classifier<bin_collision_threshold))
            # view_mask(self.voxel_pc, gripper_grasp_reachablity_mask)
            # view_mask(self.voxel_pc, self.gripper_collision_mask)
            # view_mask(self.voxel_pc, gripper_quality_mask)
            # view_mask(self.voxel_pc, self.suction_grasp_mask)

            if len(self.actions_sequence)>self.forward_counter:
                print('A predetermined sequence of action is activated')
                # print(self.actions_sequence[self.forward_counter])
                # print(self.forward_counter)
                if self.actions_sequence[self.forward_counter]!=0:
                    # print('yes2')
                    self.gripper_grasp_mask*=False
                    self.suction_grasp_mask*=False

            '''Arms at home position'''
            # proceede with normal inference
            self.main_dense_processing(shift_appealing_mask,gripper_shift_reachablity_mask,suction_shift_reachablity_mask,
                                griper_grasp_score,suction_grasp_score,handover_scores)

    def handover_processing(self,handover_scores):
        self.handover_mask=handover_scores>(0.5** quality_exponent)
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
                                       view_mask= self.gripper_grasp_mask  ,view_all=False,exclude_collision=True)

        # multi_mask_view(self.voxel_pc,[self.gripper_grasp_mask,self.suction_grasp_mask,self.gripper_shift_mask,self.suction_shift_mask])
        # view_mask(self.voxel_pc, self.gripper_grasp_mask, pivot=0.5)
        # view_mask(self.voxel_pc, self.suction_grasp_mask, pivot=0.5)
        # view_mask(self.voxel_pc, self.gripper_shift_mask, pivot=0.5)
        # view_mask(self.voxel_pc, self.suction_shift_mask, pivot=0.5)

    def actions_view(self,first_action_obj:Action,second_action_obj):
        if self.last_handover_action is not None and first_action_obj.handover_state in {1,3}: return

        scene_list = []

        first_pose_mesh=first_action_obj.pose_mesh2()
        if first_pose_mesh is not None:
            first_action_obj.print_()
            scene_list+=first_pose_mesh

        second_pose_mesh=second_action_obj.pose_mesh2()
        if second_pose_mesh is not None:
            second_action_obj.print_()
            scene_list+=second_pose_mesh

        voxel_pc_numpy=self.voxel_pc.cpu().numpy()
        masked_colors = np.ones_like(voxel_pc_numpy) * [0.52, 0.8, 0.92]
        if self.first_action_mask is not None:
            masked_colors[self.first_action_mask.cpu().numpy()] /=1.1
        pcd = numpy_to_o3d(pc=voxel_pc_numpy, color=masked_colors)
        scene_list.append(pcd)

        o3d.visualization.draw_geometries(scene_list)

    def increase_gripper_penetration_distance(self,T_d,width,distance,step_factor=0.5):
        step = dist_allowance *step_factor
        n = max(int((distance_scope - distance) / step), 10)
        for i in range(n):
            T_d_new = shift_a_distance(T_d, step).clone()
            collision_intensity2 ,low_quality_grasp= grasp_collision_detection(T_d_new, width, self.voxel_pc, visualize=False,
                                                             with_allowance=False)
            if collision_intensity2 == 0:
                T_d = T_d_new
            else:
                break
        return T_d

    def process_grasp_action(self,action_obj,  view=False):
        target_point = self.voxel_pc[action_obj.point_index]
        relative_pose_7 = self.gripper_poses_7[action_obj.point_index]
        action_obj.parrelel_gripper_pose=relative_pose_7.clone()
        T_d, width, distance = pose_7_to_transformation(relative_pose_7, target_point)

        collision_intensity,low_quality_grasp = grasp_collision_detection(T_d, width, self.voxel_pc, visualize=False)
        collision_free=collision_intensity==0
        if collision_free and activate_grasp_quality_check and (low_quality_grasp is not None):
            collision_free=collision_free & (not low_quality_grasp)

        if collision_free and enhance_gripper_firmness:
            T_d=self.increase_gripper_penetration_distance(T_d,width,distance,step_factor=0.5)
        elif zero_out_distance_when_collision:
            '''apply zero penetration distance'''
            T_d = shift_a_distance(T_d, - distance).copy()
            collision_intensity,low_quality_grasp = grasp_collision_detection(T_d, width, self.voxel_pc, visualize=False)
            collision_free = collision_intensity == 0
            if collision_free and activate_grasp_quality_check and (low_quality_grasp is not None):
                collision_free = collision_free & (not low_quality_grasp)
            if collision_free and enhance_gripper_firmness:
                T_d=self.increase_gripper_penetration_distance(T_d,width,distance,step_factor=0.25)


        action_obj.is_executable= collision_free
        action_obj.width=width
        action_obj.transformation=T_d


        if view: vis_scene(T_d, width, npy=self.voxel_pc)

    def process_shift_action(self,action_obj):
        # normal = self.normals[action_obj.point_index]
        target_point = self.voxel_pc[action_obj.point_index]

        shift_dir=self.shift_directions[action_obj.point_index]
        shift_approach=torch.cat([-shift_dir,-0.707])

        # minus_normal = -normal
        v0 = torch.tensor([1., 0., 0.],device=shift_approach.device)
        a = angle_between_vectors(v0, shift_approach)  # scalar
        b = vector_product(v0, shift_approach)  # (3,)

        T_d = rotation_matrix(a, b)  # (4,4)
        T_d[:3, 3] = target_point

        action_obj.width=torch.tensor([gripper_width_during_shift],device=T_d.device)
        action_obj.transformation=T_d

        has_collision ,low_quality_grasp= grasp_collision_detection(T_d, torch.tensor([gripper_width_during_shift],device=T_d.device), self.voxel_pc, visualize=False,allowance=0.01)
        # if has_collision:
        #     grasp_collision_detection(T_d, gripper_width_during_shift, self.voxel_pc, visualize=True,allowance=0.01)

        action_obj.is_executable=not has_collision

    def process_suction_action(self,action_obj):
        normal = self.normals[action_obj.point_index]
        target_point = self.voxel_pc[action_obj.point_index]

        action_obj.transformation=rotation_matrix_from_normal_target(normal,target_point)

        has_collision ,low_quality_grasp= grasp_collision_detection(action_obj.transformation, gripper_width_during_shift, self.voxel_pc, visualize=False,allowance=0.01)
        # if has_collision:
        #     grasp_collision_detection(T, gripper_width_during_shift, self.voxel_pc, visualize=True,allowance=0.01)

        action_obj.is_executable=not has_collision

    def process_action(self,action_obj):
        if action_obj.is_grasp:
            if action_obj.use_gripper_arm:
                self.process_grasp_action(action_obj )
            else:
                self.process_suction_action(action_obj)
        else:
            '''shift action'''
            action_obj.shift_end_point=self.shift_end_points[action_obj.point_index]
            if action_obj.use_gripper_arm:
                self.process_shift_action(action_obj)
            else:
                self.process_suction_action(action_obj)

    def get_dense_knee_extremes(self,approach_vectors):
        res_elevation = knee_ref_elevation - self.voxel_pc[:,2]
        arm_knee_margin = res_elevation / (- approach_vectors[:,2] )
        arm_knee_margin=arm_knee_margin.reshape(-1,1)
        dense_extreme = (self.voxel_pc - approach_vectors * arm_knee_margin)
        assert not torch.isnan(dense_extreme).any(), f'{dense_extreme}'
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
        x_dist = torch.abs(self.voxel_pc[:, 0] - action_obj.target_point[0])
        y_dist = torch.abs(self.voxel_pc[:, 1] - action_obj.target_point[1])
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
            second_arm_extremes=self.get_dense_knee_extremes(self.gripper_approach)
            assert not torch.isnan(second_arm_extremes).any(), f'{second_arm_extremes}'

            occupancy_mask = (self.voxel_pc[:,1] > minimum_margin)\
                             | (arm_knee_extreme<(second_arm_extremes[:,1]+knee_threeshold))\
                             |  dist_mask

        self.first_action_mask=occupancy_mask
        tmp_occupation_mask[occupancy_mask]=False
        tmp_occupation_mask=tmp_occupation_mask.reshape(-1)
        return tmp_occupation_mask, second_arm_extremes

    def view_valid_actions_mask(self):
        # four_pc_stack = np.stack([self.voxel_pc, self.voxel_pc, self.voxel_pc, self.voxel_pc])
        four_pc_stack =self.voxel_pc[None,...].repeat(4,1,1).cpu().numpy()

        four_pc_stack[1, :, 0] += 0.5
        four_pc_stack[2, :, 0] += 1.0
        four_pc_stack[3, :, 0] += 1.5

        colors = np.ones_like(four_pc_stack) * [0.5, 0.9, 0.5]
        actions_mask=self.valid_actions_on_target_mask #if self.last_handover_action is not None else self.valid_actions_mask

        actions_mask=actions_mask.reshape(-1,4)
        for i in range(4):
            mask_i = (actions_mask[:, i] > 0.5).cpu().numpy()
            (colors[i])[~mask_i] *= 0.
            (colors[i])[~mask_i] += [0.9, 0.9, 0.9]

        four_pc_stack=four_pc_stack.reshape(-1,3)
        colors=colors.reshape(-1,3)

        # four_pc_stack = np.concatenate([four_pc_stack[0], four_pc_stack[1], four_pc_stack[2], four_pc_stack[3]], axis=0)
        # colors = np.concatenate([colors[0], colors[1], colors[2], colors[3]], axis=0)
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
            # if self.priority_id is not None:
            #     priority_mask = torch.zeros_like(self.valid_actions_mask,dtype=torch.bool)
            #     if self.priority_id==0:
            #         priority_mask[:,0] = True
            #     elif self.priority_id==1:
            #         priority_mask[:,1] = True
            # else:
            #     priority_mask = torch.ones_like(self.valid_actions_mask,dtype=torch.bool)

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
                            # self.last_handover_action=action_obj
                    elif action_obj.is_shift:
                        action_obj.contact_with_container=False if self.objects_mask[action_obj.point_index] else True
                    else:
                        assert False
                    first_action_obj=action_obj
                    break

            if not first_action_obj.is_executable: exit('No executable action found ...')
            first_action_obj.target_point=self.voxel_pc[first_action_obj.point_index]
            first_action_obj.print_()

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
                second_action_obj.print_()
                # print('second arm extreme from dense extrems',second_arm_extremes[second_action_obj.point_index])

        return first_action_obj,second_action_obj

    def wait_for_robot(self):
        # wait until grasp or suction finished
        robot_feedback_ = 'Wait'
        wait = wi('Waiting for robot ...')
        print()
        counter=0
        while robot_feedback_ == 'Wait' or robot_feedback_.strip()=='':
            wait.step(0.1)
            robot_feedback_ = read_robot_feedback()
            counter+=1
        else:
            wait.end()


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
                self.ini_shift_policy()
                print('Update shift policy')
            elif counter == 4:
                self.ini_grasp_handover_policy()
                print('Update grasp and handover policy')
            elif counter==5:
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

                gripper_action, suction_action = save_grasp_sample(self.rgb, self.depth[0,0].cpu().numpy(), self.mask_numpy,
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
                gripper_action,suction_action=save_grasp_sample(self.rgb, self.depth[0,0].cpu().numpy(),self.mask_numpy, gripper_action, suction_action,self.run_sequence)

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

                self.last_handover_action=first_action_obj
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
                print('Reset arm ...')
                # set_wait_flag()
                subprocess.run(["bash", './bash/pass_command.sh', "5"])
                # self.wait_for_robot()
                # trigger_new_perception()
                # new_state_available=True

        return new_state_available

'''conventions'''
# normal is a vector emerge out of the surface
# approach direction is a vector pointing to the surface
# approach = -1 * normal
# The first three parameters of the gripper pose are approach[0] and approach [1] and -1* approach[2]
# the suction sampler outputs the normal direction
# T_0 refers to a gripper head transformation matrix with zero penetration while T_d embeds the distance term
# for any sequence we will always give the gripper the first index followed by the suction, e.g. if gripper grasp score locate at the (i) channel then the suction is located at (i+1) channel
# if the action is shift, it is always saved in the first action object and no action is assigned to the second arm
# executing both arms at the same time is only allowed when both actions are grasp
# a single run may include one action or two actions (moving both arms)
# After execution, the robot rises three flags:
    # succeed: the plan has been executed completely
    # failed: Unable to execute part or full of the path
    # reset: path plan is found but execution terminated due to an error
