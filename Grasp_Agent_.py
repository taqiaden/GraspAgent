import os.path
import subprocess

import numpy as np
import torch
import trimesh
from colorama import Fore
import open3d as o3d
from Online_data_audit.data_tracker2 import DataTracker2
from action import Action
from lib.IO_utils import save_pickle, load_pickle
from lib.grasp_utils import shift_a_distance
from lib.models_utils import number_of_parameters
from lib.report_utils import wait_indicator as wi
from Configurations.ENV_boundaries import bin_center, dist_allowance
from Configurations.config import distance_scope, gripper_width_during_shift
from Configurations.run_config import simulation_mode, \
    suction_factor, gripper_factor, report_result, \
    enhance_gripper_firmness, single_arm_operation_mode, \
    gripper_grasp,gripper_shift,suction_grasp,suction_shift
from Online_data_audit.process_feedback import save_grasp_sample
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
from registration import camera
from training.learning_objectives.shift_affordnace import shift_execution_length
from training.policy_lr import PPOLearning, PPOMemory
from visualiztion import view_npy_open3d, vis_scene

execute_suction_grasp_bash = './bash/run_robot_suction.sh'
execute_gripper_grasp_bash = './bash/run_robot_grasp.sh'
execute_suction_shift_bash = './bash/run_robot_suction_shift.sh'
execute_gripper_shift_bash = './bash/run_robot_gripper_shift.sh'
execute_both_grasp_bash = './bash/run_robot_grasp_and_suction.sh'

buffer_file='buffer.pkl'
action_data_tracker_path=r'online_data_dict'

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
    voxel_pc_t[:,1:]*=-1
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
        self.action_net = None
        self.policy_net = None
        self.suction_arm_reachability_net = None
        self.gripper_arm_reachability_net = None

        self.buffer=load_pickle(buffer_file) if os.path.exists(buffer_file) else PPOMemory()
        self.data_tracker = DataTracker2(name=action_data_tracker_path, list_size=10)
        self.online_learning=PPOLearning()

        '''modalities'''
        self.point_clouds = None
        self.depth=None
        self.rgb=None

        self.policy_learning=PPOLearning(model=self.policy_net)

        '''dense records'''
        # self.quality_masks=None
        self.gripper_poses_5=None
        self.gripper_poses_7=None
        self.gripper_grasp_mask=None
        self.suction_grasp_mask=None
        self.gripper_shift_mask=None
        self.suction_shift_mask=None
        self.voxel_pc=None
        self.normals=None
        self.q_value=None
        self.action_probs=None
        self.shift_end_points=None
        self.valid_actions_mask=None
        self.n_grasps=0
        self.n_shifts=0
        self.first_action_mask=None
        self.target_object_mask=None
        self.mask_numpy=None
        self.valid_actions_on_target_mask=None

        '''track task sequence'''
        self.run_sequence=0

    def clear(self):
        # self.quality_masks = None
        self.gripper_poses_5 = None
        self.gripper_poses_7 = None
        self.gripper_grasp_mask = None
        self.suction_grasp_mask = None
        self.gripper_shift_mask = None
        self.suction_shift_mask = None
        self.voxel_pc = None
        self.normals = None
        self.q_value = None
        self.action_probs = None
        self.shift_end_points = None
        self.valid_actions_mask = None
        self.n_grasps = 0
        self.n_shifts = 0
        self.first_action_mask = None
        self.target_object_mask = None
        self.valid_actions_on_target_mask = None
        self.mask_numpy=None


    @property
    def gripper_approach(self):
        approach=self.gripper_poses_7[:,0:3].clone()
        approach[:,2]*=-1
        return approach

    @property
    def suction_approach(self):
        return self.normals*-1

    def initialize_check_points(self):
        pi = progress_indicator('Loading check points  ', max_limit=5)

        pi.step(1)

        action_net = GANWrapper(action_module_key, ActionNet)
        action_net.ini_generator(train=False)
        self.action_net = action_net.generator

        pi.step(2)

        policy_net = ModelWrapper(model=PolicyNet(), module_key=policy_module_key)

        policy_net.ini_model(train=False)
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

    def get_suction_shift_reachability(self,positions,normals):
        approach=-normals.clone()
        suction_scope_a = self.suction_arm_reachability_net(torch.cat([positions, approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        if self.shift_end_points is None:
            self.shift_end_points = get_shift_end_points(positions)
        suction_scope_b = self.suction_arm_reachability_net(torch.cat([self.shift_end_points, approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        result=torch.stack([suction_scope_a,suction_scope_b],dim=-1)
        result,_=torch.min(result,dim=-1)
        return result

    def get_gripper_grasp_reachability(self,positions,poses):
        gripper_approach=(poses[:,0:3]).clone()
        gripper_approach[:,2]*=-1
        distance=poses[:,-2:-1]*distance_scope
        transition=positions+distance*gripper_approach
        gripper_scope=self.gripper_arm_reachability_net(torch.cat([transition, gripper_approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        return gripper_scope

    def get_gripper_shift_reachability(self,positions,normals):
        gripper_approach=normals.clone()
        gripper_approach[:,2]*=-1
        gripper_scope_a=self.gripper_arm_reachability_net(torch.cat([positions, gripper_approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        self.shift_end_points = get_shift_end_points(positions)
        gripper_scope_b=self.gripper_arm_reachability_net(torch.cat([self.shift_end_points, gripper_approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        result = torch.stack([gripper_scope_a, gripper_scope_b], dim=-1)
        result, _ = torch.min(result, dim=-1)
        return result

    def next_action(self,sample_from_target_actions=False):
        mask_=self.valid_actions_on_target_mask if sample_from_target_actions else self.valid_actions_mask
        dist=MaskedCategorical(probs=self.action_probs,mask=mask_)
        flattened_action_index= dist.sample()
        probs = torch.squeeze(dist.log_prob(flattened_action_index)).item()
        flattened_action_index = torch.squeeze(flattened_action_index).item()
        value = self.q_value[flattened_action_index]
        value = torch.squeeze(value).item()
        if sample_from_target_actions:
            self.valid_actions_on_target_mask[flattened_action_index]=False
        self.valid_actions_mask[flattened_action_index]=False
        return flattened_action_index, probs, value

    def prepare_quality_masks(self,mask,suction_seal_classifier,griper_collision_classifier,shift_appealing,
                              background_class,gripper_grasp_scope,suction_grasp_scope,gripper_shift_scope,suction_shift_scope):
        gripper_collision_mask=suction_seal_classifier.detach() > 0.5
        suction_seal_mask=griper_collision_classifier.detach() > 0.5
        shift_appeal_mask=shift_appealing.detach() > 0.5
        self.target_object_mask=background_class.detach() <= 0.5
        gripper_grasp_mask=torch.zeros_like(shift_appeal_mask)
        gripper_grasp_mask[:,:,mask]=gripper_grasp_scope.detach() >0.5
        suction_grasp_mask=torch.zeros_like(shift_appeal_mask)
        suction_grasp_mask[:,:,mask]=suction_grasp_scope.detach() >0.5
        gripper_shift_mask=torch.zeros_like(shift_appeal_mask)
        gripper_shift_mask[:,:,mask]=gripper_shift_scope.detach() >0.5
        suction_shift_mask=torch.zeros_like(shift_appeal_mask)
        suction_shift_mask[:,:,mask]=suction_shift_scope.detach() >0.5

        quality_masks=torch.cat([gripper_collision_mask,suction_seal_mask,shift_appeal_mask,gripper_grasp_mask,
                                 suction_grasp_mask,gripper_shift_mask,suction_shift_mask,self.target_object_mask],dim=1)

        return quality_masks

    def model_inference(self,depth,rgb):
        pr.title('model inference')
        self.depth=depth
        self.rgb=rgb
        depth_torch = torch.from_numpy(self.depth)[None, None, ...].to('cuda').float()
        rgb_torch = torch.from_numpy(self.rgb).permute(2,0,1)[None, ...].to('cuda').float()

        '''action net output'''
        gripper_pose, suction_direction, griper_collision_classifier, suction_seal_classifier, shift_appealing \
            , background_class, depth_features = self.action_net(depth_torch.clone(),clip=True)

        '''depth to point clouds'''
        self.voxel_pc, mask = depth_to_point_clouds(self.depth, camera)
        self.voxel_pc = transform_to_camera_frame(self.voxel_pc, reverse=True)
        voxel_pc_tensor = torch.from_numpy(self.voxel_pc).to('cuda').float()
        self.normals = suction_direction.squeeze().permute(1, 2, 0)[mask] # [N,3]
        poses = gripper_pose.squeeze().permute(1, 2, 0)[mask]

        '''grasp reachability'''
        suction_grasp_scope = self.get_suction_grasp_reachability(voxel_pc_tensor,self.normals)
        gripper_grasp_scope=self.get_gripper_grasp_reachability(voxel_pc_tensor,poses)

        '''shift reachability'''
        gripper_shift_scope=self.get_gripper_shift_reachability( voxel_pc_tensor,self.normals)
        suction_shift_scope=self.get_suction_shift_reachability(voxel_pc_tensor,self.normals)
        '''shift with the arm of best reachablity'''
        gripper_shift_scope[suction_shift_scope>gripper_shift_scope]*=0.
        suction_shift_scope[gripper_shift_scope>suction_shift_scope]*=0.

        '''target mask'''
        self.target_object_mask = background_class.detach() <= 0.5
        self.mask_numpy=self.target_object_mask.cpu().numpy()

        '''policy net output'''
        griper_grasp_score, suction_grasp_score,\
        shift_affordance_classifier, q_value, action_probs = \
            self.policy_net(rgb_torch,gripper_pose, suction_direction,self.target_object_mask)

        '''reshape'''
        griper_object_collision_classifier=griper_collision_classifier[0,0][mask]
        griper_bin_collision_classifier=griper_collision_classifier[0,0][mask]

        griper_grasp_score=griper_grasp_score.squeeze()[mask]
        suction_seal_classifier=suction_seal_classifier.squeeze()[mask]
        suction_grasp_score=suction_grasp_score.squeeze()[mask]
        background_class=background_class.squeeze()[mask]
        shift_appealing=shift_appealing.squeeze()[mask]
        self.target_object_mask=self.target_object_mask.squeeze()[mask]

        griper_grasp_score.fill_(1.)
        suction_grasp_score.fill_(1.)

        '''correct backgoround mask'''
        # min_elevation=voxel_pc_tensor[background_class<0.5,-1].min().item()
        # background_class[self.voxel_pc[:,-1]<min_elevation+0.005]=1.0

        '''actions masks'''
        object_mask=background_class<0.4
        gripper_grasp_reachablity_mask=gripper_grasp_scope>0.5
        gripper_collision_mask=(griper_object_collision_classifier<0.5) & (griper_bin_collision_classifier<0.5)
        gripper_grasbablity_mask=griper_grasp_score*gripper_factor>0.5
        suction_grasp_reachablity_mask=suction_grasp_scope>0.5
        seal_quality_mask=suction_seal_classifier>0.5
        suctionablity_mask=suction_grasp_score*suction_factor>0.5
        shift_appealing_mask=shift_appealing>0.5
        gripper_shift_reachablity_mask=gripper_shift_scope>0.5
        suction_shift_reachablity_mask=suction_shift_scope>0.5

        view_mask(self.voxel_pc,background_class<0.5)
        # view_mask(voxel_pc_,shift_appealing>0.5)

        '''grasp actions'''
        self.gripper_grasp_mask=(object_mask*gripper_grasp_reachablity_mask
                               *gripper_collision_mask
                               *gripper_grasbablity_mask* gripper_grasp)
        self.suction_grasp_mask=(object_mask*suction_grasp_reachablity_mask
                               *seal_quality_mask
                               *suctionablity_mask* suction_grasp)
        # view_mask(voxel_pc_,self.gripper_grasp_mask)
        # view_mask(voxel_pc_,self.suction_grasp_mask)

        '''shift actions'''
        self.gripper_shift_mask=(shift_appealing_mask*gripper_shift_reachablity_mask* gripper_shift)#*(shift_affordance_classifier>0.5)
        self.suction_shift_mask=(shift_appealing_mask*suction_shift_reachablity_mask* suction_shift) #*(shift_affordance_classifier>0.5)

        # view_mask(voxel_pc_,(shift_appealing>0.5))
        # view_mask(voxel_pc_,(suction_shift_scope>0.5))
        # view_mask(voxel_pc_,self.suction_shift_mask)

        '''gripper pose convention'''
        self.gripper_poses_7=poses
        self.gripper_poses_5 = vectors_to_ratio_metrics(poses.clone())

        '''initiate random policy'''
        # self.q_value=torch.rand_like(q_value.squeeze().permute(1,2,0)[mask])
        self.q_value=q_value.squeeze().permute(1,2,0)[mask]
        self.action_probs=action_probs.squeeze().permute(1,2,0)[mask]

        self.valid_actions_mask=torch.zeros_like(self.q_value,dtype=torch.bool)
        self.valid_actions_on_target_mask=torch.zeros_like(self.q_value,dtype=torch.bool)
        self.valid_actions_on_target_mask[:,0].masked_fill_(self.gripper_grasp_mask,True)
        self.valid_actions_on_target_mask[:,1].masked_fill_(self.suction_grasp_mask,True)

        '''initialize valid actions mask'''
        self.valid_actions_mask[:,0].masked_fill_(self.gripper_grasp_mask,True)
        self.valid_actions_mask[:,1].masked_fill_(self.suction_grasp_mask,True)
        self.valid_actions_mask[:,2].masked_fill_(self.gripper_shift_mask,True)
        self.valid_actions_mask[:,3].masked_fill_(self.suction_shift_mask,True)

        '''count available actions'''
        self.n_grasps=torch.count_nonzero(self.valid_actions_mask[:,0:2])
        self.n_shifts=torch.count_nonzero(self.valid_actions_mask[:,2:4])

        '''to numpy'''
        self.normals=self.normals.cpu().numpy()

        '''flatten'''
        self.q_value = self.q_value.reshape(-1)
        self.action_probs = self.action_probs.reshape(-1)
        self.valid_actions_mask = self.valid_actions_mask.reshape(-1)
        self.valid_actions_on_target_mask = self.valid_actions_on_target_mask.reshape(-1)

    def dense_view(self):
        print(Fore.CYAN, f'Action space includes {self.n_grasps} grasps and {self.n_shifts} shifts',Fore.RESET)
        self.view_valid_actions_mask()
        # multi_mask_view(self.voxel_pc,[self.gripper_grasp_mask,self.suction_grasp_mask,self.gripper_shift_mask,self.suction_shift_mask])
        # view_mask(self.voxel_pc, self.gripper_grasp_mask, pivot=0.5)
        # view_mask(self.voxel_pc, self.suction_grasp_mask, pivot=0.5)
        # view_mask(self.voxel_pc, self.gripper_shift_mask, pivot=0.5)
        # view_mask(self.voxel_pc, self.suction_shift_mask, pivot=0.5)

    def actions_view(self,first_action_obj,second_action_obj):
        scene_list = []
        first_pose_mesh=first_action_obj.pose_mesh()
        if first_pose_mesh is not None: scene_list.append(first_pose_mesh)
        second_pose_mesh=second_action_obj.pose_mesh()
        if second_pose_mesh is not None: scene_list.append(second_pose_mesh)
        masked_colors = np.ones_like(self.voxel_pc) * [0.52, 0.8, 0.92]

        if self.first_action_mask is not None:
            masked_colors[self.first_action_mask] /=1.1
        pcd = numpy_to_o3d(pc=self.voxel_pc, color=masked_colors)
        scene_list.append(pcd)
        o3d.visualization.draw_geometries(scene_list)

    def gripper_grasp_processing(self,action_obj,  view=False):
        target_point = self.voxel_pc[action_obj.point_index]
        relative_pose_5 = self.gripper_poses_5[action_obj.point_index]
        T_d, width, distance = convert_angles_to_transformation_form(relative_pose_5, target_point)

        # activate_exploration = True if np.random.rand() < exploration_probabilty else False

        collision_intensity = grasp_collision_detection(T_d, width, self.voxel_pc, visualize=False)
        if collision_intensity==0 and enhance_gripper_firmness:
            step=dist_allowance/2
            n=max(int((distance_scope-distance)/step),10)
            for i in range(n):
                T_d_new = shift_a_distance(T_d, step).copy()
                collision_intensity2 = grasp_collision_detection(T_d_new, width, self.voxel_pc, visualize=False,with_allowance=False)
                if collision_intensity2==0:T_d=T_d_new
                else:break


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
        action_obj.is_executable=True

    def suction_processing(self,action_obj):
        normal = self.normals[action_obj.point_index]
        target_point = self.voxel_pc[action_obj.point_index]

        v0 = np.array([1, 0, 0])
        a = trimesh.transformations.angle_between_vectors(v0, -normal)
        b = trimesh.transformations.vector_product(v0, -normal)
        T = trimesh.transformations.rotation_matrix(a, b)
        T[:3, 3] = target_point.T
        
        action_obj.transformation=T
        action_obj.is_executable=True

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

    def mask_arm_occupancy(self,action_obj):
        self.valid_actions_mask=self.valid_actions_mask.reshape(-1,4)

        '''mask occupied arm'''
        if action_obj.use_gripper_arm:
            self.valid_actions_mask[:, [0, 2]]=False
        else:
            self.valid_actions_mask[:, [1, 3]]=False


        '''mask occupied space'''
        arm_knee_margin = 0.5
        normal=action_obj.normal
        action_obj.target_point=self.voxel_pc[action_obj.point_index]
        arm_knee_extreme=(action_obj.target_point + normal * arm_knee_margin)[1]
        minimum_safety_margin=0.1
        x_dist = np.abs(self.voxel_pc[:, 0] - action_obj.target_point[0])
        y_dist = np.abs(self.voxel_pc[:, 1] - action_obj.target_point[1])
        dist_mask=(x_dist<minimum_safety_margin) & (y_dist<minimum_safety_margin)

        if action_obj.use_gripper_arm:
            other_arm_approach=self.suction_approach
            minimum_margin = action_obj.target_point[1] - minimum_safety_margin
            second_arm_extreme=(self.voxel_pc-other_arm_approach*arm_knee_margin)[:,1]


            occupancy_mask = (self.voxel_pc[:,1] < minimum_margin) | (arm_knee_extreme>second_arm_extreme) | dist_mask
        else:
            other_arm_approach=self.gripper_approach.cpu().numpy()
            second_arm_extreme=(self.voxel_pc-other_arm_approach*arm_knee_margin)[:,1]
            minimum_margin = action_obj.target_point[1] + minimum_safety_margin

            occupancy_mask = (self.voxel_pc[:,1] > minimum_margin) | (second_arm_extreme>arm_knee_extreme) |  dist_mask

        self.first_action_mask=occupancy_mask
        # print(occupancy_mask.shape)
        # print(self.valid_actions_mask.shape)
        self.valid_actions_mask[occupancy_mask]=False
        self.valid_actions_mask=self.valid_actions_mask.reshape(-1)
        self.valid_actions_on_target_mask=self.valid_actions_on_target_mask & self.valid_actions_mask

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

    def pick_action(self):
        pr.title('pick action/s')
        first_action_obj=Action()
        second_action_obj=Action()
        self.first_action_mask=None

        '''first action'''
        total_available_actions=torch.count_nonzero(self.valid_actions_mask).item()
        available_actions_on_target=torch.count_nonzero(self.valid_actions_on_target_mask).item()
        for i in range(total_available_actions):
            flattened_action_index, probs, value=self.next_action(sample_from_target_actions=i<available_actions_on_target)
            unflatten_index = get_unflatten_index(flattened_action_index, ori_size=(self.voxel_pc.shape[0],4))
            action_obj=Action(point_index=unflatten_index[0],action_index=unflatten_index[1], probs=probs, value=value)
            self.process_action(action_obj)
            if action_obj.is_executable:
                first_action_obj=action_obj
                if i<available_actions_on_target:first_action_obj.policy_index=1
                if action_obj.is_grasp:
                    self.mask_arm_occupancy(action_obj)
                break

        if not first_action_obj.is_executable: exit('No executable action found ...')
        first_action_obj.target_point=self.voxel_pc[first_action_obj.point_index]
        first_action_obj.print()

        if first_action_obj.policy_index==1:
            self.valid_actions_mask=self.valid_actions_on_target_mask
        else:
            self.valid_actions_mask = self.valid_actions_mask.reshape(-1, 4)
            self.valid_actions_mask[:, 2:]=False  # simultaneous operation of both arms are for grasp actions only
            self.valid_actions_mask=self.valid_actions_mask.reshape(-1)

        if first_action_obj.is_shift or single_arm_operation_mode:
            return first_action_obj, second_action_obj

        '''second action'''
        total_available_actions=torch.count_nonzero(self.valid_actions_mask).item()
        for i in range(total_available_actions):
            flattened_action_index, probs, value = self.next_action()
            unflatten_index = get_unflatten_index(flattened_action_index, ori_size=(self.voxel_pc.shape[0], 4))
            action_obj = Action(point_index=unflatten_index[0],action_index=unflatten_index[1], probs=probs, value=value)
            self.process_action(action_obj)
            if action_obj.is_executable:
                second_action_obj=action_obj
                break

        if second_action_obj.is_executable:
            second_action_obj.target_point=self.voxel_pc[second_action_obj.point_index]
            first_action_obj.is_synchronous=True
            second_action_obj.is_synchronous=True
            second_action_obj.print()

        return first_action_obj,second_action_obj

    def wait_robot_feedback(self,first_action_obj,second_action_obj):
        # wait until grasp or suction finished
        robot_feedback_ = 'Wait'
        wait = wi('Waiting for robot feedback')
        counter=0
        while robot_feedback_ == 'Wait':
            wait.step(0.5)
            if counter==0:
                '''reduce buffer size'''
                self.buffer.pop()
                '''dump the buffer as pickl'''
                save_pickle(buffer_file,self.buffer)
                '''save data tracker'''
                self.data_tracker.save()
            elif counter==1:
                if self.buffer.episodes_counter>0:
                    '''step policy training'''
                    self.online_learning.step(self.policy_net,self.buffer)
            robot_feedback_ = read_robot_feedback()
            counter+=1
        else:
            wait.end()
            print('Robot msg: ' + robot_feedback_)
        first_action_obj.robot_feedback = robot_feedback_
        second_action_obj.robot_feedback = robot_feedback_
        return first_action_obj,second_action_obj

    def execute(self,first_action_obj,second_action_obj):
        pr.title('execute action')
        pr.print('Deploy action commands')
        if not first_action_obj.is_valid: return first_action_obj,second_action_obj
        deploy_action( first_action_obj)
        if  second_action_obj.is_valid: deploy_action(second_action_obj)

        if not simulation_mode:
            pr.print('Run robot')
            set_wait_flag()
            if second_action_obj.is_valid and (first_action_obj.is_grasp and second_action_obj.is_grasp):
                '''grasp with the two arms'''
                subprocess.run(execute_both_grasp_bash)
            elif first_action_obj.is_grasp:
                '''grasp'''
                if first_action_obj.use_gripper_arm:
                    subprocess.run(execute_gripper_grasp_bash)
                else:
                    '''suction'''
                    subprocess.run(execute_suction_grasp_bash)
            elif first_action_obj.is_shift:
                '''shift'''
                if first_action_obj.use_gripper_arm:
                    subprocess.run(execute_gripper_shift_bash)
                else:
                    '''suction'''
                    subprocess.run(execute_suction_shift_bash)

        return first_action_obj,second_action_obj

    def process_feedback(self,first_action_obj:Action,second_action_obj:Action, img_grasp_pre, img_suction_pre,img_main_pre):
        pr.title('process feedback')
        if first_action_obj.robot_feedback == 'Succeed' or first_action_obj.robot_feedback == 'reset':
            trigger_new_perception()
        if  report_result:
            if first_action_obj.policy_index==1:self.run_sequence=0

            img_suction_after, img_grasp_after,img_main_after = get_side_bins_images()

            '''save feedback to data pool'''
            if first_action_obj.robot_feedback == 'Succeed':
                first_action_obj.executed=True
                second_action_obj.executed=True
                if first_action_obj.is_shift:
                    first_action_obj.shift_result=check_image_similarity(img_main_pre, img_main_after)

                if first_action_obj.use_gripper_arm:
                    gripper_action=first_action_obj
                    suction_action=second_action_obj
                else:
                    gripper_action = second_action_obj
                    suction_action = first_action_obj

                '''check changes in side bins'''
                if gripper_action.is_grasp:
                    gripper_action.grasp_result=check_image_similarity(img_grasp_pre, img_grasp_after)
                    if gripper_action.grasp_result is None:
                        print(Fore.LIGHTCYAN_EX, 'Unable to detect the grasp result for the gripper',Fore.RESET)
                    elif gripper_action.grasp_result:
                        print(Fore.GREEN, 'A new object is detected at the gripper side of the bin',Fore.RESET)
                    else:
                        print(Fore.GREEN, 'No object is detected at to the gripper side of the bin',Fore.RESET)


                if suction_action.is_grasp:
                    suction_action.grasp_result=check_image_similarity(img_suction_pre, img_suction_after)
                    if suction_action.grasp_result is None:
                        print(Fore.LIGHTCYAN_EX, 'Unable to detect the grasp result for the suction',Fore.RESET)
                    elif suction_action.grasp_result:
                        print(Fore.GREEN, 'A new object is detected at the suction side of the bin',Fore.RESET)
                    else:
                        print(Fore.GREEN, 'No object is detected at to the suction side of the bin',Fore.RESET)


                '''save action instance'''
                if gripper_action.result is not None or suction_action.result is not None:
                    save_grasp_sample(self.rgb, self.depth,self.mask_numpy, gripper_action, suction_action,self.run_sequence)
                self.run_sequence+=1

                '''update buffer and tracker'''
                if gripper_action.is_executable:
                    self.buffer.push(gripper_action)
                    self.data_tracker.push(gripper_action)
                if suction_action.is_executable:
                    self.buffer.push(suction_action)
                    self.data_tracker.push(suction_action)
            else:
                first_action_obj.executed=False
                second_action_obj.executed=False

'''conventions'''
# normal is a vector emerge out of the surface
# approach direction is a vector pointing to the surface
# approach = -1 * normal
# the gripper sampler first three parameters are approach[0] and approach [1] and -1* approach[2]
# the suction sampler outputs the normal direction
# T_0 refers to a gripper head transformation matrix with zero penetration while T_d embeds distance term
# for any sequence we will always give the gripper the first index followed by the suction, e.g. if gripper grasp score locate at the (i) channel then the suction is located at (i+1) channel
# if the action is shift, it is always saved in the first action object
# executing both arms at the same time will be only when both actions are grasp actions
# a single run may include one action or two actions (moving both arms)
# after execution the robot rises three flage:
    # succeed: the plan has been executed completly
    # failed: Unable to execute part or full of the path
    # reset: path plan is found but execution termimnated due to an error