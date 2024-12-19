import math
import subprocess

import numpy as np
import torch
import trimesh
from colorama import Fore

from Configurations.ENV_boundaries import bin_center
from Configurations.config import distance_scope
from Configurations.run_config import simulation_mode, \
    suction_factor, gripper_factor, report_result, use_gripper, use_suction
from Online_data_audit.process_feedback import save_grasp_sample
from check_points.check_point_conventions import GANWrapper, ModelWrapper

from grasp_post_processing import  exploration_probabilty, view_masked_grasp_pose
from lib.ROS_communication import wait_for_feedback, deploy_action
from lib.bbox import convert_angles_to_transformation_form
from lib.custom_print import my_print

from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from lib.gripper_exploration import local_exploration
from lib.image_utils import check_image_similarity
from lib.report_utils import progress_indicator
from models.action_net import ActionNet, action_module_key
from models.scope_net import scope_net_vanilla, gripper_scope_module_key, suction_scope_module_key
from models.value_net import ValueNet, value_module_key
from pose_object import vectors_to_ratio_metrics
from process_perception import get_side_bins_images, trigger_new_perception
from registration import camera
from training.learning_objectives.shift_affordnace import shift_execution_length
from visualiztion import view_npy_open3d, vis_scene

execute_suction_grasp_bash = './bash/run_robot_suction.sh'
execute_gripper_grasp_bash = './bash/run_robot_grasp.sh'
execute_suction_shift_bash = './bash/run_robot_suction_shift.sh'
execute_gripper_shift_bash = './bash/run_robot_gripper_shift.sh'
execute_both_grasp_bash = './bash/run_robot_grasp_and_suction.sh'

pr=my_print()

class Action():
    def __init__(self,point_index,action_index):
        self.point_index=point_index
        self.action_index=action_index
        self.is_shift=action_index>1
        self.is_grasp=action_index<=1
        self.use_gripper_arm=(action_index==0) or (action_index==2)
        self.use_suction_arm=(action_index==1) or (action_index==3)

        self.target_point=None
        
        self.transformation=None
        self.width=None
        
        self.is_executable=None

        self.robot_feedback=None

        self.grasp_result=None
        self.shift_result=None
        

def get_shift_end_points(start_points):
    targets=torch.zeros_like(start_points)
    targets[:,0:2]+=torch.from_numpy(bin_center[0:2]).cuda()
    targets[:,2] += start_points[:,2]
    directions = targets - start_points
    end_points = start_points + ((directions * shift_execution_length) / torch.linalg.norm(directions,axis=-1,keepdims=True))
    return end_points

def view_mask(voxel_pc, score, pivot=0.5):
    mask_=score.cpu().numpy()>pivot
    if mask_.sum() > 0:
        colors = np.ones_like(voxel_pc) * [0.52, 0.8, 0.92]
        colors[~mask_] /= 1.5
        view_npy_open3d(voxel_pc, color=colors)


class GraspAgent():
    def __init__(self):

        '''models'''
        self.action_net = None
        self.value_net = None

        self.suction_arm_reachability_net = None
        self.gripper_arm_reachability_net = None

        '''modalities'''
        self.point_clouds = None
        self.depth=None
        self.rgb=None

        '''dense candidates'''
        self.gripper_poses=None
        self.gripper_grasp_mask=None
        self.suction_grasp_mask=None
        self.gripper_shift_mask=None
        self.suction_shift_mask=None
        self.voxel_pc=None
        self.normals=None
        self.q_value=None
        self.trace_mask=None

        self.n_grasps=0
        self.n_shifts=0

        '''execution results'''
        self.actions=[]
        self.states=[]
        self.data=[]

    def initialize_check_points(self):
        pi = progress_indicator('Loading check points  ', max_limit=5)

        pi.step(1)

        action_net = GANWrapper(action_module_key, ActionNet)
        action_net.ini_generator(train=False)
        self.action_net = action_net.generator

        pi.step(2)

        value_net = ModelWrapper(model=ValueNet(), module_key=value_module_key)
        value_net.ini_model(train=False)
        self.value_net = value_net.model

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

    def get_suction_grasp_reachability(self,positions,normals):
        approach=normals.clone()
        approach[:,2]*=-1
        suction_scope = self.suction_arm_reachability_net(torch.cat([positions, approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        return suction_scope

    def get_suction_shift_reachability(self,positions,normals):
        approach=normals.clone()
        approach[:,2]*=-1
        suction_scope_a = self.suction_arm_reachability_net(torch.cat([positions, approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        shift_end_positions=get_shift_end_points(positions)
        suction_scope_b = self.suction_arm_reachability_net(torch.cat([shift_end_positions, approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
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
        shift_end_positions = get_shift_end_points(positions)
        gripper_scope_b=self.gripper_arm_reachability_net(torch.cat([shift_end_positions, gripper_approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        result = torch.stack([gripper_scope_a, gripper_scope_b], dim=-1)
        result, _ = torch.min(result, dim=-1)
        return result

    def next_action(self, epsilon=0.0):
        masked_q_value=self.q_value*(1-self.trace_mask)
        if np.random.rand()>epsilon:
            flattened_index=torch.argmax(masked_q_value)
            point_index = math.floor(flattened_index / 4)
            action_index = flattened_index - int(4 * point_index)
        else:
            available_indexes=torch.nonzero(masked_q_value)
            random_pick=np.random.random_integers(0,available_indexes.shape[0]-1)
            point_index = available_indexes[random_pick][0]
            action_index = available_indexes[random_pick][1]

        self.trace_mask[point_index,action_index]=1
        return point_index,action_index


    def model_inference(self,depth,rgb):
        pr.title('model inference')
        self.depth=depth
        self.rgb=rgb
        depth_torch = torch.from_numpy(self.depth)[None, None, ...].to('cuda').float()
        rgb_torch = torch.from_numpy(self.rgb).permute(2,0,1)[None, ...].to('cuda').float()

        '''action net output'''
        gripper_pose, suction_direction, griper_collision_classifier, suction_seal_classifier, shift_appealing \
            , background_class, depth_features = self.action_net(depth_torch.clone())

        '''value net output'''
        griper_grasp_score, suction_grasp_score, shift_affordance_classifier, q_value = self.value_net(rgb_torch,
                                                                                                             depth_features,
                                                                                                             gripper_pose,
                                                                                                             suction_direction)

        '''depth to point clouds'''
        self.voxel_pc, mask = depth_to_point_clouds(self.depth, camera)
        self.voxel_pc = transform_to_camera_frame(self.voxel_pc, reverse=True)

        '''pixels to points'''
        self.normals = suction_direction.squeeze().permute(1, 2, 0)[mask] # [N,3]
        poses = gripper_pose.squeeze().permute(1, 2, 0)[mask]
        griper_collision_classifier=griper_collision_classifier.squeeze()[mask]
        griper_grasp_score=griper_grasp_score.squeeze()[mask]
        suction_seal_classifier=suction_seal_classifier.squeeze()[mask]
        suction_grasp_score=suction_grasp_score.squeeze()[mask]
        background_class=background_class.squeeze()[mask]
        shift_appealing=shift_appealing.squeeze()[mask]
        positions = torch.from_numpy(self.voxel_pc).to('cuda').float()

        '''grasp reachability'''
        suction_grasp_scope = self.get_suction_grasp_reachability(positions,self.normals)
        gripper_grasp_scope=self.get_gripper_grasp_reachability(positions,poses)

        '''grasp actions'''
        self.gripper_grasp_mask=((background_class<0.5)*(gripper_grasp_scope>0.5)
                               *(griper_collision_classifier>0.5)
                               *(griper_grasp_score*gripper_factor>0.5)* int(use_gripper))
        self.suction_grasp_mask=((background_class<0.5)*(suction_grasp_scope>0.5)
                               *(suction_seal_classifier>0.5)
                               *(suction_grasp_score*suction_factor>0.5)* int(use_suction))
        self.suction_grasp_mask=self.suction_grasp_mask.squeeze()
        self.gripper_grasp_mask=self.gripper_grasp_mask.squeeze()

        '''shift reachability'''
        gripper_shift_scope=self.get_gripper_shift_reachability( positions,self.normals)
        suction_shift_scope=self.get_suction_shift_reachability(positions,self.normals)

        '''shift actions'''
        self.gripper_shift_mask=(shift_appealing>0.5)*(gripper_shift_scope>0.5)#*(shift_affordance_classifier>0.5)
        self.suction_shift_mask=(shift_appealing>0.5)*(suction_shift_scope>0.5)#*(shift_affordance_classifier>0.5)

        '''gripper pose convention'''
        self.gripper_poses = vectors_to_ratio_metrics(poses)

        '''initiate random policy'''
        self.q_value=torch.randn_like(q_value.squeeze().permute(1,2,0)[mask])
        self.trace_mask=torch.zeros_like(self.q_value)

        '''mask the action-value'''
        self.q_value[:,0]*=self.gripper_grasp_mask
        self.q_value[:,1]*=self.suction_grasp_mask
        self.q_value[:,2]*=self.gripper_shift_mask
        self.q_value[:,2]*=self.suction_shift_mask

        '''count available actions'''
        self.n_grasps=torch.count_nonzero(self.q_value[:,0:2])
        self.n_shifts=torch.count_nonzero(self.q_value[:,2:4])

        print(Fore.CYAN, f'Action space includes {self.n_grasps} grasps and {self.n_shifts} shifts',Fore.RESET)

    def view(self):
        view_mask(self.voxel_pc, self.gripper_grasp_mask, pivot=0.5)
        view_mask(self.voxel_pc, self.suction_grasp_mask, pivot=0.5)
        view_mask(self.voxel_pc, self.gripper_shift_mask, pivot=0.5)
        view_mask(self.voxel_pc, self.suction_shift_mask, pivot=0.5)

    def gripper_grasp_processing(self,action_obj,  view=False):
        target_point = self.voxel_pc[action_obj.point_index]
        relative_pose_5 = self.gripper_poses[action_obj.point_index]
        T_d, width, distance = convert_angles_to_transformation_form(relative_pose_5, target_point)

        activate_exploration = True if np.random.rand() < exploration_probabilty else False

        T_d, distance, width, collision_intensity = local_exploration(T_d, width, distance, self.voxel_pc,
                                                                      exploration_attempts=5,
                                                                      explore_if_collision=False,
                                                                      view_if_sucess=view_masked_grasp_pose,
                                                                      explore=activate_exploration)
        action_obj.is_executable= collision_intensity == 0
        action_obj.width=width
        action_obj.transformation=T_d

        if view: vis_scene(T_d, width, npy=self.voxel_pc)

    def gripper_shift_processing(self,action_obj, view=False):
        normal = self.normals[action_obj.point_index]
        target_point = self.voxel_pc[action_obj.point_index]

        v0 = np.array([1, 0, 0])
        a = trimesh.transformations.angle_between_vectors(v0, -normal)
        b = trimesh.transformations.vector_product(v0, -normal)
        T_d = trimesh.transformations.rotation_matrix(a, b)
        T_d[:3, 3] = target_point.T

        width = np.array([0])
        action_obj.transformation=T_d
        action_obj.is_executable=True

        if view: vis_scene(T_d, width, npy=self.voxel_pc)


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
            if action_obj.use_gripper_arm:
                self.gripper_shift_processing(action_obj)
            else:
                self.suction_processing(action_obj)

    def mask_arm_occupancy(self,action_obj):
        '''mask occupied arm'''
        if action_obj.use_gripper_arm:
            self.trace_mask[:, [0, 2]] = self.trace_mask[:, [0, 2]] * 0 + 1
        else:
            self.trace_mask[:, [1, 3]] = self.trace_mask[:, [1, 3]] * 0 + 1

        '''mask occupied space'''
        approach = action_obj.transformation[0:3, 0]
        normal = approach.copy()
        normal[2] *= -1
        bounding_box_point_1 = self.voxel_pc[action_obj.point_index]
        safety_margin = 0.2
        bounding_box_point_2 = bounding_box_point_1 + normal * safety_margin
        if action_obj.use_gripper_arm:
            max_y = np.max(bounding_box_point_1[1], bounding_box_point_2[1])
            occupied_space_mask = self.voxel_pc < max_y
        else:
            mn_y = np.min(bounding_box_point_1[1], bounding_box_point_2[1])
            occupied_space_mask = self.voxel_pc > mn_y

        self.trace_mask[occupied_space_mask] = self.trace_mask[occupied_space_mask] * 0 + 1

    def pick_action(self):
        pr.title('pick action/s')
        first_action_obj=None
        second_action_obj=None

        '''first action'''
        available_actions_size=int((self.q_value>0.0).sum())
        for i in range(available_actions_size):
            point_index, action_index=self.next_action(epsilon=0.0)
            action_obj=Action(point_index, action_index)
            self.process_action(action_obj)
            if action_obj.is_executable:
                first_action_obj=action_obj
                self.mask_arm_occupancy(action_obj)
                break

        '''second action'''
        available_actions_size=int(((self.q_value*(1-self.trace_mask))>0.0).sum())
        for i in range(available_actions_size):
            point_index, action_index=self.next_action(epsilon=0.0)
            action_obj = Action(point_index, action_index)
            self.process_action(action_obj)
            if action_obj.is_executable:
                second_action_obj=action_obj
                break

        first_action_obj.target_point=self.voxel_pc[first_action_obj.point_index]
        if second_action_obj is not None: second_action_obj.target_point=self.voxel_pc[second_action_obj.point_index]

        return first_action_obj,second_action_obj


    def execute(self,first_action_obj,second_action_obj):
        pr.title('execute action')
        deploy_action( first_action_obj)
        deploy_action(second_action_obj)

        if not simulation_mode:
            if second_action_obj is not None and (first_action_obj.is_grasp and second_action_obj.is_grasp):
                '''grasp with the two arms'''
                subprocess.run(execute_both_grasp_bash)
            elif first_action_obj.is_grasp:
                '''grasp'''
                if first_action_obj.use_gripper:
                    subprocess.run(execute_gripper_grasp_bash)
                else:
                    '''suction'''
                    subprocess.run(execute_suction_grasp_bash)
            elif first_action_obj.is_shift:
                '''shift'''
                if first_action_obj.use_gripper:
                    subprocess.run(execute_gripper_shift_bash)
                else:
                    '''suction'''
                    subprocess.run(execute_suction_shift_bash)

        '''get robot feedback'''
        robot_feedback_ = wait_for_feedback()

        first_action_obj.robot_feedback(robot_feedback_)
        if second_action_obj is not None: second_action_obj.robot_feedback(robot_feedback_)

        return first_action_obj,second_action_obj

    def process_feedback(self,first_action_obj,second_action_obj, img_grasp_pre, img_suction_pre):
        pr.title('process feedback')
        if first_action_obj.robot_feedback == 'Succeed' or first_action_obj.robot_feedback == 'reset': trigger_new_perception()
        if not report_result: return

        img_suction_after, img_grasp_after = get_side_bins_images()

        '''report change if robot moves'''
        if first_action_obj.robot_feedback == 'Succeed':
            if first_action_obj.use_gripper:
                gripper_action=first_action_obj
                suction_action=second_action_obj
            else:
                gripper_action = second_action_obj
                suction_action = first_action_obj

            '''check changes in side bins'''
            gripper_action.grasp_result=check_image_similarity(img_grasp_pre, img_grasp_after)
            suction_action.grasp_result=check_image_similarity(img_suction_pre, img_suction_after)

            '''save action instance'''
            save_grasp_sample(self.rgb, self.depth, gripper_action, suction_action)
