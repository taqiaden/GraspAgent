import math
import subprocess
import numpy as np
import torch
import trimesh
from colorama import Fore
import open3d as o3d
from Configurations.ENV_boundaries import bin_center
from Configurations.config import distance_scope, gripper_width_during_shift
from Configurations.run_config import simulation_mode, \
    suction_factor, gripper_factor, report_result, use_gripper, use_suction, activate_grasp, activate_shift
from Online_data_audit.process_feedback import save_grasp_sample
from check_points.check_point_conventions import GANWrapper, ModelWrapper
from lib.ROS_communication import wait_for_feedback, deploy_action
from lib.bbox import convert_angles_to_transformation_form
from lib.collision_unit import grasp_collision_detection
from lib.custom_print import my_print
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from lib.image_utils import check_image_similarity
from lib.mesh_utils import construct_gripper_mesh_2
from lib.pc_utils import numpy_to_o3d
from lib.report_utils import progress_indicator
from models.action_net import ActionNet, action_module_key
from models.scope_net import scope_net_vanilla, gripper_scope_module_key, suction_scope_module_key
from models.value_net import ValueNet, value_module_key
from pose_object import vectors_to_ratio_metrics
from process_perception import get_side_bins_images, trigger_new_perception
from registration import camera
from training.learning_objectives.shift_affordnace import shift_execution_length
from visualiztion import view_npy_open3d, vis_scene, get_arrow

execute_suction_grasp_bash = './bash/run_robot_suction.sh'
execute_gripper_grasp_bash = './bash/run_robot_grasp.sh'
execute_suction_shift_bash = './bash/run_robot_suction_shift.sh'
execute_gripper_shift_bash = './bash/run_robot_gripper_shift.sh'
execute_both_grasp_bash = './bash/run_robot_grasp_and_suction.sh'

pr=my_print()

class Action():
    def __init__(self,point_index=None,action_index=None):
        self.point_index=point_index
        self.action_index=action_index
        if action_index is None:
            self.is_shift=None
            self.is_grasp=None
            self.use_gripper_arm=None
            self.use_suction_arm=None
        else:
            self.is_shift = action_index > 1
            self.is_grasp = action_index <= 1
            self.use_gripper_arm = ((action_index == 0) or (action_index == 2))
            self.use_suction_arm = ((action_index == 1) or (action_index == 3))

        self.target_point=None
        
        self.transformation=None
        self.width=None
        
        self.is_executable=None

        self.robot_feedback=None

        self.grasp_result=None
        self.shift_result=None

        self.shift_end_point=None


    @property
    def is_valid(self):
        return self.use_gripper_arm is not None or self.use_suction_arm is not None
    @property
    def arm_name(self):
        if self.use_gripper_arm:
            return 'gripper'
        elif self.use_suction_arm:
            return 'suction'

    @property
    def action_name(self):
        if self.is_grasp:
            return 'grasp'
        elif self.is_shift:
            return 'shift'

    def check_collision(self,point_cloud):
        if self.is_valid and self.is_grasp and self.use_gripper_arm:
            return grasp_collision_detection(self.transformation, self.width, point_cloud, visualize=False) > 0
        else:
            return None

    def get_gripper_mesh(self,color=None):
        if self.is_valid and self.use_gripper_arm:
            mesh = construct_gripper_mesh_2(self.width, self.transformation)
            mesh.paint_uniform_color([0.5, 0.9, 0.5]) if color is None else mesh.paint_uniform_color(color)
            return mesh
        else:
            return None

    @property
    def approach(self):
        if self.transformation is not None:
            return self.transformation[0:3, 0]
        else:
            return None

    @property
    def normal(self):
        if self.transformation is not None:
            normal = self.approach*-1
            return normal
        else:
            return None

    def get_approach_mesh(self):
        if self.is_valid and self.transformation is not None:
            arrow_emerge_point=self.target_point-self.approach*0.05
            arrow_mesh=get_arrow(end=self.target_point,origin=arrow_emerge_point,scale=1.3)
            return arrow_mesh
        else:
            return None

    def pose_mesh(self):
        if self.is_valid:
            if self.use_gripper_arm:
                if self.is_grasp:
                    return self.get_gripper_mesh()
                else:
                    return self.get_gripper_mesh(color=[0.5, 0.5, 0.5])
            else:
                arrow_mesh=self.get_approach_mesh()
                if self.is_grasp:
                    arrow_mesh.paint_uniform_color([0.5, 0.9, 0.5])
                else:
                    arrow_mesh.paint_uniform_color([0.5, 0.5, 0.5])
                return arrow_mesh
        else:
            return None

    def view(self,point_clouds,mask=None):
        scene_list = []
        pose_mesh=self.pose_mesh()
        # o3d_line(start, end2, colors_=[0, 0.5, 0])
        if pose_mesh is not None:
            scene_list.append(pose_mesh)
            masked_colors = np.ones_like(point_clouds) * [0.5, 0.9, 0.5]
            if mask is not None:
                masked_colors[mask]=(masked_colors[mask]*0)+[0.9, 0.5, 0.5]
            pcd = numpy_to_o3d(pc=point_clouds, color=masked_colors)
            scene_list.append(pcd)
            o3d.visualization.draw_geometries(scene_list)

    def print(self):
        if  self.is_valid:
            pr.print('Action details:')
            pr.step_f()
            pr.print(f'{self.action_name} using {self.arm_name} arm')
            if self.target_point is not None:
                pr.print(f'target point {self.target_point}')

            if self.robot_feedback is not None:
                pr.print(f'Robot feedback message : {self.robot_feedback}')

            if self.grasp_result is not None:
                pr.print(f'Grasp result : {self.grasp_result}')

            if self.shift_result is not None:
                pr.print(f'Shift result : {self.shift_result}')

            pr.step_b()

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
    colors=masked_color(voxel_pc, score, pivot=0.5)
    view_npy_open3d(voxel_pc, color=colors)

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
        self.value_net = None
        self.suction_arm_reachability_net = None
        self.gripper_arm_reachability_net = None

        '''modalities'''
        self.point_clouds = None
        self.depth=None
        self.rgb=None

        '''dense records'''
        self.gripper_poses_5=None
        self.gripper_poses_7=None
        self.gripper_grasp_mask=None
        self.suction_grasp_mask=None
        self.gripper_shift_mask=None
        self.suction_shift_mask=None
        self.voxel_pc=None
        self.normals=None
        self.q_value=None
        self.biased_q_value=None
        self.shift_end_points=None
        self.valid_actions_mask=None
        self.n_grasps=0
        self.n_shifts=0
        self.first_action_mask=None

    def clear(self):
        self.gripper_poses_5=None
        self.gripper_poses_7=None
        self.gripper_grasp_mask=None
        self.suction_grasp_mask=None
        self.gripper_shift_mask=None
        self.suction_shift_mask=None
        self.voxel_pc=None
        self.normals=None
        self.q_value=None
        self.biased_q_value=None
        self.shift_end_points=None
        self.valid_actions_mask=None
        self.n_grasps=0
        self.n_shifts=0
        self.first_action_mask=None


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
        if self.shift_end_points is None:
            self.shift_end_points = get_shift_end_points(positions)
        gripper_scope_b=self.gripper_arm_reachability_net(torch.cat([self.shift_end_points, gripper_approach], dim=-1)).squeeze()#.squeeze().detach().cpu().numpy()
        result = torch.stack([gripper_scope_a, gripper_scope_b], dim=-1)
        result, _ = torch.min(result, dim=-1)
        return result

    def next_action(self, epsilon=0.0):
        masked_q_value=self.biased_q_value*self.valid_actions_mask

        if np.random.rand()>epsilon:
            flattened_index=torch.argmax(masked_q_value).item()
            point_index = math.floor(flattened_index / 4)
            action_index = flattened_index - int(4 * point_index)

            max_val=masked_q_value[point_index,action_index]
            if max_val==0: return None,None

        else:
            available_indexes=torch.nonzero(masked_q_value)
            if available_indexes.shape[0]==0: return None,None
            random_pick=np.random.random_integers(0,available_indexes.shape[0]-1)
            point_index = available_indexes[random_pick][0]
            action_index = available_indexes[random_pick][1]

        self.valid_actions_mask[point_index,action_index]=0
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
                               *(griper_grasp_score*gripper_factor>0.5)* int(use_gripper)* int(activate_grasp))>0.5
        self.suction_grasp_mask=((background_class<0.5)*(suction_grasp_scope>0.5)
                               *(suction_seal_classifier>0.5)
                               *(suction_grasp_score*suction_factor>0.5)* int(use_suction)* int(activate_grasp))>0.5

        '''shift reachability'''
        gripper_shift_scope=self.get_gripper_shift_reachability( positions,self.normals)
        suction_shift_scope=self.get_suction_shift_reachability(positions,self.normals)

        '''shift actions'''
        self.gripper_shift_mask=((shift_appealing>0.5)*(gripper_shift_scope>0.5)* int(use_gripper)* int(activate_shift))>0.5#*(shift_affordance_classifier>0.5)
        self.suction_shift_mask=((shift_appealing>0.5)*(suction_shift_scope>0.5)* int(use_suction)* int(activate_shift))>0.5#*(shift_affordance_classifier>0.5)

        '''gripper pose convention'''
        self.gripper_poses_7=poses
        self.gripper_poses_5 = vectors_to_ratio_metrics(poses.clone())

        '''initiate random policy'''
        # self.q_value=torch.rand_like(q_value.squeeze().permute(1,2,0)[mask])
        self.q_value=q_value.squeeze().permute(1,2,0)[mask]
        self.biased_q_value=self.q_value.clone()
        min_q_value=torch.min(self.biased_q_value).item()
        self.biased_q_value+=(1-min(min_q_value,0.))

        self.valid_actions_mask=torch.zeros_like(self.q_value)

        '''initialize valid actions mask'''
        self.valid_actions_mask[:,0][self.gripper_grasp_mask]+=1
        self.valid_actions_mask[:,1][self.suction_grasp_mask]+=1
        self.valid_actions_mask[:,2][self.gripper_shift_mask]+=1
        self.valid_actions_mask[:,3][self.suction_shift_mask]+=1

        '''count available actions'''
        self.n_grasps=torch.count_nonzero(self.valid_actions_mask[:,0:2])
        self.n_shifts=torch.count_nonzero(self.valid_actions_mask[:,2:4])

        '''to numpy'''
        self.normals=self.normals.cpu().numpy()

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
        '''mask occupied arm'''
        if action_obj.use_gripper_arm:
            self.valid_actions_mask[:, [0, 2]] *=  0 
        else:
            self.valid_actions_mask[:, [1, 3]] *=  0 

        '''mask occupied space'''
        arm_knee_margin = 0.2
        normal=action_obj.normal
        action_obj.target_point=self.voxel_pc[action_obj.point_index]
        arm_knee_extreme=(action_obj.target_point + normal * arm_knee_margin)[1]
        minimum_safety_margin=0.05
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

        self.valid_actions_mask[occupancy_mask] *=  0

    def view_valid_actions_mask(self):
        four_pc_stack = np.stack([self.voxel_pc, self.voxel_pc, self.voxel_pc, self.voxel_pc])
        four_pc_stack[1, :, 0] += 0.5
        four_pc_stack[2, :, 0] += 1.0
        four_pc_stack[3, :, 0] += 1.5

        colors = np.ones_like(four_pc_stack) * [0.5, 0.9, 0.5]

        for i in range(4):
            mask_i = (self.valid_actions_mask[:, i] > 0.5).cpu().numpy()
            (colors[i])[~mask_i] *= 0.
            (colors[i])[~mask_i] += [0.9, 0.9, 0.9]
        four_pc_stack = np.concatenate([four_pc_stack[0], four_pc_stack[1], four_pc_stack[2], four_pc_stack[3]], axis=0)
        colors = np.concatenate([colors[0], colors[1], colors[2], colors[3]], axis=0)
        view_npy_open3d(four_pc_stack, color=colors)

    def pick_action(self):
        pr.title('pick action/s')
        first_action_obj=Action()
        second_action_obj=Action()
        self.first_action_mask=None


        '''first action'''
        available_actions_size=int((self.valid_actions_mask>0.0).sum())
        for i in range(available_actions_size):
            point_index, action_index=self.next_action(epsilon=0.0)
            action_obj=Action(point_index, action_index)
            self.process_action(action_obj)
            if action_obj.is_executable:
                first_action_obj=action_obj
                if action_obj.is_grasp:
                    self.mask_arm_occupancy(action_obj)
                break

        self.valid_actions_mask[:,2:]*=0 # simultaneous operation of both arms are for grasp actions only
        if first_action_obj.is_shift:self.valid_actions_mask*=0

        first_action_obj.target_point=self.voxel_pc[first_action_obj.point_index]
        first_action_obj.print()
        # first_action_obj.view(self.voxel_pc)
        # self.view_valid_actions_mask()

        '''second action'''
        available_actions_size=int((self.valid_actions_mask>0.0).sum())
        for i in range(available_actions_size):
            point_index, action_index=self.next_action(epsilon=0.0)
            action_obj = Action(point_index, action_index)
            self.process_action(action_obj)
            if action_obj.is_executable:
                second_action_obj=action_obj
                break

        if second_action_obj is not None: second_action_obj.target_point=self.voxel_pc[second_action_obj.point_index]

        second_action_obj.print()

        return first_action_obj,second_action_obj

    def execute(self,first_action_obj,second_action_obj):
        pr.title('execute action')
        pr.print('Deploy action commands')
        if not first_action_obj.is_valid: return first_action_obj,second_action_obj
        deploy_action( first_action_obj)
        if  second_action_obj.is_valid: deploy_action(second_action_obj)

        if not simulation_mode:
            pr.print('Run robot')
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

        '''get robot feedback'''
        robot_feedback_ = wait_for_feedback()

        first_action_obj.robot_feedback=robot_feedback_
        second_action_obj.robot_feedback=robot_feedback_

        return first_action_obj,second_action_obj

    def process_feedback(self,first_action_obj,second_action_obj, img_grasp_pre, img_suction_pre):
        pr.title('process feedback')
        if first_action_obj.robot_feedback == 'Succeed' or first_action_obj.robot_feedback == 'reset': trigger_new_perception()
        if not report_result: return

        img_suction_after, img_grasp_after = get_side_bins_images()

        '''report change if robot moves'''
        if first_action_obj.robot_feedback == 'Succeed':
            if first_action_obj.use_gripper_arm:
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


'''conventions'''
# normal is a vector emerge out of the surface
# approach direction is a vector pointing to the surface
# approach = -1 * normal
# the gripper sampler first three parameters are approach[0] and approach [1] and -1* approach[2]
# the suction sampler outputs the normal direction
# T_0 refers to a gripper head transformation matrix with zero penetration while T_d embeds distance term

