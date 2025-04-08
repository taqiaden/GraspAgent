import numpy as np
from Configurations import config
from lib.collision_unit import grasp_collision_detection
from lib.custom_print import my_print
from lib.mesh_utils import construct_gripper_mesh_2
from lib.pc_utils import numpy_to_o3d
from visualiztion import get_arrow
import open3d as o3d

pr=my_print()

class Action():
    def __init__(self,point_index=None,action_index=None, probs=None, value=None):
        '''
        This class maintains record to all data associated to an action
        '''
        self.point_index=point_index
        self.action_index=action_index
        if action_index is None:
            self.is_shift=None
            self.is_grasp=None
            self.use_gripper_arm=None
            self.use_suction_arm=None
            self.arm_index=None
        else:
            self.is_shift = action_index > 1
            self.is_grasp = action_index <= 1
            self.use_gripper_arm = ((action_index == 0) or (action_index == 2))
            self.use_suction_arm = ((action_index == 1) or (action_index == 3))
            self.arm_index=0 if self.use_gripper_arm else 1

        '''
        ---Handover state index---
        0: for initial state of handover (delivering),
        1: for the second attempt of the handover (rotating), 
        2: for the final step of the handover (catching)
        3: drop
        '''

        self.handover_state=None

        '''rl data'''
        self.value=probs
        self.prob=value
        self.reward=None

        self.is_synchronous=None
        self.policy_index=None # 0 for stochastic policy, 1 for deterministic policy, 2 for random policy
        self.file_id=None

        '''pose'''
        self.target_point=np.full((3),np.nan)
        self.transformation=np.full((4,4),np.nan)
        self.width=None
        self.shift_end_point=np.full((3),np.nan)

        '''quality'''
        self.is_executable=None
        self.executed=None
        self.robot_feedback=None
        self.grasp_result=None
        self.shift_result=None

    @property
    def action_name(self):
        if self.is_grasp:
            return 'grasp'
        else:
            return 'shift'

    @property
    def arm_name(self):
        if self.use_gripper_arm:
            return 'gripper'
        else:
            return 'suction'

    @property
    def result(self):
        if self.is_grasp:
            return self.grasp_result
        else:
            return self.shift_result

    @property
    def real_width(self):
        if self.width is not None:
            real_width = self.width * config.width_scale
            if real_width > 25:
                return 25
            elif real_width < 0:
                return 0
            else:
                return real_width
        else:
            return None

    @property
    def scaled_width(self):
        return self.width

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
        elif self.handover_state is not None:
            return 'handover'

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