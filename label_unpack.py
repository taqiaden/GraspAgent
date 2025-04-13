import numpy as np
import torch

from Configurations import config
from lib.collision_unit import grasp_collision_detection
from lib.depth_map import transform_to_camera_frame, point_clouds_to_depth, get_pixel_index, depth_to_point_clouds, \
    pixel_to_point
from pose_object import encode_gripper_pose_npy, pose_7_to_transformation
from registration import camera


class LabelObj():
    def __init__(self,label=None,point_clouds=None,depth=None,RGB=None):


        '''modalities'''
        self.point_clouds=point_clouds
        self.depth=depth
        self.RGB=RGB

        '''label data'''
        if label is not None:
            self.is_suction = label[23]
            self.is_gripper = label[4]
            self.success = label[3]
            self.failure = label[3] == 0
            self.target_point=label[:3]

            '''gripper parameters'''
            if self.is_gripper :
                self.T_d=label[5:21].copy().reshape(-1, 4)
                self.width=label[21] / config.width_scale
                self.distance=label[22]

            '''suction parameters'''
            if self.is_suction:
                self.normal = label[24:27]
            else:
                self.normal=np.array([0.0,0.0,0.0])


    def get_depth(self,point_clouds=None):
        if self.depth is not None and point_clouds is None:
            return self.depth
        else:
            if point_clouds is not None: self.point_clouds=point_clouds
            transformed_pc = transform_to_camera_frame(self.point_clouds)
            self.depth = point_clouds_to_depth(transformed_pc, camera)
            return self.depth

    def get_point_clouds_from_depth(self,depth=None):
        if depth is not None:
            self.depth=depth
        point_clouds, mask = depth_to_point_clouds(self.depth, camera)
        point_clouds = transform_to_camera_frame(point_clouds, reverse=True)
        return point_clouds

    def get_gripper_pose_7(self):
        if self.is_gripper:
            R = self.T_d[0:3, 0:3]
            pose_7 = encode_gripper_pose_npy(self.distance, self.width, R)
        else:
            pose_7=np.array([0.0]*7)
        return pose_7

    def get_pixel_index(self):
        pixel_index=get_pixel_index( camera, self.target_point)
        return pixel_index

    def check_collision(self,depth,visualize=False):
        assert self.is_gripper==True
        pc = self.get_point_clouds_from_depth(depth=depth)
        pixel_index = self.get_pixel_index()

        depth_value = depth[pixel_index[0], pixel_index[1]]

        '''get pose parameters'''
        target_point = pixel_to_point(pixel_index, depth_value, camera)
        target_point = transform_to_camera_frame(target_point[None, :], reverse=True)[0]

        target_pose = self.get_gripper_pose_7()
        T_d, width, distance = pose_7_to_transformation(torch.from_numpy(target_pose), target_point)

        collision_intensity = grasp_collision_detection(T_d, width, pc, visualize=visualize)

        return collision_intensity>0

    # tabulated data:
    # [0:3]: gripper_target_point
    # [3:6]: suction_target_point
    # [6:22]: gripper_transformation
    # [22:38]: suction_transformation
    # [38]: gripper_width
    # [39:42] gripper shift end position
    # [42:45] suction shift end position
    # [45]: gripper_action_index
    # [46]: suction_action_index
    # [47]: gripper_result
    # [48]: suction_result
    # [49]: run_sequence

class GripperParameters:
    def __init__(self,label):
        self.used=label[45]!=0
        self.target_point = np.float64(label[0:3])
        self.transformation = np.float64(label[6:22].copy().reshape(-1, 4))
        self.width =-1 if label[38] is None else np.float64(label[38] / config.width_scale)
        self.is_grasp=1 if label[45]==1 else 0
        self.is_shift=1 if label[45]==2 else 0
        if self.is_shift:
            self.shift_end_location=np.float64(label[39:42])
        self.result=label[47] if label[47] is not None else -1
        self.handover_angle=label[51]
    @property
    def approach(self):
        return self.transformation[0:3, 0]

    @property
    def transition(self):
        return self.transformation[0:3, 3]

    @property
    def distance(self):
        return ((self.transition-self.target_point)/self.approach).mean()

    @property
    def pose_7(self):
        if self.used:
            R = self.transformation[0:3, 0:3]
            pose_7 = encode_gripper_pose_npy(self.distance, self.width, R)
            return pose_7
        else:
            return np.array([0]*7)

    def pixel_index(self):
        if self.used:
            pixel_index=get_pixel_index( camera, self.target_point)
            return pixel_index
        else: return np.array([0,0])

    def check_collision(self,depth,pc,visualize=False):
        pixel_index = self.pixel_index(depth)
        depth_value = depth[pixel_index[0], pixel_index[1]]

        '''get pose parameters'''
        target_point = pixel_to_point(pixel_index, depth_value, camera)
        target_point = transform_to_camera_frame(target_point[None, :], reverse=True)[0]

        target_pose = self.pose_7()
        T_d, width, distance = pose_7_to_transformation(torch.from_numpy(target_pose), target_point)

        collision_intensity = grasp_collision_detection(T_d, width, pc, visualize=visualize)

        return collision_intensity>0

class SuctionParameters:
    def __init__(self,label):
        self.used=label[46]!=0
        self.target_point = np.float64(label[3:6])
        self.transformation = np.float64(label[22:38].copy().reshape(-1, 4))
        self.is_grasp=1 if label[46]==1 else 0
        self.is_shift=1 if label[46]==2 else 0
        if self.is_shift:
            self.shift_end_location=np.float64(label[42:45])
        self.result=label[48] if label[48] is not None else -1
        self.handover_angle=label[52]

    @property
    def approach(self):
        return self.transformation[0:3, 0]

    @property
    def normal(self):
        if self.used:
            return -self.approach
        else:
            return np.array([0]*3)

    def pixel_index(self):
        if self.used:
            pixel_index=get_pixel_index( camera, self.target_point)
            return pixel_index
        else: return np.array([0,0])

class LabelObj2:
    def __init__(self,label=None,point_clouds=None,depth=None,RGB=None):
        '''modalities'''
        self.point_clouds=point_clouds
        self.depth=depth
        self.RGB=RGB

        '''label data'''
        if label is not None:

            '''gripper parameters'''
            self.gripper=GripperParameters(label)

            '''suction parameters'''
            self.suction=SuctionParameters(label)

            self.step_number=label[49]
            self.is_end_of_task=label[50]


    def get_depth(self,point_clouds=None):
        if self.depth is not None and point_clouds is None:
            return self.depth
        else:
            if point_clouds is not None: self.point_clouds=point_clouds
            transformed_pc = transform_to_camera_frame(self.point_clouds)
            self.depth = point_clouds_to_depth(transformed_pc, camera)
            return self.depth

    def get_point_clouds_from_depth(self,depth=None):
        if depth is not None:
            self.depth=depth
        point_clouds, mask = depth_to_point_clouds(self.depth, camera)
        point_clouds = transform_to_camera_frame(point_clouds, reverse=True)
        return point_clouds





