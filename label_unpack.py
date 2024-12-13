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
        pixel_index=get_pixel_index(self.depth, camera, self.target_point)
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
