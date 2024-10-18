from Configurations import config
from lib.depth_map import transform_to_camera_frame, point_clouds_to_depth, get_pixel_index, depth_to_point_clouds
from pose_object import encode_gripper_pose_npy
from registration import camera


class LabelObj():
    def __init__(self,label,point_clouds=None,depth=None,RGB=None):
        '''general info'''
        self.is_suction=label[23]
        self.is_gripper=label[4]
        self.success=label[3]
        self.failure = label[3]==0

        '''modalities'''
        self.point_clouds=point_clouds
        self.depth=depth
        self.RGB=RGB

        '''target point'''
        self.target_point=label[:3]

        '''gripper parameters'''
        if self.is_gripper:
            self.T_d=label[5:21].copy().reshape(-1, 4)
            self.width=label[21] / config.width_scale
            self.distance=label[22]

        '''suction parameters'''
        if self.is_suction:
            self.normal = label[24:27]


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
        point_clouds, mask = depth_to_point_clouds(depth, camera)
        point_clouds = transform_to_camera_frame(point_clouds, reverse=True)
        return point_clouds

    def get_gripper_pose_7(self):
        R = self.T_d[0:3, 0:3]
        pose_7 = encode_gripper_pose_npy(self.distance, self.width, R)
        return pose_7

    def get_pixel_index(self):
        pixel_index=get_pixel_index(self.depth, camera, self.target_point)
        return pixel_index
