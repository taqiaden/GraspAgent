import numpy as np
import torch
from colorama import Fore
from torch.utils import data

from Configurations import config
from lib.collision_unit import grasp_collision_detection
from lib.depth_map import point_clouds_to_depth, get_pixel_index, depth_to_point_clouds
from pose_object import encode_gripper_pose_npy, pose_7_to_pose_good_grasp
from registration import camera, transform_to_camera_frame
from lib.dataset_utils import training_data, online_data
from lib.report_utils import progress_indicator as pi
import random


training_data=training_data()
online_data=online_data()

def load_training_buffer(size):
    file_indexes = online_data.get_indexes()
    random.shuffle(file_indexes)

    progress_indicator=pi(f'load {size} samples for training ',size)

    counter=0
    for i,target_file_index in enumerate(file_indexes):
        '''get data'''
        try:
            # depth=online_data.load_depth(target_file_index)
            label = online_data.label.load_as_numpy(target_file_index)
            '''selection rules'''
            if label[23]==1:    continue
            if label[3] == 0: continue

            pc = online_data.point_clouds.load_as_numpy(target_file_index)

            '''load depth map'''
            transformed_pc = transform_to_camera_frame(pc)
            depth=point_clouds_to_depth(transformed_pc, camera)

            '''check collision'''
            pc, mask = depth_to_point_clouds(depth, camera)
            pc = transform_to_camera_frame(pc, reverse=True)
            pose_7 = process_label_for_gripper(label)
            target_point = label[:3]
            pose_7=torch.from_numpy(pose_7).squeeze()
            target_point=torch.from_numpy(target_point)
            pose_good_grasp = pose_7_to_pose_good_grasp(pose_7, target_point)
            collision_intensity = grasp_collision_detection(pose_good_grasp, pc, visualize=False)
            if collision_intensity>0:
                continue


        except Exception as e:
            print(Fore.RED, str(e),Fore.RESET)
            continue

        '''save to buffer'''
        training_data.depth.save_as_numpy(depth,target_file_index)
        training_data.label.save_as_numpy(label,target_file_index)

        '''update counter'''
        counter+=1
        progress_indicator.step(counter)
        if counter >= size:
            break

def process_label_for_gripper(label):
    assert label[4] == 1 and label[23] == 0, f'{label[4]},   {label[23]}'
    distance = label[22]
    width = label[21] / config.width_scale
    transformation = label[5:21].copy().reshape(-1, 4)
    target_point = label[:3]
    transformation[0:3, 3] = target_point + transformation[0:3,0] * distance  # update the center point of the transformation
    rotation = transformation[0:3, 0:3]
    pose_7 = encode_gripper_pose_npy(distance, width, rotation)
    return pose_7

class Grasp_GAN_dataset(data.Dataset):
    def __init__(self, data_pool):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = self.data_pool.depth.get_indexes()

    def __getitem__(self, idx):
        target_index = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(target_index)
        label = self.data_pool.label.load_as_numpy(target_index)
        assert label[4]==1 and label[23]==0,f'{label[4]},   {label[23]}'
        pose_7=process_label_for_gripper(label)
        target_point = label[:3]
        pixel_index=get_pixel_index(depth, camera, target_point)
        return depth[np.newaxis,:,:],pose_7, pixel_index

    def __len__(self):
        return len(self.files_indexes)

