import os
import random
import numpy as np
import smbclient
from colorama import Fore
from torch.utils import data
from Configurations import config
from lib.pc_utils import point_index
from dataset.load_test_data import  random_sampling_augmentation

from lib.dataset_utils import data as d

from pose_object import  encode_gripper_pose_npy

class gripper_dataset(data.Dataset):
    def __init__(self, num_points=config.num_points, path='dataset/realtime_dataset/', shuffle=False):
        super().__init__()
        self.num_points, self.path = num_points, path
        self.is_local_dir = os.path.exists(path)
        self.dataset = d(path, is_local=self.is_local_dir)
        self.label_filenames = self.dataset.get_label_names()
        if shuffle: random.shuffle(self.label_filenames)
        self.collision_threshold=0.01

    def _load_data_file(self, idx):
        try:
            label_filename_ = self.label_filenames[idx]
            point_data, label = self.dataset.load_labeled_data(label_filename=label_filename_)
        except Exception as e:
            print(Fore.RED, 'Warning: ', str(e), f', File label name {self.label_filenames[idx]}', Fore.RESET)
            label_filename_ = self.label_filenames[idx + 1]
            point_data, label = self.dataset.load_labeled_data(label_filename=label_filename_)
        return point_data, label

    def __getitem__(self, idx):
        point_data, label = self._load_data_file(idx)
        center_point = label[:3]

        # tabulated data:
        # [0:3]: center_point
        # [3]: score
        # [4]: grasp
        # [5:21]: rotation_matrix
        # [21]: width
        # [22]: distance
        # [23]: suction
        # [24:27]: pred_normal
        assert label[4]==1 and label[23]==0,f'{label[4]},   {label[23]}'
        score = label[3]
        distance = label[22]
        width = label[21] / config.width_scale

        if point_data.shape[0] == self.num_points:
            point_data_choice = point_data
            index = point_index(point_data[:,0:3], center_point)
            assert index is not None
        else:
            point_data_choice, index = random_sampling_augmentation(center_point, point_data, self.num_points)

        new_point_data = point_data_choice
        transformation = label[5:21].copy().reshape(-1, 4)
        transformation[0:3, 3] = label[:3] + transformation[0:3, 0] * distance # update the center point of the transformation

        pc = new_point_data.copy()

        score = np.asarray(score, dtype=np.float32)

        rotation = transformation[ 0:3, 0:3]

        pose_7 = encode_gripper_pose_npy(distance, width, rotation)

        return pc, pose_7, score, index


    def __len__(self):
        if self.is_local_dir:
            return len(os.listdir(self.path + 'point_cloud/'))
        else:
            return len(smbclient.listdir(self.path + 'point_cloud/'))