import os
import random
import numpy as np
import smbclient
from colorama import Fore
from torch.utils import data

from lib.pc_utils import point_index
from dataset.load_test_data import  random_sampling_augmentation

from Configurations import config

from lib.dataset_utils import data as d
class suction_dataset(data.Dataset):
    def __init__(self, num_points=config.num_points, path='dataset/realtime_dataset/', shuffle=False):
        super().__init__()
        self.num_points, self.path = num_points, path
        self.is_local_dir = os.path.exists(path)
        self.dataset = d(path, is_local=self.is_local_dir)
        self.label_filenames = self.dataset.get_label_names()
        if shuffle: random.shuffle(self.label_filenames)

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

        # tabulated data:
        # [0:3]: center_point
        # [3]: score
        # [4]: grasp
        # [5:21]: rotation_matrix
        # [21]: width
        # [22]: distance
        # [23]: suction
        # [24:27]: pred_normal
        assert label[4]==0 and label[23]==1,f'{label[4]},   {label[23]}'
        score = label[3]
        normal_label = label[24:27]

        center_point = label[:3]

        if point_data.shape[0] == self.num_points:
            point_data_choice = point_data
            index = point_index(point_data[:,0:3], center_point)
            assert index is not None
        else:
            point_data_choice, index = random_sampling_augmentation(center_point, point_data, self.num_points)

        pc = point_data_choice.copy()

        normal_label = np.asarray(normal_label, dtype=np.float32)
        score = np.asarray(score, dtype=np.float32)

        # if  score==1 :
        #     from visualiztion import visualize_suction_pose
        #     suction_xyz, pre_grasp_mat, end_effecter_mat, suction_pose, T, pred_approch_vector=get_suction_pose_(center_point.reshape(1,3), normal_label)
        #     visualize_suction_pose(center_point, normal_label, T, end_effecter_mat,npy=pc)


        return pc,normal_label,score,index


    def __len__(self):
        if self.is_local_dir:
            return len(os.listdir(self.path + 'point_cloud/'))
        else:
            return len(smbclient.listdir(self.path + 'point_cloud/'))
