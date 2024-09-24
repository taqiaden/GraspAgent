import numpy as np
from colorama import Fore
from torch.utils import data

from Online_data_audit.sample_training_buffer import get_selection_probabilty
from lib.depth_map import point_clouds_to_depth, get_pixel_index
from registration import camera, transform_to_camera_frame
from lib.dataset_utils import training_data, online_data
from lib.report_utils import progress_indicator as pi
import random

training_data=training_data()
online_data=online_data()

force_balanced_data=True

def load_training_buffer_kd(size):
    file_indexes = online_data.get_indexes()
    random.shuffle(file_indexes)

    progress_indicator=pi(f'load {size} samples for training ',size)
    counter=0
    for i,target_file_index in enumerate(file_indexes):
        '''get data'''
        try:
            # depth=online_data.load_depth(target_file_index)
            pc=online_data.point_clouds.load_as_numpy(target_file_index)
            transformed_pc = transform_to_camera_frame(pc)
            depth=point_clouds_to_depth(transformed_pc, camera)
        except Exception as e:
            print(Fore.RED, str(e),Fore.RESET)
            continue

        '''save to buffer'''
        training_data.depth.save_as_numpy(depth,target_file_index)


        '''update counter'''
        counter+=1
        progress_indicator.step(counter)
        if counter >= size: break

def load_training_buffer(size):
    file_indexes = online_data.get_indexes()
    random.shuffle(file_indexes)
    selection_p = get_selection_probabilty(file_indexes)

    progress_indicator=pi(f'load {size} samples for training ',size)
    counter=0
    positive_samples=0
    negative_samples=0

    for i,target_file_index in enumerate(file_indexes):
        if np.random.rand() > selection_p[i]: continue
        '''get data'''
        try:
            label = online_data.label.load_as_numpy(target_file_index)
            '''selection rules'''
            if label[4] == 1: continue
            if force_balanced_data:
                if label[3] == 1 and (1+negative_samples)/(1+positive_samples)<1:
                    continue
                elif label[3] == 0 and (1+positive_samples)/(1+negative_samples)<1:
                    continue
            # depth=online_data.load_depth(target_file_index)

            '''load depth map'''
            pc=online_data.point_clouds.load_as_numpy(target_file_index)
            transformed_pc = transform_to_camera_frame(pc)
            depth=point_clouds_to_depth(transformed_pc, camera)

        except Exception as e:
            print(Fore.RED, str(e),Fore.RESET)
            continue

        '''save to buffer'''
        training_data.depth.save_as_numpy(depth,target_file_index)
        training_data.label.save_as_numpy(label,target_file_index)

        '''update counters'''
        if label[3]==1:positive_samples+=1
        else: negative_samples+=1
        counter+=1
        progress_indicator.step(counter)
        if counter >= size:
            print(f'Sampled buffer contains {positive_samples} positive samples and {negative_samples} negative samples')
            break

class suction_quality_dataset_kd(data.Dataset):
    def __init__(self, data_pool):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = self.data_pool.depth.get_indexes()

    def __getitem__(self, idx):
        target_index = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(target_index)
        return depth[np.newaxis,:,:]

    def __len__(self):
        return len(self.files_indexes)

class suction_quality_dataset(data.Dataset):
    def __init__(self, data_pool):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = self.data_pool.depth.get_indexes()

    def __getitem__(self, idx):
        target_index = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(target_index)
        label = self.data_pool.label.load_as_numpy(target_index)
        assert label[4] == 0 and label[23] == 1, f'{label[4]},   {label[23]}'
        score = label[3]
        normal= label[24:27]

        target_point = label[:3]
        pixel_index = get_pixel_index(depth, camera, target_point)

        return depth[np.newaxis,:,:],normal,score,pixel_index

    def __len__(self):
        return len(self.files_indexes)


