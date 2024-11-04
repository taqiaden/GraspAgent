import numpy as np
from colorama import Fore
from torch.utils import data
from lib.depth_map import  point_clouds_to_depth
from registration import camera, transform_to_camera_frame
from lib.dataset_utils import training_data, online_data
from lib.report_utils import progress_indicator as pi
import random


training_data=training_data()
online_data=online_data()

# def load_training_buffer(size):
#     file_indexes = online_data.get_indexes()
#     random.shuffle(file_indexes)
#
#     progress_indicator=pi(f'load {size} samples for training',size)
#     counter=0
#     for i,target_file_index in enumerate(file_indexes):
#         '''get data'''
#         try:
#             # depth=online_data.load_depth(target_file_index)
#             pc=online_data.point_clouds.load_as_numpy(target_file_index)
#             transformed_pc = transform_to_camera_frame(pc)
#             depth=point_clouds_to_depth(transformed_pc, camera)
#         except Exception as e:
#             print(Fore.RED, str(e),Fore.RESET)
#             continue
#
#         '''save to buffer'''
#         training_data.depth.save_as_numpy(depth,target_file_index)
#
#         '''update counter'''
#         counter+=1
#         progress_indicator.step(counter)
#         if counter >= size: break


class suction_sampler_dataset(data.Dataset):
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
