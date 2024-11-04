import numpy as np
from colorama import Fore
from torch.utils import data
from Online_data_audit.sample_training_buffer import get_selection_probabilty
from label_unpack import LabelObj
from lib.dataset_utils import  online_data, data_pool
from lib.report_utils import progress_indicator as pi
import random

training_buffer_dir='dataset/BS_training_buffer/'

class BSBuffer(data_pool):
    def __init__(self):
        super(BSBuffer,self).__init__(dir=training_buffer_dir,dataset_name='training')
        self.main_modality=self.depth

online_data=online_data()

# def load_training_buffer(size):
#     training_buffer = BSBuffer()
#
#     file_indexes = online_data.get_indexes()
#     random.shuffle(file_indexes)
#     selection_p = get_selection_probabilty(file_indexes)
#
#     progress_indicator=pi(f'load {size} samples for training ',size)
#     counter=0
#     for i,target_file_index in enumerate(file_indexes):
#         if np.random.rand() > selection_p[i]: continue
#         '''get data'''
#         try:
#             label_obj = LabelObj()
#
#             '''load depth map'''
#             pc=online_data.point_clouds.load_as_numpy(target_file_index)
#             depth=label_obj.get_depth(point_clouds=pc)
#
#         except Exception as e:
#             print(Fore.RED, str(e),Fore.RESET)
#             continue
#
#         '''save to buffer'''
#         training_buffer.depth.save_as_numpy(depth,target_file_index)
#
#         '''update counters'''
#         counter+=1
#         progress_indicator.step(counter)
#         if counter >= size:
#             break

class BackgroundSegDataset(data.Dataset):
    def __init__(self, data_pool):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = self.data_pool.depth.get_indexes()

    def __getitem__(self, idx):
        target_index = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(target_index)
        return depth[np.newaxis,:,:],target_index

    def __len__(self):
        return len(self.files_indexes)
