import numpy as np
from colorama import Fore
from torch.utils import data
from label_unpack import LabelObj
from lib.collision_unit import  grasp_collision_detection
from lib.dataset_utils import training_data, online_data
from lib.report_utils import progress_indicator as pi
import random

training_data=training_data()
online_data=online_data()

# def load_training_buffer(size):
#     file_indexes = online_data.get_indexes()
#     random.shuffle(file_indexes)
#
#     progress_indicator=pi(f'load {size} samples for training ',size)
#
#     counter=0
#     for i,target_file_index in enumerate(file_indexes):
#         '''get data'''
#         try:
#             # depth=online_data.load_depth(target_file_index)
#             label = online_data.label.load_as_numpy(target_file_index)
#             label_obj = LabelObj(label=label)
#
#             '''selection rules'''
#             if label_obj.is_suction:    continue
#             if label_obj.failure: continue
#
#             pc = online_data.point_clouds.load_as_numpy(target_file_index)
#
#             '''load depth map'''
#             depth=label_obj.get_depth(point_clouds=pc)
#
#             '''check collision'''
#             pc=label_obj.get_point_clouds_from_depth(depth=depth)
#             collision_intensity = grasp_collision_detection(label_obj.T_d, label_obj.width, pc, visualize=False)
#
#             if collision_intensity>0:
#                 continue
#
#
#         except Exception as e:
#             print(Fore.RED, str(e),Fore.RESET)
#             continue
#
#         '''save to buffer'''
#         training_data.depth.save_as_numpy(depth,target_file_index)
#         training_data.label.save_as_numpy(label,target_file_index)
#
#         '''update counter'''
#         counter+=1
#         progress_indicator.step(counter)
#         if counter >= size:
#             break

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

        label_obj=LabelObj(label=label,depth=depth)

        pose_7=label_obj.get_gripper_pose_7()
        pixel_index=label_obj.get_pixel_index()

        return depth[np.newaxis,:,:],pose_7, pixel_index

    def __len__(self):
        return len(self.files_indexes)

