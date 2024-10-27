import numpy as np
from colorama import Fore
from torch.utils import data
from Online_data_audit.sample_training_buffer import get_selection_probabilty
from label_unpack import LabelObj
from lib.dataset_utils import  online_data, data_pool
from lib.report_utils import progress_indicator as pi
import random

training_buffer_dir='dataset/GQ_training_buffer/'

class GQBuffer(data_pool):
    def __init__(self):
        super(GQBuffer,self).__init__(dir=training_buffer_dir,dataset_name='training')
        self.main_modality=self.depth

online_data=online_data()
force_balanced_data=True

def load_training_buffer(size):
    training_buffer = GQBuffer()
    file_indexes = online_data.get_indexes()
    random.shuffle(file_indexes)
    selection_p = get_selection_probabilty(file_indexes)

    progress_indicator=pi(f'load {size} samples for training',size)
    positive_samples=0
    negative_samples=0
    counter=0
    for i,target_file_index in enumerate(file_indexes):
        if np.random.rand() > selection_p[i]: continue
        '''get data'''
        try:
            # depth=online_data.load_depth(target_file_index)
            label = online_data.label.load_as_numpy(target_file_index)
            label_obj = LabelObj(label=label)
            '''selection rules'''
            if label_obj.is_suction:    continue
            if force_balanced_data:
                if label_obj.success and (1+negative_samples)/(1+positive_samples)<1:
                    continue
                elif label_obj.failure and (1+positive_samples)/(1+negative_samples)<1:
                    continue

            '''load depth map'''
            pc = online_data.point_clouds.load_as_numpy(target_file_index)
            depth=label_obj.get_depth(point_clouds=pc)

        except Exception as e:
            print(Fore.RED, str(e),Fore.RESET)
            continue

        '''save to buffer'''
        training_buffer.depth.save_as_numpy(depth,target_file_index)
        training_buffer.label.save_as_numpy(label,target_file_index)

        '''update counter'''
        if label[3]==1:positive_samples+=1
        else: negative_samples+=1
        counter+=1
        progress_indicator.step(counter)
        if counter >= size:
            print(
                f'Sampled buffer contains {positive_samples} positive samples and {negative_samples} negative samples')
            break


class gripper_quality_dataset(data.Dataset):
    def __init__(self, data_pool):
        super().__init__()
        self.data_pool = data_pool
        self.files_indexes = self.data_pool.depth.get_indexes()

    def __getitem__(self, idx):
        target_index = self.files_indexes[idx]
        depth = self.data_pool.depth.load_as_numpy(target_index)
        label = self.data_pool.label.load_as_numpy(target_index)
        assert label[4]==1 and label[23]==0,f'{label[4]},   {label[23]}'

        label_obj = LabelObj(label=label, depth=depth)
        score=label_obj.success
        # score = np.asarray(score, dtype=np.float32)
        pose_7=label_obj.get_gripper_pose_7()
        pixel_index=label_obj.get_pixel_index()

        return depth[np.newaxis,:,:],pose_7, score,pixel_index

    def __len__(self):
        return len(self.files_indexes)

