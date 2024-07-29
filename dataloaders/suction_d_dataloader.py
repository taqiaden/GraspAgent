import os
import random
import numpy as np
import smbclient
import torch
from colorama import Fore
from torch.utils import data
from lib.IO_utils import unbalance_check, update_balance_counter
from lib.dataset_utils import training_data,online_data
from Online_data_audit.sample_training_buffer import get_selection_probabilty
from lib.models_utils import initialize_model
from lib.pc_utils import point_index
from Configurations import config

from lib.dataset_utils import data as d
from models.GAGAN import dense_gripper_generator_path, gripper_generator
from models.gripper_D import gripper_discriminator, dense_gripper_discriminator_path
from process_perception import random_sampling_augmentation
from suction_sampler import estimate_suction_direction

online_data=online_data()
training_data=training_data()

skip_unbalanced_samples = True
only_success=False

activate_balance_data=True
def load_training_data_from_online_pool(number_of_online_samples):
    regular_dis = initialize_model(gripper_discriminator,dense_gripper_discriminator_path)

    regular_dis.eval()
    generator = initialize_model(gripper_generator,dense_gripper_generator_path)
    generator.eval()
    # counters to balance positive and negative
    # [n_grasp_positive,n_grasp_negative,n_suction_positive,n_suction_negative]
    # second, copy from online data to training data with down sampled point cloud
    balance_indicator=0.0
    unbalance_allowance=0
    online_pc_filenames = online_data.get_pc_names()
    # assert len(online_pc_filenames) >= number_of_online_samples
    random.shuffle(online_pc_filenames)
    selection_p=get_selection_probabilty(online_data,online_pc_filenames)

    from lib.report_utils import progress_indicator as pi
    progress_indicator=pi(f'Get {number_of_online_samples} training data from online pool ',number_of_online_samples)

    balance_counter2 = np.array([0, 0, 0, 0])
    counter=0
    for i,file_name in enumerate(online_pc_filenames):
        if np.random.rand() > selection_p[i]: continue
        sample_index=online_data.get_index(file_name)
        try:
            label=online_data.load_label_by_index(sample_index)
            if label[4]==1 :continue
            if only_success and label[3]==0: continue
            if label[3]==1 and balance_indicator>unbalance_allowance:
                continue
            elif label[3]==0 and balance_indicator<-1*unbalance_allowance:
                continue

            point_data=online_data.load_pc(file_name)
        except Exception as e:
            print(Fore.RED, str(e),Fore.RESET)
            continue
        center_point = np.copy(label[:3])
        if activate_balance_data:
            balance_data= unbalance_check(label, balance_counter2)==0
            if skip_unbalanced_samples and not balance_data:continue


        # for j in range(n):
        j=0
        # print(j)
        # print(center_point)

        down_sampled_pc, index = random_sampling_augmentation(center_point, point_data, config.num_points)
        if index is None:
            break

        # data-centric adversial measure
        if label[3] == 1:
            with torch.no_grad():
                pc_torch = torch.from_numpy(down_sampled_pc).to('cuda')[None, :, 0:3].float()
                dense_pose = generator(pc_torch)
                quality_score, grasp_ability_score = regular_dis(pc_torch, dense_pose)
                target_score = quality_score[0, 0, index].item()

                if target_score > 0.5:
                    print(Fore.RED, '   >', target_score, Fore.RESET)
                    continue
                else:
                    print(Fore.GREEN, '   >', target_score, Fore.RESET)



        normals = estimate_suction_direction(down_sampled_pc, view=False)
        down_sampled_pc=np.concatenate([down_sampled_pc,normals],axis=-1)

        label[:3] = down_sampled_pc[index,0:3]
        training_data.save_labeled_data(down_sampled_pc,label,sample_index + f'_aug{j}')
        if label[3]==1:
            balance_indicator+=1
        else:
            balance_indicator-=1

        if index is None:
            continue

        balance_counter2 = update_balance_counter(balance_counter2, is_grasp=label[4] == 1,score=label[3])


        counter+=1
        progress_indicator.step(counter)

        if counter >= number_of_online_samples: break

    return balance_counter2

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
