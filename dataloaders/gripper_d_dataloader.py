import os
import random
import numpy as np
import smbclient
import torch
from colorama import Fore
from Online_data_audit.sample_training_buffer import get_selection_probabilty
from lib.IO_utils import unbalance_check, update_balance_counter
from lib.bbox import decode_gripper_pose
from lib.collision_unit import grasp_collision_detection
from lib.dataset_utils import data as d
from torch.utils import data
from Configurations import config
from lib.models_utils import initialize_model, initialize_model_state
from lib.pc_utils import point_index
from models.GAGAN import gripper_generator, dense_gripper_generator_path
from models.gripper_D import gripper_discriminator, dense_gripper_discriminator_path
from models.suction_D import affordance_net, affordance_net_model_path
from pose_object import encode_gripper_pose_npy, vectors_to_ratio_metrics
from lib.dataset_utils import training_data, online_data
from process_perception import random_sampling_augmentation
from suction_sampler import estimate_suction_direction

training_data=training_data()
online_data=online_data()

skip_unbalanced_samples = False

activate_balance_data=True
def load_training_data_from_online_pool(number_of_online_samples):
    # counters to balance positive and negative
    # [n_grasp_positive,n_grasp_negative,n_suction_positive,n_suction_negative]
    # second, copy from online data to training data with down sampled point cloud
    generator = initialize_model(gripper_generator,dense_gripper_generator_path)
    generator.eval()
    discriminator=initialize_model(gripper_discriminator,dense_gripper_discriminator_path)

    discriminator.eval()

    suction_model = affordance_net()
    suction_model = initialize_model_state(suction_model, affordance_net_model_path)
    suction_model.eval()

    balance_indicator=0
    unbalance_allowance=0
    online_pc_filenames = online_data.get_indexes()

    # assert len(online_pc_filenames) >= number_of_online_samples
    random.shuffle(online_pc_filenames)
    selection_p=get_selection_probabilty(online_data,online_pc_filenames)
    s_counter=0
    max_aug_from_suction=0.2*number_of_online_samples

    from lib.report_utils import progress_indicator as pi
    progress_indicator=pi(f'Get {number_of_online_samples} training data from online pool',number_of_online_samples)

    balance_counter2 = np.array([0, 0, 0, 0])
    counter=0
    for i,file_name in enumerate(online_pc_filenames):
        if np.random.rand() > selection_p[i]: continue
        sample_index=online_data.get_index(file_name)

        try:
            label=online_data.load_label_by_index(sample_index)
            # print(label.shape)
            # exit()
            # if label[4]==1 and load_grasp==False:continue
            # if label[23]==1 and load_suction==False: continue
            # if only_success and label[3]==0: continue
            if label[23] == 1:# and s_counter>max_aug_from_suction:
                continue
            if label[23] == 1 and label[3]==0:continue

            if label[3]==1 and balance_indicator>unbalance_allowance and label[4]==1:
                continue
            if ((label[4]==1 and label[3]==0) or (label[23] == 1)) and balance_indicator<-unbalance_allowance:
                continue

            point_data=online_data.load_pc(file_name)
        except Exception as e:
            print(Fore.RED, str(e),Fore.RESET)
            continue
        center_point = np.copy(label[:3])
        # print(balance_indicator)
        # assert -unbalance_allowance <= balance_indicator <= unbalance_allowance , f'{balance_indicator}'

        if activate_balance_data:
            balance_data= unbalance_check(label, balance_counter2)==0
            if skip_unbalanced_samples and not balance_data:
                continue

        # for j in range(n):
        j=0
        down_sampled_pc, index = random_sampling_augmentation(center_point, point_data, config.num_points)
        if index is None:
            break
        ####################################################################################################

        # normals = estimate_suction_direction(down_sampled_pc, view=False)
        # down_sampled_pc = np.concatenate([down_sampled_pc, normals], axis=-1)

        label[:3] = down_sampled_pc[index, 0:3]

        global arbitrary_label


        ####################
        # get bad label from generator
        normals = estimate_suction_direction(down_sampled_pc, view=False)
        down_sampled_pc = np.concatenate([down_sampled_pc, normals], axis=-1)
        if label[3]==1:
            with torch.no_grad():
                pc_torch=torch.from_numpy(down_sampled_pc).to('cuda')[None,:,0:3].float()
                generated_poses=generator(pc_torch)

                # scores,_=discriminator(pc_torch,generated_poses)

            # max_score_index=torch.argmax(scores.squeeze()).item()

            # make the generated label
            # new_label = np.copy(label)
            # center_point = down_sampled_pc[max_score_index,0:3]
            # print(generated_poses.shape)
            generated_poses_5 = vectors_to_ratio_metrics(generated_poses)
            pose=generated_poses_5[:,:,index]
            pose_good_grasp = decode_gripper_pose(pose, center_point)
            # Transformation = get_homogenous_matrix(pose_good_grasp)
            # print(pose_good_grasp)
            # rotation_matrix=



            # pose_good_grasp[0, 0] = generated_width_ratio*config.width_scope
            collision_intensity = grasp_collision_detection(pose_good_grasp, point_data, visualize=False)
            if collision_intensity<=0. and label[3]==1:

                with torch.no_grad():
                    pc_torch = torch.from_numpy(down_sampled_pc).to('cuda')[None, :, :].float()
                    # _, _, dense_pose = generator(pc_torch)
                    suction_pred_ ,suction_pred_2 = suction_model(pc_torch)
                    target_score = suction_pred_[0, 0, index].item()

                    if target_score > 0.5:
                        print(Fore.RED, '   >', target_score, Fore.RESET)
                        continue
                    else:
                        print(Fore.GREEN, '   >', target_score, Fore.RESET)
                # print(Fore.GREEN, '-S', Fore.RESET)
            # elif collision_intensity>0. and label[3]==0:
            #     print(Fore.GREEN, '-F', Fore.RESET)
            else:
                print(Fore.RED, '-', Fore.RESET)
                continue
        ###########################################
        training_data.save_labeled_data(down_sampled_pc,label,sample_index + f'_aug{j}')

        if label[23]==1:
            arbitrary_label=None

        if label[3]==1:
            balance_indicator+=1
        else:
            # assert label[3]==0 , f'{label[3]}'
            balance_indicator-=1
        # if index is None:
        #     continue

        balance_counter2 = update_balance_counter(balance_counter2, is_grasp=label[4] == 1,score=label[3])


        counter+=1
        progress_indicator.step(counter)

        if counter >= number_of_online_samples: break

    # view_data_summary(balance_counter=balance_counter2)
    return balance_counter2

class gripper_dataset(data.Dataset):
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



        return pc,pose_7, score,index

    def __len__(self):
        if self.is_local_dir:
            return len(os.listdir(self.path + 'point_cloud/'))
        else:
            return len(smbclient.listdir(self.path + 'point_cloud/'))
