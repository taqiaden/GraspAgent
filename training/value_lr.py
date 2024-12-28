import random

import numpy as np
import torch
from colorama import Fore
from torch import nn
from torch.utils import data

from Configurations.config import workers
from Online_data_audit.data_tracker import gripper_grasp_tracker, sample_all_positive_and_negatives, \
    suction_grasp_tracker
from check_points.check_point_conventions import GANWrapper, ModelWrapper
from dataloaders.value_dl import ValueDataset
from lib.IO_utils import custom_print
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data
from lib.report_utils import progress_indicator
from models.action_net import ActionNet, action_module_key
from models.value_net import ValueNet, value_module_key
from records.training_satatistics import TrainingTracker

# from action_lr import module_key as action_module_key

instances_per_sample=1

training_buffer = online_data()
training_buffer.main_modality=training_buffer.depth
normal_weight = 10

print=custom_print

mes_loss=nn.MSELoss()
bce_loss=nn.BCELoss()


class TrainValueNet:
    def __init__(self,batch_size=1,n_samples=None,epochs=1,learning_rate=5e-5):

        self.batch_size=batch_size
        self.n_samples=n_samples
        self.size=n_samples
        self.epochs=epochs
        self.learning_rate=learning_rate

        '''model wrapper'''
        self.value_net=self.value_model_wrapper()
        self.action_net=self.action_model_wrapper()
        self.data_loader=self.prepare_data_loader()

        '''initialize statistics records'''
        self.gripper_grasp_statistics = TrainingTracker(name=value_module_key + '_gripper_grasp',
                                                       iterations_per_epoch=len(self.data_loader),
                                                       samples_size=self.size, track_prediction_balance=True)
        self.suction_grasp_statistics = TrainingTracker(name=value_module_key + '_suction_grasp',
                                                       iterations_per_epoch=len(self.data_loader),
                                                       samples_size=self.size, track_prediction_balance=True)

        self.swiped_samples =0

    def prepare_data_loader(self):
        gripper_positive_labels,gripper_negative_labels=sample_all_positive_and_negatives(gripper_grasp_tracker, shuffle=True, disregard_collision_samples=True)
        suction_positive_labels,suction_negative_labels=sample_all_positive_and_negatives(suction_grasp_tracker, shuffle=True, disregard_collision_samples=False)
        sampling_size=min(len(gripper_positive_labels),len(gripper_negative_labels),len(suction_positive_labels),len(suction_negative_labels))
        assert sampling_size>1
        if self.n_samples is not None : sampling_size=min(sampling_size,int(self.n_samples/4))

        file_ids = (gripper_positive_labels[0:sampling_size]+gripper_negative_labels[0:sampling_size]
                    +suction_positive_labels[0:sampling_size]+suction_negative_labels[0:sampling_size])
        random.shuffle(file_ids)

        print(Fore.CYAN, f'Buffer size = {len(file_ids)}')
        dataset = ValueDataset(data_pool=training_buffer, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=workers,
                                                       shuffle=True)
        self.size=len(dataset)
        return data_loader

    def value_model_wrapper(self):
        '''load  models'''
        value_net = ModelWrapper(model=ValueNet(),module_key=value_module_key)
        value_net.ini_model(train=True)

        '''optimizers'''
        value_net.ini_adam_optimizer(learning_rate=self.learning_rate)
        return value_net

    def action_model_wrapper(self):
        '''load  models'''
        action_net = GANWrapper(action_module_key, ActionNet)
        action_net.ini_generator(train=False)
        return action_net

    def begin(self):
        pi = progress_indicator('Begin new training round: ', max_limit=len(self.data_loader))
        for i, batch in enumerate(self.data_loader, 0):
            depth,pose_7,pixel_index,score,normal,is_gripper,random_rgb= batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pose_7 = pose_7.cuda().float()
            score = score.cuda().float()
            normal = normal.cuda().float()
            is_gripper = is_gripper.cuda().float()
            random_rgb = random_rgb.cuda().float().permute(0,3,1,2)

            b = depth.shape[0]

            '''zero grad'''
            self.value_net.model.zero_grad()

            '''generated grasps'''
            with torch.no_grad():
                gripper_pose, suction_direction, griper_collision_classifier, suction_quality_classifier, shift_affordance_classifier \
                    ,background_class,depth_features= self.action_net.generator(depth.clone())
                target_object_mask = background_class.detach() <= 0.5
                '''process label'''
                for j in range(b):
                    pix_A = pixel_index[j, 0]
                    pix_B = pixel_index[j, 1]
                    gripper_pose[j, :, pix_A, pix_B] = pose_7[j]
                    suction_direction[j, :, pix_A, pix_B] = normal[j]

            griper_grasp_score,suction_grasp_score,shift_affordance_classifier,q_value=self.value_net.model(random_rgb,depth_features,gripper_pose,suction_direction,target_object_mask)

            '''learning objectives'''
            gripper_grasp_loss = torch.tensor([0.],device=griper_grasp_score.device)
            suction_grasp_loss = torch.tensor([0.],device=griper_grasp_score.device)

            for j in range(b):
                pix_A = pixel_index[j, 0]
                pix_B = pixel_index[j, 1]
                label_ = score[j:j + 1]
                if is_gripper[j] == 1:
                    prediction_ = griper_grasp_score[j, :, pix_A, pix_B]
                    l=bce_loss(prediction_, label_)**2
                    # print(f'g {prediction_.item()}, {label_}')
                    gripper_grasp_loss += l
                    self.gripper_grasp_statistics.loss=l.item()
                    self.gripper_grasp_statistics.update_confession_matrix(label_,prediction_.detach())
                else:
                    prediction_ = suction_grasp_score[j, :, pix_A, pix_B]
                    l=bce_loss(prediction_, label_)**2
                    suction_grasp_loss += l
                    # print(f's {prediction_.item()}, {label_}')

                    self.suction_grasp_statistics.loss = l.item()
                    self.suction_grasp_statistics.update_confession_matrix(label_,prediction_.detach())


            loss = (gripper_grasp_loss+suction_grasp_loss)/b

            reversed_decay_= lambda scores:torch.clamp(torch.ones_like(scores)-scores,0).mean()
            decay_loss=reversed_decay_(griper_grasp_score)+reversed_decay_(suction_grasp_score)+reversed_decay_(shift_affordance_classifier)
            decay_loss*=0.3

            '''q value initialization'''
            q_value_loss=torch.tensor([0.],device=q_value.device)


            for j in range(b):
                target_object_mask_j=target_object_mask[j,0]
                permuted_q_value=q_value[j,0:2].permute(1,2,0)
                q_value_loss += (torch.clamp(
                    permuted_q_value[~target_object_mask_j] - permuted_q_value[target_object_mask_j].mean(), 0.) ** 2).mean()

                grasp_q_values=q_value[j,0:2]
                shift_q_values=q_value[j,2:]

                '''initialize lower q-value for shift compared to grasp'''
                q_value_loss+=(torch.clamp(shift_q_values-grasp_q_values.mean(),0.)**2).mean()

                x1=np.random.randint(0,711)
                x2=np.random.randint(0,711)

                max_index=max(x1,x2)
                min_index=min(x1,x2)

                '''grasp q-value initialization'''
                q_value_loss+=(torch.clamp(grasp_q_values[0,:,max_index].mean()-grasp_q_values[0,:,min_index].mean(),0.)**2).mean()
                q_value_loss+=(torch.clamp(grasp_q_values[1,:,min_index].mean()-grasp_q_values[1,:,max_index].mean(),0.)**2).mean()

                '''shift q-value initialization'''
                q_value_loss+=(torch.clamp(shift_q_values[0,:,max_index].mean()-shift_q_values[0,:,min_index].mean(),0.)**2).mean()
                q_value_loss+=(torch.clamp(shift_q_values[1,:,min_index].mean()-shift_q_values[1,:,max_index].mean(),0.)**2).mean()

            loss=loss+decay_loss+q_value_loss
            loss.backward()
            self.value_net.optimizer.step()

            self.swiped_samples += b
            if i%100==0 and i!=0:
                self.export_check_points()
                self.view()

            pi.step(i)
        pi.end()

        self.export_check_points()
        self.view()
        self.clear()

    def view(self):
        self.gripper_grasp_statistics.print(self.swiped_samples)
        self.suction_grasp_statistics.print(self.swiped_samples)

    def export_check_points(self):
        self.value_net.export_model()
        self.value_net.export_optimizer()

        self.gripper_grasp_statistics.save()
        self.suction_grasp_statistics.save()

    def clear(self):
        self.gripper_grasp_statistics.clear()
        self.suction_grasp_statistics.clear()

if __name__ == "__main__":
    # with torch.no_grad():
    train_value_net = TrainValueNet(batch_size=2, n_samples=None, learning_rate=5e-5)
    train_value_net.begin()
    for i in range(10000):
        try:
            cuda_memory_report()
            train_value_net=TrainValueNet(batch_size=2, n_samples=None, learning_rate=5e-5)
            train_value_net.begin()
        except Exception as error_message:
            print(error_message)