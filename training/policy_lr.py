import time

import numpy as np
import torch
from Configurations.config import workers
from Online_data_audit.data_tracker2 import DataTracker2
from check_points.check_point_conventions import ModelWrapper
from dataloaders.policy_dl import GraspQualityDataset
from lib.dataset_utils import online_data2
from lib.report_utils import progress_indicator
from models.policy_net import policy_module_key, PolicyNet
from records.training_satatistics import TrainingTracker, MovingRate
from training.ppo_memory import PPOMemory
import random
from lib.report_utils import wait_indicator as wi

buffer_file='buffer.pkl'
action_data_tracker_path=r'online_data_dict.pkl'

online_data2=online_data2()

bce_loss= torch.nn.BCELoss()

def policy_loss(new_policy_probs,old_policy_probs,advantages,epsilon=0.2):
    ratio = new_policy_probs / old_policy_probs
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    objective = torch.min(ratio * advantages, clipped_ratio * advantages)
    return objective

class TrainPolicyNet:
    def __init__(self,learning_rate=5e-5):

        self.learning_rate=learning_rate
        self.model_wrapper=ModelWrapper(model=PolicyNet(), module_key=policy_module_key)

        self.quality_dataloader=None

        '''initialize statistics records'''
        self.gripper_quality_net_statistics = TrainingTracker(name=policy_module_key + '_gripper_quality',
                                                              track_label_balance=True)
        self.suction_quality_net_statistics = TrainingTracker(name=policy_module_key + '_suction_quality',
                                                              track_label_balance=True)

        self.buffer = online_data2.load_pickle(buffer_file) if online_data2.file_exist(buffer_file) else PPOMemory()
        self.data_tracker = DataTracker2(name=action_data_tracker_path, list_size=10)
        self.last_tracker_size=len(self.data_tracker)

        self.buffer_time_stamp=None
        self.data_tracker_time_stamp=None

        self.gripper_sampling_rate=MovingRate('gripper_sampling_rate')

    def initialize_model(self):
        self.model_wrapper.ini_model(train=True)
        self.model_wrapper.ini_adam_optimizer(learning_rate=self.learning_rate)

    # @property
    # def training_trigger(self):
    #     return len(self.data_tracker)>self.last_tracker_size

    def synchronize_buffer(self):
        new_buffer=False
        new_data_tracker=False
        new_buffer_time_stamp=online_data2.file_time_stamp(buffer_file)
        if self.buffer_time_stamp is None or new_buffer_time_stamp!=self.buffer_time_stamp:
            self.buffer = online_data2.load_pickle(buffer_file) if online_data2.file_exist(buffer_file) else PPOMemory()
            self.buffer_time_stamp=new_buffer_time_stamp
            new_buffer=True

        new_data_tracker_time_stamp=online_data2.file_time_stamp(action_data_tracker_path)
        if self.data_tracker_time_stamp is None or self.data_tracker_time_stamp!=new_data_tracker_time_stamp:
            self.data_tracker = DataTracker2(name=action_data_tracker_path, list_size=10)
            self.data_tracker_time_stamp=new_data_tracker_time_stamp
            new_data_tracker=True

        return new_buffer,new_data_tracker

    def experience_sampling(self,replay_size):
        suction_size=int(self.gripper_sampling_rate.val*replay_size)
        gripper_size=replay_size-suction_size

        gripper_ids=self.data_tracker.gripper_grasp_sampling(gripper_size,self.gripper_quality_net_statistics.label_balance_indicator)

        suction_ids=self.data_tracker.suction_grasp_sampling(suction_size,self.suction_quality_net_statistics.label_balance_indicator)

        return gripper_ids+suction_ids

    def mixed_buffer_sampling(self,buffer:PPOMemory,data_tracker:DataTracker2,batch_size,n_batches,online_ratio):
        total_size=int(batch_size*n_batches)
        online_size=int(total_size*online_ratio)
        replay_size=total_size-online_size

        '''sample from online pool'''
        indexes=random.sample(np.arange(len(buffer.non_episodic_buffer_file_ids),dtype=np.int64),online_size)
        online_ids=buffer.non_episodic_buffer_file_ids[indexes]

        '''sample from old experience'''
        replay_ids=data_tracker.selective_grasp_sampling(size=replay_size,sampling_rates=(self.g_p_sampling_rate.val,self.g_n_sampling_rate.val,
                                                                                    self.s_p_sampling_rate.val,self.s_n_sampling_rate.val))
        return online_ids+replay_ids


    def init_quality_data_loader(self,file_ids,batch_size):
        dataset = GraspQualityDataset(data_pool=online_data2, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                                       shuffle=True)
        return  data_loader

    def step_quality_training(self,max_size=100,batch_size=1):
        file_ids=self.experience_sampling(max_size)
        data_loader=self.init_quality_data_loader(file_ids,batch_size)
        pi = progress_indicator('Begin new training round: ', max_limit=len(data_loader))
        # assert size==len(file_ids)

        for i, batch in enumerate(data_loader, 0):

            rgb, depth,mask, pose_7, gripper_pixel_index, \
                suction_pixel_index, gripper_score, \
                suction_score, normal, used_gripper, used_suction = batch

            rgb = rgb.cuda().float().permute(0, 3, 1, 2)
            mask = mask.cuda().float()
            depth=depth.cuda().float()
            pose_7 = pose_7.cuda().float()
            gripper_score = gripper_score.cuda().float()
            suction_score = suction_score.cuda().float()
            normal = normal.cuda().float()

            b = rgb.shape[0]
            w = rgb.shape[2]
            h = rgb.shape[3]

            '''zero grad'''
            self.model_wrapper.model.zero_grad()

            '''process pose'''
            pose_7_stack = torch.zeros((b, 7, w, h), device=rgb.device)
            normal_stack = torch.zeros((b, 3, w, h), device=rgb.device)

            for j in range(b):
                g_pix_A = gripper_pixel_index[j, 0]
                g_pix_B = gripper_pixel_index[j, 1]
                s_pix_A = suction_pixel_index[j, 0]
                s_pix_B = suction_pixel_index[j, 1]
                pose_7_stack[j, :, g_pix_A, g_pix_B] = pose_7[j]
                normal_stack[j, :, s_pix_A, s_pix_B] = normal[j]

            griper_grasp_score, suction_grasp_score, \
                shift_affordance_classifier, q_value, action_probs = \
                self.model_wrapper.model(rgb, depth,pose_7_stack, normal_stack, mask)

            # reversed_decay_ = lambda scores:  torch.clamp(torch.ones_like(scores) - scores, 0).mean()

            '''accumulate loss'''
            loss = torch.tensor(0., device=rgb.device)*griper_grasp_score.mean()
            for j in range(b):
                if used_gripper[j]:
                    label = gripper_score[j]
                    if label==-1:continue
                    g_pix_A = gripper_pixel_index[j, 0]
                    g_pix_B = gripper_pixel_index[j, 1]
                    prediction = griper_grasp_score[j, 0, g_pix_A, g_pix_B]
                    l=bce_loss(prediction, label)

                    self.gripper_sampling_rate.update(1)

                    self.gripper_quality_net_statistics.loss=l.item()
                    self.gripper_quality_net_statistics.update_confession_matrix(label,prediction)
                    loss += l

                if used_suction[j]:
                    label = suction_score[j]
                    if label==-1:continue
                    s_pix_A = suction_pixel_index[j, 0]
                    s_pix_B = suction_pixel_index[j, 1]
                    prediction = suction_grasp_score[j, 0, s_pix_A, s_pix_B]
                    l=bce_loss(prediction, label)

                    self.gripper_sampling_rate.update(0)

                    self.suction_quality_net_statistics.loss=l.item()
                    self.suction_quality_net_statistics.update_confession_matrix(label,prediction)

                    loss += l

            loss.backward()
            self.model_wrapper.optimizer.step()

            pi.step(i)
        pi.end()

    def view_result(self):
        with torch.no_grad():
            self.gripper_quality_net_statistics.print()
            self.suction_quality_net_statistics.print()

            self.gripper_sampling_rate.view()


    def save_statistics(self):
        self.gripper_quality_net_statistics.save()
        self.suction_quality_net_statistics.save()

        self.gripper_sampling_rate.save()

    def export_check_points(self):
        self.model_wrapper.export_model()
        self.model_wrapper.export_optimizer()

    def clear(self):
        self.gripper_quality_net_statistics.clear()
        self.suction_quality_net_statistics.clear()

if __name__ == "__main__":
    lr = 1e-4
    train_action_net = TrainPolicyNet(  learning_rate=lr)
    train_action_net.initialize_model()
    train_action_net.synchronize_buffer()

    wait = wi('Begin synchronized trianing')

    while True:
        new_buffer,new_data_tracker=train_action_net.synchronize_buffer()

        train_action_net.step_quality_training(max_size=30)
        train_action_net.export_check_points()
        train_action_net.save_statistics()
        train_action_net.view_result()
        # if new_data_tracker:
        #     train_action_net.step_quality_training(max_size=10)
        #     train_action_net.export_check_points()
        #     train_action_net.save_statistics()
        #     train_action_net.view_result()
        # else:
        #     wait.step(0.5)
        # if train_action_net.training_trigger:
        #     train_action_net.initialize_model()
        #     train_action_net.synchronize_buffer()
        #     train_action_net.step_quality_training()
        # else:
        #     time.sleep(0.5)