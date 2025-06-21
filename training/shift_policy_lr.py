import numpy as np
import torch
from Configurations.config import workers
from Online_data_audit.data_tracker2 import DataTracker2
from check_points.check_point_conventions import ModelWrapper, ActorCriticWrapper
from dataloaders.shift_policy_dl import ClearPolicyDataset
from lib.Multible_planes_detection.plane_detecttion import  bin_planes_detection
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data2
from lib.depth_map import depth_to_point_clouds, transform_to_camera_frame
from lib.image_utils import view_image
from lib.report_utils import progress_indicator
from models.action_net import action_module_key, ActionNet
from models.shift_policy_net import shift_policy_module_key,  ShiftPolicyCriticNet, ShiftPolicyActorNet
from records.training_satatistics import MovingRate
from registration import camera
from training.ppo_memory import PPOMemory
from lib.report_utils import wait_indicator as wi
from visualiztion import view_npy_open3d

buffer_file = 'buffer.pkl'
action_data_tracker_path = r'online_data_dict.pkl'
cache_name = 'clustering'

online_data2 = online_data2()


def policy_loss(new_policy_probs, old_policy_probs, advantages, epsilon=0.2):
    ratio = new_policy_probs / old_policy_probs
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    objective = torch.min(ratio * advantages, clipped_ratio * advantages)
    return objective


class TrainPolicyNet:
    def __init__(self, learning_rate=5e-5):

        self.policy_clip_margin = 0.2
        self.action_net = None
        self.learning_rate = learning_rate
        self.model_wrapper = ActorCriticWrapper(module_key=shift_policy_module_key,actor=ShiftPolicyActorNet,critic=ShiftPolicyCriticNet)

        '''initialize statistics records'''

        self.buffer = online_data2.load_pickle(buffer_file) if online_data2.file_exist(buffer_file) else PPOMemory()

        self.data_tracker = DataTracker2(name=action_data_tracker_path, list_size=10)
        self.last_tracker_size = len(self.data_tracker)

        self.buffer_time_stamp = None
        self.data_tracker_time_stamp = None

        self.gripper_sampling_rate = MovingRate('gripper_sampling_rate', min_decay=0.01)

        ''''statistics tracker'''
        self.ini_policy_moving_loss = MovingRate(shift_policy_module_key + '_ini_policy_moving_loss', min_decay=0.01)
        self.ini_value_moving_loss = MovingRate(shift_policy_module_key + '_ini_value_moving_loss', min_decay=0.01)

    def initialize_model(self):
        self.model_wrapper.ini_models(train=True)
        self.model_wrapper.actor_adam_optimizer(learning_rate=self.learning_rate, beta1=0.9)
        self.model_wrapper.critic_adam_optimizer(learning_rate=self.learning_rate, beta1=0.9)

    def synchronize_buffer(self):
        new_buffer = False
        new_data_tracker = False
        new_buffer_time_stamp = online_data2.file_time_stamp(buffer_file)
        if self.buffer_time_stamp is None or new_buffer_time_stamp != self.buffer_time_stamp:
            self.buffer = online_data2.load_pickle(buffer_file) if online_data2.file_exist(buffer_file) else PPOMemory()
            self.buffer_time_stamp = new_buffer_time_stamp
            new_buffer = True

        new_data_tracker_time_stamp = online_data2.file_time_stamp(action_data_tracker_path)
        if self.data_tracker_time_stamp is None or self.data_tracker_time_stamp != new_data_tracker_time_stamp:
            self.data_tracker = DataTracker2(name=action_data_tracker_path, list_size=10)
            self.data_tracker_time_stamp = new_data_tracker_time_stamp
            new_data_tracker = True

        return new_buffer, new_data_tracker

    def get_clear_policy_dataloader(self, file_ids, batch_size, buffer):
        dataset = ClearPolicyDataset(data_pool=online_data2, policy_buffer=buffer, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                                  shuffle=True)
        return data_loader

    def initialize_action_net(self):
        actions_net = ModelWrapper(model=ActionNet(), module_key=action_module_key)
        actions_net.ini_model(train=False)
        self.action_net = actions_net.model

    def get_point_clouds(self, depth):
        pcs = []
        masks = []
        for j in range(depth.shape[0]):
            pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
            pc = transform_to_camera_frame(pc, reverse=True)

            pcs.append(pc)
            masks.append(mask)

        return pcs, masks


    def analytical_bin_mask(self, pc, file_ids):
        try:
            bin_mask ,floor_elevation= bin_planes_detection(pc, sides_threshold=0.005, floor_threshold=0.0015, view=False,
                                            file_index=file_ids[0], cache_name='bin_planes2')
        except Exception as error_message:
            print(file_ids[0])
            print(error_message)
            bin_mask = None
        return bin_mask

    def forward_clear_policy_loss(self, clear_policy_batch):

        (rgb, depth, target_masks, values, advantages,
         action_indexes, point_indexes, probs, rewards,
         end_of_episodes) = clear_policy_batch

        rgb = rgb.cuda().float().permute(0, 3, 1, 2)
        target_masks = target_masks.cuda().float()
        depth = depth.cuda().float()

        pcs, masks = self.get_point_clouds(depth)

        b = rgb.shape[0]
        w = rgb.shape[2]
        h = rgb.shape[3]

        '''process pose'''
        pose_7_stack = torch.zeros((b, 7, w, h), device=rgb.device)
        normal_stack = torch.zeros((b, 3, w, h), device=rgb.device)

        action_probs= self.model_wrapper.actor(rgb, depth.clone(), target_masks,shift_mask=None)
        q_value= self.model_wrapper.critic(rgb, depth.clone(), target_masks)



        '''accumulate critic actor loss'''
        actor_loss = torch.tensor(0., device=q_value.device)
        critic_loss = torch.tensor(0., device=q_value.device)
        for j in range(b):
            mask = masks[j]
            action_index = action_indexes[j]
            point_index = point_indexes[j]

            q_value_j = q_value[j].permute(1, 2, 0)[mask]
            action_probs_j = action_probs[j].permute(1, 2, 0)[mask]

            old_probs = probs[j]
            new_probs = action_probs_j[point_index, action_index]
            new_probs = torch.log(new_probs)

            prob_ratio = new_probs.exp() / old_probs.exp()
            weighted_probs = advantages[j] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip_margin,
                                                 1 + self.policy_clip_margin) * advantages[j]

            actor_loss += -torch.min(weighted_probs, weighted_clipped_probs)

            critic_value = q_value_j[point_index, action_index]
            returns = advantages[j] + values[j]

            critic_loss += (returns - critic_value) ** 2

        total_loss = actor_loss + 0.5 * critic_loss

        return total_loss

    def train(self, batch_size=1, n_epochs=4):
        '''dataloaders'''
        clear_policy_ids = self.buffer.generate_batches(batch_size)

        clear_policy_data_loader = self.get_clear_policy_dataloader(clear_policy_ids, batch_size, self.buffer)

        counter = 0
        for e in range(n_epochs):
            for  clear_policy_batch in  clear_policy_data_loader:
                counter += 1
                self.model_wrapper.model.zero_grad()

                clear_policy_loss = self.forward_clear_policy_loss(clear_policy_batch)
                clear_policy_loss.backward()

                self.model_wrapper.optimizer.step()
                self.model_wrapper.optimizer.zero_grad()

    def view_result(self):
        with torch.no_grad():

            self.gripper_sampling_rate.view()

            self.ini_policy_moving_loss.view()
            self.ini_value_moving_loss.view()

    def save_statistics(self):
        self.gripper_sampling_rate.save()

        self.ini_policy_moving_loss.save()
        self.ini_value_moving_loss.save()

    def export_check_points(self):
        try:
            self.model_wrapper.export_models()
            self.model_wrapper.export_optimizers()
        except Exception as e:
            print(str(e))


if __name__ == "__main__":
    # seeds(0)
    lr = 1e-4
    train_action_net = TrainPolicyNet(learning_rate=lr)

    train_action_net.initialize_model()s
    train_action_net.synchronize_buffer()

    wait = wi('Begin synchronized trianing')

    while True:
        new_buffer, new_data_tracker = train_action_net.synchronize_buffer()

        '''test code'''
        train_action_net.train(max_size=10)

        train_action_net.export_check_points()
        train_action_net.save_statistics()
        train_action_net.view_result()
