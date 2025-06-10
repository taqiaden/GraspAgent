import os
import numpy as np
import torch
from Configurations.config import workers
from Online_data_audit.data_tracker2 import DataTracker2
from check_points.check_point_conventions import ModelWrapper
from dataloaders.grasp_handover_dl import DemonstrationsDataset, SeizePolicyDataset
from lib.IO_utils import load_pickle, save_pickle
from lib.Multible_planes_detection.plane_detecttion import cache_dir, bin_planes_detection
from lib.dataset_utils import online_data2, demonstrations_data
from lib.depth_map import depth_to_point_clouds, transform_to_camera_frame
from lib.image_utils import view_image
from lib.loss.D_loss import binary_smooth_l1, binary_l1
from lib.report_utils import progress_indicator
from lib.sklearn_clustering import dbscan_clustering
from models.Grasp_handover_policy_net import GraspHandoverPolicyNet, grasp_handover_policy_module_key
from models.action_net import action_module_key, ActionNet
from records.training_satatistics import TrainingTracker, MovingRate
from registration import camera
import random
from lib.report_utils import wait_indicator as wi
from training.ppo_memory import PPOMemory
from visualiztion import view_npy_open3d

buffer_file = 'buffer.pkl'
action_data_tracker_path = r'online_data_dict.pkl'
cache_name = 'clustering'

online_data2 = online_data2()
demonstrations_data = demonstrations_data()

bce_loss= torch.nn.BCELoss()

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
        self.model_wrapper = ModelWrapper(model=GraspHandoverPolicyNet(), module_key=grasp_handover_policy_module_key)
        self.quality_dataloader = None

        '''initialize statistics records'''
        self.gripper_quality_net_statistics = TrainingTracker(name=grasp_handover_policy_module_key + '_gripper_quality',
                                                              track_label_balance=True, min_decay=0.005)
        self.suction_quality_net_statistics = TrainingTracker(name=grasp_handover_policy_module_key + '_suction_quality',
                                                              track_label_balance=True, min_decay=0.005)

        self.demonstrations_statistics = TrainingTracker(name=grasp_handover_policy_module_key + '_demonstrations',
                                                         track_label_balance=False, min_decay=0.005)

        self.buffer = online_data2.load_pickle(buffer_file) if online_data2.file_exist(buffer_file) else PPOMemory()

        self.data_tracker = DataTracker2(name=action_data_tracker_path, list_size=10)
        self.last_tracker_size = len(self.data_tracker)

        self.buffer_time_stamp = None
        self.data_tracker_time_stamp = None

        self.gripper_sampling_rate = MovingRate('gripper_sampling_rate', min_decay=0.005)

    def initialize_model(self):
        self.model_wrapper.ini_model(train=True)
        self.model_wrapper.ini_adam_optimizer(learning_rate=self.learning_rate, beta1=0.9)

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

    def experience_sampling(self, replay_size):
        suction_size = int(self.gripper_sampling_rate.val * replay_size)
        gripper_size = replay_size - suction_size
        gripper_ids = self.data_tracker.gripper_grasp_sampling(gripper_size,
                                                               self.gripper_quality_net_statistics.label_balance_indicator)
        suction_ids = self.data_tracker.suction_grasp_sampling(suction_size,
                                                               self.suction_quality_net_statistics.label_balance_indicator)

        return gripper_ids + suction_ids

    def demonstrations_buffer_sampling(self, batch_size, n_batches):
        all_ids = demonstrations_data.get_indexes()
        sampled_size = int(batch_size * n_batches)
        if sampled_size<len(all_ids):
            sampled_ids = random.sample(all_ids, sampled_size)
            return sampled_ids
        else:
            return all_ids

    def mixed_buffer_sampling(self, batch_size, n_batches, online_ratio=0.5):
        total_size = int(batch_size * n_batches)
        online_size = int(total_size * online_ratio)
        available_buffer_size = len(self.buffer.non_episodic_file_ids)
        online_size = min(available_buffer_size, online_size)
        replay_size = total_size - online_size
        print(f'Sample {online_size} from online buffer and {replay_size} from experience.')

        '''sample from old experience'''
        replay_ids = self.experience_sampling(replay_size)

        '''sample from online pool'''
        if available_buffer_size == 0:
            print('No file is found in the recent buffer')
            return replay_ids
        else:
            indexes = np.random.choice(np.arange(available_buffer_size, dtype=np.int64), online_size).tolist()
            online_ids = [self.buffer.non_episodic_file_ids[i] for i in indexes]

        return online_ids + replay_ids

    def get_seize_policy_dataloader(self, file_ids, batch_size):
        dataset = SeizePolicyDataset(data_pool=online_data2, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                                  shuffle=True)
        return data_loader

    def get_demonstrations_data_loader(self, file_ids, batch_size):
        dataset = DemonstrationsDataset(data_pool=demonstrations_data, file_ids=file_ids)
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

    def quality_loss(self, griper_grasp_quality_score, suction_grasp_quality_score, gripper_score, suction_score,
                     used_gripper, used_suction, gripper_pixel_index, suction_pixel_index):
        loss = torch.tensor(0., device=griper_grasp_quality_score.device) * griper_grasp_quality_score.mean()
        for j in range(griper_grasp_quality_score.shape[0]):

            weight = 1.0
            if used_gripper[j]:
                label = gripper_score[j]
                if label == -1: continue
                g_pix_A = gripper_pixel_index[j, 0]
                g_pix_B = gripper_pixel_index[j, 1]
                prediction = griper_grasp_quality_score[j, 0, g_pix_A, g_pix_B]
                l = binary_smooth_l1(prediction, label)

                self.gripper_sampling_rate.update(1)

                self.gripper_quality_net_statistics.loss = l.item()
                self.gripper_quality_net_statistics.update_confession_matrix(label, prediction)
                loss += l
                if used_suction[j]: weight = 2.0

            if used_suction[j]:
                label = suction_score[j]
                if label == -1: continue
                s_pix_A = suction_pixel_index[j, 0]
                s_pix_B = suction_pixel_index[j, 1]
                prediction = suction_grasp_quality_score[j, 0, s_pix_A, s_pix_B]
                l = binary_smooth_l1(prediction, label)

                self.gripper_sampling_rate.update(0)

                self.suction_quality_net_statistics.loss = l.item()
                self.suction_quality_net_statistics.update_confession_matrix(label, prediction)

                loss += l * weight

        return loss

    def analytical_bin_mask(self, pc, file_ids):
        try:
            bin_mask = bin_planes_detection(pc, sides_threshold=0.005, floor_threshold=0.0015, view=False,
                                            file_index=file_ids[0], cache_name='bin_planes2')
        except Exception as error_message:
            print(file_ids[0])
            print(error_message)
            bin_mask = None
        return bin_mask

    def simulate_elevation_variations(self, original_depth, objects_mask, max_elevation=0.2, exponent=2.0):
        '''Elevation-based Augmentation'''
        shift_entities_mask = objects_mask & (original_depth > 0.0001)
        new_depth = original_depth.clone().detach()
        new_depth[shift_entities_mask] -= max_elevation * (np.random.rand() ** exponent) * camera.scale
        return new_depth

    def train(self, max_size=100, batch_size=1):
        '''dataloaders'''
        # seize_policy_ids = self.mixed_buffer_sampling(batch_size=batch_size, n_batches=max_size)
        seize_policy_ids = self.experience_sampling(int(batch_size * max_size))
        seize_policy_data_loader = self.get_seize_policy_dataloader(seize_policy_ids, batch_size)

        '''demonstrations'''
        demonstrations_ids = self.demonstrations_buffer_sampling(batch_size=batch_size, n_batches=max_size)
        demonstrations_data_loader = self.get_demonstrations_data_loader(demonstrations_ids, batch_size)

        pi = progress_indicator('Begin new training round: ', max_limit=len(seize_policy_data_loader))
        if self.action_net is None: self.initialize_action_net()

        for i, (seize_policy_batch, demonstrations_batch) in enumerate(
                zip(seize_policy_data_loader, demonstrations_data_loader), start=0):

            '''learn from demonstrations'''
            rgb, depth, labels, file_ids = demonstrations_batch
            rgb = rgb.cuda().float().permute(0, 3, 1, 2)
            depth = depth.cuda().float()

            '''zero grad'''
            self.model_wrapper.model.zero_grad()

            b = rgb.shape[0]
            pcs, masks = self.get_point_clouds(depth)
            objects_masks=[]

            '''Elevation-based augmentation'''
            for k in range(depth.shape[0]):
                '''background detection head'''
                bin_mask = self.analytical_bin_mask(pcs[k], file_ids[k])
                if bin_mask is None: continue
                objects_mask_numpy = bin_mask <= 0.5
                objects_mask = torch.from_numpy(objects_mask_numpy).cuda()
                objects_masks.append(objects_mask)
                objects_mask_pixel_form = torch.ones_like(depth)
                objects_mask_pixel_form[0, 0][masks[k]] = objects_mask_pixel_form[0, 0][masks[k]] * objects_mask
                objects_mask_pixel_form = objects_mask_pixel_form > 0.5
                if np.random.rand() > 0.7:
                    depth[i:i + 1] = self.simulate_elevation_variations(depth[i:i + 1], objects_mask_pixel_form,
                                                                        exponent=5.0)
                    pcs[k], masks[k] = depth_to_point_clouds(depth[0, 0].cpu().numpy(), camera)
                    pcs[k] = transform_to_camera_frame(pcs[k], reverse=True)

            with torch.no_grad():
                gripper_pose, normal_direction, _, _, _ \
                    , background_class = self.action_net(depth.clone())

            griper_grasp_quality_score, suction_grasp_quality_score, handover_class = \
                self.model_wrapper.model(rgb, depth.clone(), gripper_pose, normal_direction)

            demonstration_loss = torch.tensor(0., device=rgb.device)
            for j in range(b):
                objects_mask = objects_masks[j]
                if labels[j, 0] != -1:
                    pass
                elif labels[j, 1] != -1:
                    '''No grasp points'''
                    target_predictions = griper_grasp_quality_score[j, 0][masks[j]]
                    ground_truth = torch.zeros_like(target_predictions)
                    demonstration_loss += (binary_l1(target_predictions[objects_mask], ground_truth[objects_mask]) ** 2.).mean()

                elif labels[j, 2] != -1:
                    '''No suction points'''
                    target_predictions = suction_grasp_quality_score[j, 0][masks[j]]
                    ground_truth = torch.zeros_like(target_predictions)
                    demonstration_loss += (binary_l1(target_predictions[objects_mask], ground_truth[objects_mask]) ** 2.).mean()
                elif labels[j, 3] != -1:
                    '''Priority to grasp'''
                    target_gripper_predictions = griper_grasp_quality_score[j, 0][masks[j]]
                    target_suction_predictions = suction_grasp_quality_score[j, 0][masks[j]]
                    demonstration_loss += (
                        torch.clamp(target_suction_predictions[objects_mask] - target_gripper_predictions[objects_mask],
                                    0.)).mean()
                elif labels[j, 4] != -1:
                    '''Priority to suction'''
                    target_gripper_predictions = griper_grasp_quality_score[j, 0][masks[j]]
                    target_suction_predictions = suction_grasp_quality_score[j, 0][masks[j]]
                    demonstration_loss += (torch.clamp(target_gripper_predictions[objects_mask] - target_suction_predictions[objects_mask],
                                    0.)).mean()
                else:
                    assert False, f'{labels}'

            self.demonstrations_statistics.loss = demonstration_loss.item()
            demonstration_loss.backward()

            self.model_wrapper.optimizer.step()

            '''learn from robot actions'''
            rgb, depth, pose_7, gripper_pixel_index, \
                suction_pixel_index, gripper_score, \
                suction_score, normal, used_gripper, used_suction, file_ids = seize_policy_batch

            rgb = rgb.cuda().float().permute(0, 3, 1, 2)
            depth = depth.cuda().float()
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


            griper_grasp_quality_score, suction_grasp_quality_score, handover_class= \
                self.model_wrapper.model(rgb, depth.clone(), pose_7_stack, normal_stack)


            '''accumulate loss'''
            quality_loss = self.quality_loss(griper_grasp_quality_score, suction_grasp_quality_score,
                                             gripper_score, suction_score, used_gripper, used_suction,
                                             gripper_pixel_index, suction_pixel_index)


            loss = quality_loss

            assert not torch.isnan(loss).any(), f'{loss}'

            loss.backward()

            self.model_wrapper.optimizer.step()

            pi.step(i)
        pi.end()

    def view_result(self):
        with torch.no_grad():
            self.gripper_quality_net_statistics.print()
            self.suction_quality_net_statistics.print()
            self.demonstrations_statistics.print()

            self.gripper_sampling_rate.view()


    def save_statistics(self):
        self.gripper_quality_net_statistics.save()
        self.suction_quality_net_statistics.save()
        self.demonstrations_statistics.save()

        self.gripper_sampling_rate.save()


    def export_check_points(self):
        try:
            self.model_wrapper.export_model()
            self.model_wrapper.export_optimizer()
        except Exception as e:
            print(str(e))

    def clear(self):
        self.gripper_quality_net_statistics.clear()
        self.suction_quality_net_statistics.clear()
        self.demonstrations_statistics.clear()


if __name__ == "__main__":
    lr = 1e-4
    train_action_net = TrainPolicyNet(learning_rate=lr)
    train_action_net.initialize_model()
    train_action_net.synchronize_buffer()
    wait = wi('Begin synchronized trianing')

    while True:
        # try:
        new_buffer, new_data_tracker = train_action_net.synchronize_buffer()

        train_action_net.train(max_size=100)

        train_action_net.export_check_points()
        train_action_net.save_statistics()
        train_action_net.view_result()
