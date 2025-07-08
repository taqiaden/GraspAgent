import copy
import math
import os
import numpy as np
import torch
from colorama import Fore
from filelock import FileLock
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from Configurations.config import workers
from Online_data_audit.data_tracker import  gripper_grasp_tracker, DataTracker
from analytical_suction_sampler import estimate_suction_direction
from check_points.check_point_conventions import ModelWrapper, GANWrapper
from dataloaders.action_dl import  ActionDataset2
from lib.IO_utils import custom_print, load_pickle, save_pickle
from lib.Multible_planes_detection.plane_detecttion import bin_planes_detection, cache_dir
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data2
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from lib.image_utils import view_image
from lib.loss.balanced_bce_loss import BalancedBCELoss
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from models.Grasp_GAN import G, N_gripper_sampling_module_key
from models.action_net import ActionNet, action_module_key
from models.scope_net import scope_net_vanilla, gripper_scope_module_key
from records.training_satatistics import TrainingTracker, MovingRate, truncate
from registration import camera
from training.N_Grasp_GAN_training import get_normal_direction
from training.learning_objectives.gripper_collision import  gripper_object_collision_loss, gripper_bin_collision_loss
from training.learning_objectives.shift_affordnace import shift_affordance_loss
from training.learning_objectives.suction_seal import get_suction_seal_loss
from visualiztion import view_o3d, view_npy_open3d

detach_backbone = False
lock = FileLock("file.lock")

view_mode = False
batch_size = 16

training_buffer = online_data2()
training_buffer.main_modality = training_buffer.depth

bce_loss = nn.BCELoss()

balanced_bce_loss = BalancedBCELoss()
print = custom_print

cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
cache_name = 'normals'
discrepancy_distance = 1.0

collision_expo=2.0
firmness_expo=2.0

import torch

def suction_sampler_loss(pc, target_normal,objects_mask, file_index):
    file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
    if os.path.exists(file_path):
        labels = load_pickle(file_path)
    else:
        labels = estimate_suction_direction(pc,
                                            view=False)  # inference time on local computer = 1.3 s        if file_index is not None:
        file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
        save_pickle(file_path, labels)

    labels = torch.from_numpy(labels).to('cuda')

    return ((1 - cos(target_normal[objects_mask], labels.squeeze()[objects_mask])) ** 2).mean()

def balanced_sampling(values, mask=None, exponent=2.0, balance_indicator=1.0):
    with torch.no_grad():
        max_ = values.max().item()
        min_ = values.min().item()
        range_ = max_ - min_

        assert range_>0.

        pivot_point = np.sqrt(np.abs(balance_indicator)) * np.sign(balance_indicator)
        xa = ((max_ - values) / range_) * pivot_point
        selection_probability = ((1 - pivot_point) / 2 + xa + 0.5 * (1 - abs(pivot_point)))
        selection_probability = selection_probability ** exponent

        # print(f'max={selection_probability.max()}, min={selection_probability.min()}')

        if mask is None:
            dist = Categorical(probs=selection_probability)
        else:
            dist = MaskedCategorical(probs=selection_probability, mask=torch.from_numpy(mask).cuda())

        target_index = dist.sample()

        return target_index

class TrainActionNet:
    def __init__(self, n_samples=None, epochs=1, learning_rate=5e-5):
        self.n_samples = n_samples
        self.size = n_samples
        self.epochs = epochs
        self.learning_rate = learning_rate

        '''model wrapper'''
        self.actions_net = self.prepare_model_wrapper()
        self.data_loader = None

        '''Moving rates'''
        self.moving_collision_rate = None
        self.moving_firmness = None
        self.moving_out_of_scope = None
        self.relative_sampling_timing = None
        self.superior_A_model_moving_rate = None
        self.gradient_moving_rate = None

        '''initialize statistics records'''
        self.bin_collision_statistics = None
        self.objects_collision_statistics = None
        self.suction_head_statistics = None
        self.shift_head_statistics = None
        self.gripper_sampler_statistics = None
        self.suction_sampler_statistics = None
        self.background_detector_statistics = None
        self.data_tracker = None


        self.sampling_centroid = None
        self.diversity_momentum = 1.0

        self.skipped_last_G_training=True

        self.gripper_model_N=None
        self.initialize_gripper_sampling_model()


    def initialize(self, n_samples=None):
        self.n_samples = n_samples
        self.prepare_data_loader()

        '''Moving rates'''
        self.moving_collision_rate = MovingRate(action_module_key + '_collision', decay_rate=0.01, initial_val=1.)
        self.moving_firmness = MovingRate(action_module_key + '_firmness', decay_rate=0.01, initial_val=0.)
        self.moving_out_of_scope = MovingRate(action_module_key + '_out_of_scope', decay_rate=0.01, initial_val=1.)
        self.relative_sampling_timing = MovingRate(action_module_key + '_relative_sampling_timing', decay_rate=0.01,
                                                   initial_val=1.)
        self.superior_A_model_moving_rate=MovingRate(action_module_key + '_superior_A_model', decay_rate=0.01,
                                                   initial_val=0.)
        self.gradient_moving_rate = MovingRate(action_module_key + '_gradient', decay_rate=0.01, initial_val=1000.)


        '''initialize statistics records'''
        self.suction_head_statistics = TrainingTracker(name=action_module_key + '_suction_head',
                                                       iterations_per_epoch=len(self.data_loader),
                                                       track_label_balance=True)
        self.bin_collision_statistics = TrainingTracker(name=action_module_key + '_bin_collision',
                                                        iterations_per_epoch=len(self.data_loader),
                                                        track_label_balance=True)
        self.objects_collision_statistics = TrainingTracker(name=action_module_key + '_objects_collision',
                                                            iterations_per_epoch=len(self.data_loader),
                                                            track_label_balance=True)
        self.shift_head_statistics = TrainingTracker(name=action_module_key + '_shift_head',
                                                     iterations_per_epoch=len(self.data_loader),
                                                     track_label_balance=True)
        self.gripper_sampler_statistics = TrainingTracker(name=action_module_key + '_gripper_sampler',
                                                          iterations_per_epoch=len(self.data_loader),
                                                          track_label_balance=False)
        self.suction_sampler_statistics = TrainingTracker(name=action_module_key + '_suction_sampler',
                                                          iterations_per_epoch=len(self.data_loader),
                                                          track_label_balance=False)

        self.background_detector_statistics = TrainingTracker(name=action_module_key + '_background_detector',
                                                              iterations_per_epoch=len(self.data_loader),
                                                              track_label_balance=False)

        self.data_tracker = DataTracker(name=gripper_grasp_tracker)

        gripper_scope = ModelWrapper(model=scope_net_vanilla(in_size=6), module_key=gripper_scope_module_key)
        gripper_scope.ini_model(train=False)
        self.gripper_arm_reachability_net = gripper_scope.model

    def initialize_gripper_sampling_model(self):
        # v_ = GANWrapper(gripper_sampling_module_key, G)
        # v_.ini_generator(train=False)
        # self.gripper_model_V=v_.generator
        #
        # for param in self.gripper_model_V.parameters():
        #     param.requires_grad_(False)

        n_ = GANWrapper(N_gripper_sampling_module_key,G)
        n_.ini_generator(train=False)
        self.gripper_model_N=n_.generator

        for param in self.gripper_model_N.parameters():
            param.requires_grad_(False)

    def prepare_data_loader(self):
        file_ids = training_buffer.get_indexes()
        # file_ids = sample_positive_buffer(size=self.n_samples, dict_name=gripper_grasp_tracker,
        #                                   disregard_collision_samples=True,sample_with_probability=False)
        print(Fore.CYAN, f'Buffer size = {len(file_ids)}', Fore.RESET)
        dataset = ActionDataset2(data_pool=training_buffer, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers,
                                                  shuffle=True)
        self.size = len(dataset)
        self.data_loader = data_loader

    def prepare_model_wrapper(self):
        '''load  models'''
        actions_net = ModelWrapper( model=ActionNet(),module_key=action_module_key)
        actions_net.ini_model(train=True)
        actions_net.ini_adam_optimizer(learning_rate=self.learning_rate, beta1=0.9)
        return actions_net

    def simulate_elevation_variations(self, original_depth,objects_mask, max_elevation=0.15, exponent=2.0):
        '''Elevation-based Augmentation'''
        shift_entities_mask = objects_mask & (original_depth > 0.0001)
        new_depth = original_depth.clone().detach()
        new_depth[shift_entities_mask] -= max_elevation * (np.random.rand() ** exponent) * camera.scale
        return new_depth

    def analytical_bin_mask(self, pc, file_ids):
        try:
            bin_mask,floor_elevation_ = bin_planes_detection(pc, sides_threshold=0.005, floor_threshold=0.0015, view=False,
                                            file_index=file_ids[0], cache_name='bin_planes2')
            return bin_mask, floor_elevation_
        except Exception as error_message:
            print(file_ids[0])
            print(error_message)
            return None,None

    def begin(self):
        pi = progress_indicator('Begin new training round: ', max_limit=len(self.data_loader))
        for i, batch in enumerate(self.data_loader, 0):
            depth, file_ids = batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pi.step(i)

            pc, mask = depth_to_point_clouds(depth[0, 0].cpu().numpy(), camera)
            pc = transform_to_camera_frame(pc, reverse=True)

            '''background detection head'''
            bin_mask,floor_elevation_ = self.analytical_bin_mask(pc, file_ids)
            if bin_mask is None: continue
            objects_mask_numpy = bin_mask <= 0.5
            objects_mask = torch.from_numpy(objects_mask_numpy).cuda()
            objects_mask_pixel_form = torch.ones_like(depth)
            objects_mask_pixel_form[0, 0][mask] = objects_mask_pixel_form[0, 0][mask] * objects_mask
            objects_mask_pixel_form = objects_mask_pixel_form > 0.5

            '''Elevation-based augmentation'''
            if np.random.rand()>0.7:
                depth=self.simulate_elevation_variations(depth,objects_mask_pixel_form,exponent=5.0)
                pc, mask = depth_to_point_clouds(depth[0, 0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)

            '''Approach vectors'''
            verticle_approach=torch.zeros_like(depth)
            verticle_approach=verticle_approach.repeat(1, 3, 1, 1)
            verticle_approach[0,2,...]+=1.0 # [b,3,h,w]

            normal_approach_point_wise_form=get_normal_direction(pc, file_ids[0]).float()
            normal_approach=torch.zeros_like(depth)
            normal_approach=normal_approach.repeat(1,3,1,1)
            normal_approach=normal_approach.permute(0,2,3,1)
            normal_approach[0][mask] = normal_approach[0][mask] +normal_approach_point_wise_form
            normal_approach=normal_approach.permute(0,3,1,2)

            if i % 30 == 0 and i != 0:
                self.export_check_points()
                self.save_statistics()
                self.view_result()
                try:
                    self.initialize_gripper_sampling_model()
                except Exception as e:
                    print(str(e))

            '''zero grad'''
            self.actions_net.model.zero_grad(set_to_none=True)
            # cuda_memory_report()

            '''generated grasps'''
            gripper_pose_tuble, predicted_normal, griper_collision_classifier_tuble, suction_quality_classifier, shift_affordance_classifier, background_class = self.actions_net.model(
                depth.clone(),approach1=verticle_approach,approach2=normal_approach, detach_backbone=detach_backbone)

            '''loss computation'''
            suction_seal_loss = torch.tensor(0.,device=predicted_normal.device)
            gripper_collision_loss = torch.tensor(0.,device=predicted_normal.device)
            shift_appealing_loss = torch.tensor(0.,device=predicted_normal.device)
            background_detection_loss = torch.tensor(0.,device=predicted_normal.device)
            suction_sampling_loss = torch.tensor(0.,device=predicted_normal.device)

            # '''normal dependency mask'''
            # normal_dependency_mask=(suction_quality_classifier>0.5) | (shift_affordance_classifier>0.5) | (griper_collision_classifier_tuble[1][0]<0.5)

            suction_sampling_loss += suction_sampler_loss(pc, predicted_normal.permute(0, 2, 3, 1)[0][mask],objects_mask,
                                                          file_index=file_ids[0])

            '''gripper sampling loss'''
            with torch.no_grad():
                gripper_pose_V = self.gripper_model_N(
                    depth.clone(),verticle_approach)
                gripper_pose_N = self.gripper_model_N(
                    depth.clone(), normal_approach)

                gripper_pose_V[:,5:7,...]=torch.clamp(gripper_pose_V[:,5:7,...],0.01,0.99)
                gripper_pose_N[:,5:7,...]=torch.clamp(gripper_pose_N[:,5:7,...],0.01,0.99)

                label_V_pose_collision_score= self.actions_net.model.collision_prediction(depth.clone(),gripper_pose_V)
                label_N_pose_collision_score= self.actions_net.model.collision_prediction(depth.clone(),gripper_pose_N)
                # print(griper_collision_classifier_tuble)
                V_objects_collision=griper_collision_classifier_tuble[0][0,0][mask]
                N_objects_collision=griper_collision_classifier_tuble[1][0,0][mask]

                label_V_objects_collision=label_V_pose_collision_score[0,0][mask]
                label_N_objects_collision=label_N_pose_collision_score[0,0][mask]
                label_V_bin_collision=label_V_pose_collision_score[0,1][mask]
                label_N_bin_collision=label_N_pose_collision_score[0,1][mask]

                '''superior approach mask'''
                v_is_superior=torch.ones_like(objects_mask)>0.
                bad_V_mask=(label_V_bin_collision>0.5) & (label_V_objects_collision>0.5)
                good_N_mask=(label_N_bin_collision<0.5) & (label_N_objects_collision<0.5)
                v_is_superior[bad_V_mask&good_N_mask]*=False

                def col_weight(x):
                    return 4*x*(1-x)

                v_weights=(0.001+objects_mask.float())*col_weight(V_objects_collision.detach().clone())*(v_is_superior.float()+0.001)
                N_weights=(0.001+objects_mask.float())*col_weight(N_objects_collision.detach().clone())*(1.0-v_is_superior.float()+0.001)

                # v_best_index=torch.argmax(v_weights)
                # N_best_index=torch.argmax(N_weights)

                v_weights/=v_weights.sum()+1e-3
                N_weights/=N_weights.sum()+1e-3

            label_V_pose = gripper_pose_V[0].permute(1, 2, 0)[mask]
            label_N_pose = gripper_pose_N[0].permute(1, 2, 0)[mask]
            prediction_V_pose = gripper_pose_tuble[0][0].permute(1, 2, 0)[mask]
            prediction_N_pose = gripper_pose_tuble[1][0].permute(1, 2, 0)[mask]

            # print(f'v:label-{label_V_pose[v_best_index][3:].detach()},--pred-{prediction_V_pose[v_best_index][3:].detach()}')
            # print(f'N:label-{label_N_pose[N_best_index][3:].detach()},--pred-{prediction_N_pose[N_best_index][3:].detach()}')

            dist_loss=((prediction_V_pose[:,-2]-label_V_pose[:,-2])**2.)*v_weights
            dist_loss+=((prediction_N_pose[:,-2]-label_N_pose[:,-2])**2.)*N_weights

            width_loss=((prediction_V_pose[:,-1]-label_V_pose[:,-1])**2.)*v_weights
            width_loss+=((prediction_N_pose[:,-1]-label_N_pose[:,-1])**2.)*N_weights

            beta_loss=((1.001 - F.cosine_similarity(prediction_V_pose[:,3:5],
                                           label_V_pose[:,3:5], dim=-1)))*v_weights
            beta_loss += ((1.001 - F.cosine_similarity(prediction_N_pose[:, 3:5],
                                                     label_N_pose[:, 3:5], dim=-1)))*N_weights

            total_gripper_regression_loss=dist_loss.sum()+width_loss.sum()*0.4+beta_loss.sum()*5.8

            self.gripper_sampler_statistics.loss=total_gripper_regression_loss.item()

            gripper_poses1 = gripper_pose_tuble[0][0].permute(1, 2, 0)[mask].detach()
            gripper_poses2 = gripper_pose_tuble[1][0].permute(1, 2, 0)[mask].detach() 

            suction_head_predictions = suction_quality_classifier[0, 0][mask]
            gripper_head_predictions1 = griper_collision_classifier_tuble[0][0, :].permute(1, 2, 0)[mask]
            gripper_head_predictions2 = griper_collision_classifier_tuble[1][0, :].permute(1, 2, 0)[mask]
            shift_head_predictions = shift_affordance_classifier[0, 0][mask]
            background_class_predictions = background_class.permute(0, 2, 3, 1)[0, :, :, 0][mask]
            normals = predicted_normal[0].permute(1, 2, 0)[mask].detach().cpu().numpy()

            label = torch.from_numpy(bin_mask).to(background_class_predictions.device).float()
            background_detection_loss += bce_loss(background_class_predictions, label)
            self.background_detector_statistics.update_confession_matrix(label,
                                                                         background_class_predictions.detach())

            for k in range(batch_size):
                '''gripper-object collision - 1'''
                while True:
                    gripper_target_index = balanced_sampling(gripper_head_predictions1[:, 0], mask=objects_mask_numpy,
                                                             exponent=20.0,
                                                             balance_indicator=self.objects_collision_statistics.label_balance_indicator)
                    gripper_target_point = pc[gripper_target_index]
                    gripper_prediction_ = gripper_head_predictions1[gripper_target_index]
                    gripper_target_pose = gripper_poses1[gripper_target_index]
                    loss,counted= gripper_object_collision_loss(gripper_target_pose, gripper_target_point, pc,
                                                                  objects_mask_numpy, gripper_prediction_,
                                                                  self.objects_collision_statistics)
                    if counted:break
                gripper_collision_loss+=loss/ batch_size

            for k in range(batch_size):
                '''gripper-object collision - 2'''
                while True:
                    gripper_target_index = balanced_sampling(gripper_head_predictions2[:, 0], mask=objects_mask_numpy,
                                                             exponent=20.0,
                                                             balance_indicator=self.objects_collision_statistics.label_balance_indicator)
                    gripper_target_point = pc[gripper_target_index]
                    gripper_prediction_ = gripper_head_predictions2[gripper_target_index]
                    gripper_target_pose = gripper_poses2[gripper_target_index]
                    loss,counted= gripper_object_collision_loss(gripper_target_pose, gripper_target_point, pc,
                                                                  objects_mask_numpy, gripper_prediction_,
                                                                  self.objects_collision_statistics)
                    if counted:break
                gripper_collision_loss+=loss/ batch_size

            for k in range(batch_size):
                '''gripper-bin collision - 1'''
                while True:
                    gripper_target_index = balanced_sampling(gripper_head_predictions1[:, 1], mask=objects_mask_numpy,
                                                             exponent=20.0,
                                                             balance_indicator=self.bin_collision_statistics.label_balance_indicator)
                    gripper_target_point = pc[gripper_target_index]
                    gripper_prediction_ = gripper_head_predictions1[gripper_target_index]
                    gripper_target_pose = gripper_poses1[gripper_target_index]
                    loss,counted = gripper_bin_collision_loss(gripper_target_pose, gripper_target_point, pc,
                                                               objects_mask_numpy, gripper_prediction_,
                                                               self.bin_collision_statistics,floor_elevation_)

                    if counted:break
                gripper_collision_loss+=loss/ batch_size

            for k in range(batch_size):
                '''gripper-bin collision - 2'''
                while True:
                    gripper_target_index = balanced_sampling(gripper_head_predictions2[:, 1], mask=objects_mask_numpy,
                                                             exponent=20.0,
                                                             balance_indicator=self.bin_collision_statistics.label_balance_indicator)
                    gripper_target_point = pc[gripper_target_index]
                    gripper_prediction_ = gripper_head_predictions2[gripper_target_index]
                    gripper_target_pose = gripper_poses2[gripper_target_index]
                    loss,counted = gripper_bin_collision_loss(gripper_target_pose, gripper_target_point, pc,
                                                               objects_mask_numpy, gripper_prediction_,
                                                               self.bin_collision_statistics,floor_elevation_)
                    if counted:break
                gripper_collision_loss+=loss/ batch_size

            for k in range(batch_size):
                '''suction seal head'''
                suction_target_index = balanced_sampling(suction_head_predictions, mask=objects_mask_numpy,
                                                         exponent=20.0,
                                                         balance_indicator=self.suction_head_statistics.label_balance_indicator)
                suction_prediction_ = suction_head_predictions[suction_target_index]
                suction_seal_loss += get_suction_seal_loss(pc, normals, suction_target_index, suction_prediction_,
                                                  self.suction_head_statistics, objects_mask_numpy) / batch_size

            for k in range(batch_size):
                '''shift affordance head'''
                shift_target_index = balanced_sampling(shift_head_predictions, mask=None, exponent=20.0,
                                                       balance_indicator=self.shift_head_statistics.label_balance_indicator)
                shift_target_point = pc[shift_target_index]
                shift_prediction_ = shift_head_predictions[shift_target_index]
                shift_appealing_loss += shift_affordance_loss(pc, shift_target_point, objects_mask_numpy,
                                                    self.shift_head_statistics, shift_prediction_) / batch_size

            loss=   suction_seal_loss*1  + gripper_collision_loss*1  + shift_appealing_loss*1    +background_detection_loss*30  + suction_sampling_loss*100 +total_gripper_regression_loss*10
            assert not torch.isnan(loss)

            # print(f'loss details: seal={suction_seal_loss.item()}, collision={gripper_collision_loss.item()}, shift={shift_appealing_loss.item()}, back_det={background_detection_loss.item()}, suction={suction_sampling_loss.item()}, gripper_samp={total_gripper_regression_loss.item()}')
            loss.backward()

            # if    total_norm.item()<max(1.0,self.gradient_moving_rate.val*1.5):
            self.actions_net.optimizer.step()
            # else:
            #     print(Fore.RED, 'Heigh gradient warning, pass training',Fore.RESET)

            self.actions_net.model.zero_grad(set_to_none=True)
            self.actions_net.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                self.suction_sampler_statistics.loss = suction_sampling_loss.item()
                self.suction_head_statistics.loss = suction_seal_loss.item()
                self.shift_head_statistics.loss = shift_appealing_loss.item()
                self.background_detector_statistics.loss = background_detection_loss.item()


        pi.end()

        # self.view_result(gripper_poses[unvalid_mask])

        self.export_check_points()
        self.clear()

    def view_result(self):
        with torch.no_grad():
            self.suction_sampler_statistics.print()
            self.suction_head_statistics.print()
            self.bin_collision_statistics.print()
            self.objects_collision_statistics.print()

            self.shift_head_statistics.print()
            self.background_detector_statistics.print()
            self.gripper_sampler_statistics.print()

            # values = gripper_pose.permute(1, 0, 2, 3).flatten(1).detach()


            self.moving_collision_rate.view()
            self.moving_firmness.view()
            self.moving_out_of_scope.view()
            self.relative_sampling_timing.view()
            self.superior_A_model_moving_rate.view()
            self.gradient_moving_rate.view()

    def save_statistics(self):
        self.moving_collision_rate.save()
        self.moving_firmness.save()
        self.moving_out_of_scope.save()
        self.relative_sampling_timing.save()
        self.superior_A_model_moving_rate.save()
        self.gradient_moving_rate.save()

        self.suction_head_statistics.save()
        self.bin_collision_statistics.save()
        self.objects_collision_statistics.save()
        self.shift_head_statistics.save()
        self.background_detector_statistics.save()
        self.gripper_sampler_statistics.save()
        self.suction_sampler_statistics.save()

        self.data_tracker.save()

    def export_check_points(self):
        self.actions_net.export_model()
        self.actions_net.export_optimizer()

    def clear(self):
        self.suction_head_statistics.clear()
        self.bin_collision_statistics.clear()
        self.objects_collision_statistics.clear()
        self.shift_head_statistics.clear()
        self.gripper_sampler_statistics.clear()
        self.suction_sampler_statistics.clear()
        self.background_detector_statistics.clear()

if __name__ == "__main__":
    lr = 1e-4
    train_action_net = TrainActionNet(n_samples=None, learning_rate=lr)
    torch.cuda.empty_cache()
    # torch.autograd.set_detect_anomaly(True)
    for i in range(10000):
        try:
            cuda_memory_report()
            train_action_net.initialize(n_samples=None)
            train_action_net.begin()
        except Exception as e:
            print(str(e))
