import os
import numpy as np
from colorama import Fore
from filelock import FileLock
from torch import nn, autograd
import torch.nn.functional as F
from torch.distributions import Categorical

from Configurations.config import workers
from Online_data_audit.data_tracker import gripper_grasp_tracker, DataTracker
from analytical_suction_sampler import estimate_suction_direction
from check_points.check_point_conventions import GANWrapper
from dataloaders.Grasp_GAN_dl import GraspGANDataset2
from lib.IO_utils import custom_print, load_pickle, save_pickle
from lib.Multible_planes_detection.plane_detecttion import bin_planes_detection, cache_dir
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data2
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds, transform_to_camera_frame_torch
from lib.loss.balanced_bce_loss import BalancedBCELoss
from lib.models_utils import reshape_for_layer_norm
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from models.Grasp_GAN import  D
from models.action_net import action_module_with_GAN_key, ActionNet
from records.training_satatistics import TrainingTracker, MovingRate
from registration import camera
from training.Actions_learning import suction_sampler_loss
from training.learning_objectives.gripper_collision import evaluate_grasps3, gripper_bin_collision_loss, \
    gripper_object_collision_loss
from training.learning_objectives.shift_affordnace import shift_affordance_loss
from training.learning_objectives.suction_seal import get_suction_seal_loss
from visualiztion import dense_grasps_visualization

lock = FileLock("file.lock")
max_samples_per_image = 1

max_n = 50
batch_size = 16
batch_size2=16
freeze_backbone = False

training_buffer = online_data2()
training_buffer.main_modality = training_buffer.depth

bce_loss = nn.BCELoss()

balanced_bce_loss = BalancedBCELoss()
print = custom_print

cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
cache_name = 'normals'
discrepancy_distance = 1.0

collision_expo = 1.0
firmness_expo = 1.0
generator_expo = 1.0

m = 0.2

import torch


def relativistic_loss(positive_score, negative_score):
    return F.binary_cross_entropy_with_logits(positive_score - negative_score, torch.ones_like(positive_score))


def hinge_contrastive_loss(positive_score, negative_score, margin=1.0, curc_mask=.0):
    return torch.clamp(negative_score - positive_score + margin, 0.) * (1 - curc_mask) + curc_mask * torch.clamp(
        positive_score - negative_score - 1., 0.)

def firmness_loss(c_, s_, f_, q_, prediction_, label_, prediction_uniqness, dist_factor):
    if sum(s_) == 0 and sum(c_) == 0:
        if (f_[1] - f_[0] > 0.) and q_[1] > 0.5:
            relative_firmness = (abs(f_[1] - f_[0]) / (f_[1] + f_[0]))
            margin=(dist_factor ** (1 + m - (q_[1] ** m))) * (q_[1] ** m)*(1-dist_factor)**0.2
            margin=margin**0.5
            # margin = (f_[1] ** m)

            # loss = torch.clamp(1 - prediction_, 0.) ** 2 + torch.clamp(1 - label_, 0.) ** 2 + torch.clamp(
            #     prediction_ - label_ + relative_firmness, 0.) ** 2
            loss=0
            if relative_firmness**2<np.random.rand(): return False, None, None, 0, 0, 0
            return True, label_, prediction_, relative_firmness, 0, loss
        elif (f_[0] - f_[1] > 0.) and q_[0] > 0.5:
            relative_firmness = (abs(f_[1] - f_[0]) / (f_[1] + f_[0]))
            margin=(dist_factor ** (1 + m - (q_[0] ** m))) * (q_[0] ** m)
            margin = (f_[0] ** m)

            # loss = torch.clamp(1 - prediction_, 0.) ** 2 + torch.clamp(1 - label_, 0.) ** 2 + torch.clamp(
            #     prediction_ - label_ + relative_firmness, 0.) ** 2
            loss=0

            return True,  prediction_,label_, relative_firmness, 0, loss
        else:
            return False, None, None, 0, 0, 0
    else:
        return False, None, None, 0, 0, 0

def collision_loss(c_, s_, f_, q_, prediction_, label_, prediction_uniqness, dist_factor):
    if sum(s_) > 0:
        if s_[1] == 0 and c_[1] == 0 and f_[1] > 0 and q_[1] > 0.5:
            # margin =  (dist_factor ** (1 + m - (f_[1] ** m))) * (f_[1] ** m)
            # margin=(dist_factor ** (1 + m - (q_[1] ** m))) * (q_[1] ** m)
            margin=(dist_factor ** (1 + m - (q_[1] ** m))) * (q_[1] ** m)*(1-dist_factor)**0.2
            margin=margin**0.5
            # margin = (f_[1] ** m)
            # loss = torch.clamp(1 + prediction_, 0.) ** 2 + torch.clamp(1 - label_, 0.) ** 2 + torch.clamp(
            #     prediction_ - label_ + 1, 0.) ** 2
            loss=0

            return True, label_, prediction_, 1, margin, loss

        else:
            return False, None, None, 0, 0, 0
    elif sum(s_) == 0 and sum(c_) > 0:
        if c_[1] == 0 and f_[1] > 0. and q_[1] > 0.5 and c_[0]>0:
            # margin =  (dist_factor ** (1 + m - (f_[1] ** m))) * (f_[1] ** m)
            # margin=(dist_factor ** (1 + m - (q_[1] ** m))) * (q_[1] ** m)
            margin=(dist_factor ** (1 + m - (q_[1] ** m))) * (q_[1] ** m)*(1-dist_factor)**0.2
            margin=margin**0.5
            # loss = torch.clamp(1 + prediction_, 0.) ** 2 + torch.clamp(1 - label_, 0.) ** 2 + torch.clamp(
            #     prediction_ - label_ + 1, 0.) ** 2
            loss=0
            # margin = (f_[1] ** m)

            return True, label_, prediction_, 1, 1, loss
        elif c_[0] == 0 and f_[0] > 0. and q_[0] > 0.5 and c_[1]>0:
            # margin =  (dist_factor ** (1 + m - (f_[1] ** m))) * (f_[1] ** m)
            margin=(dist_factor ** (1 + m - (q_[1] ** m))) * (q_[1] ** m)
            # loss = torch.clamp(1 + prediction_, 0.) ** 2 + torch.clamp(1 - label_, 0.) ** 2 + torch.clamp(
            #     prediction_ - label_ + 1, 0.) ** 2
            loss=0
            margin = (f_[0] ** m)


            return True, prediction_,label_, 1, 1, loss
        else:
            return False, None, None, 0, 0, 0
    else:
        return False, None, None, 0, 0, 0
def balanced_sampling(values, mask=None, exponent=2.0, balance_indicator=1.0):
    with torch.no_grad():
        max_ = values.max().item()
        min_ = values.min().item()
        range_ = max_ - min_


        if not range_>0.:
            # print(Fore.RED,'Warning: mode collapse detected',Fore.RESET)
            selection_probability=torch.rand_like(values)
        else:
            pivot_point = np.sqrt(np.abs(balance_indicator)) * np.sign(balance_indicator)
            # xa = ((max_ - values) / range_) * pivot_point
            xa=(1-values)* pivot_point
            selection_probability = ((1 - pivot_point) / 2 + xa + 0.5 * (1 - abs(pivot_point)))
            selection_probability = selection_probability ** exponent

        # print(f'max={selection_probability.max()}, min={selection_probability.min()}')

        if mask is None:
            dist = Categorical(probs=selection_probability)
        else:
            dist = MaskedCategorical(probs=selection_probability, mask=mask)

        target_index = dist.sample()

        return target_index

class TrainGraspGAN:
    def __init__(self, n_samples=None, epochs=1, learning_rate=5e-5):
        self.n_samples = n_samples
        self.size = n_samples
        self.epochs = epochs
        self.learning_rate = learning_rate

        '''model wrapper'''
        self.gan = self.prepare_model_wrapper()
        # self.ref_generator = self.initialize_ref_generator()
        self.data_loader = None

        '''Moving rates'''
        self.moving_collision_rate = None
        self.moving_firmness = None
        self.moving_out_of_scope = None
        self.relative_sampling_timing = None
        self.superior_A_model_moving_rate = None
        self.moving_anneling_factor = None
        # self.moving_scores_std = None

        self.bin_collision_statistics = None
        self.objects_collision_statistics = None
        self.suction_head_statistics = None
        self.shift_head_statistics = None
        self.suction_sampler_statistics = None
        self.background_detector_statistics = None

        '''initialize statistics records'''
        self.gripper_sampler_statistics = None
        self.critic_statistics = None
        self.data_tracker = None

        self.sampling_centroid = None
        self.diversity_momentum = 1.0

        # self.sample_from_latent = True

        # self.action_net = None
        # self.policy_net=None
        # self.load_action_model()
        # self.load_grasp_policy()

        self.freeze_approach = False
        self.freeze_beta = False
        self.freeze_distance = False
        self.freeze_width = False



    def generate_random_beta_dist_widh(self, size):
        sampled_approach = (torch.rand((size, 2), device='cuda') - 0.5)  # *1.5
        # sampled_approach*=torch.abs(sampled_approach)
        ones_ = torch.ones_like(sampled_approach[:, 0:1])
        sampled_approach = torch.cat([sampled_approach, ones_], dim=1)

        verticle = torch.zeros((size, 3), device='cuda')
        verticle[:, -1] += 1
        # sampled_approach=verticle
        sampled_approach = sampled_approach * 0.5 + verticle * 0.5
        sampled_approach = F.normalize(sampled_approach, dim=1)

        sampled_beta = (torch.rand((size, 2), device='cuda') - 0.5) * 2
        sampled_beta = F.normalize(sampled_beta, dim=1)

        sampled_dist = torch.distributions.LogNormal(loc=-1.337, scale=0.791)
        sampled_dist = sampled_dist.sample((size, 1)).cuda()  # *(0.1+0.9*(1-self.relative_sampling_timing.val))

        # sampled_dist/=5

        # sampled*=1-self.an

        # sampled_dist = torch.rand((size, 1), device='cuda') ** 5
        # sampled_dist[sampled_dist > 0.7] /= 2
        # sampled_dist[sampled_dist < 0.1] += 0.1
        sampled_width = torch.distributions.LogNormal(loc=-1.312, scale=0.505)
        sampled_width = 1. - sampled_width.sample((size, 1)).cuda()
        # sampled_width = 1 - torch.rand((size, 1), device='cuda') ** 5
        # sampled_width[sampled_width < 0.3] *= 3

        sampled_pose = torch.cat([sampled_approach, sampled_beta, sampled_dist, sampled_width], dim=1)

        return sampled_pose

    def pose_noisfication(self, gripper_pose, objects_mask, pc, bin_mask, altered_objects_elevation, mask):
        assert objects_mask.sum() > 0

        ref_pose = gripper_pose.detach().clone()

        ref_pose[:, 5:, ...] = torch.clamp(ref_pose[:, 5:, ...], 0.01, 0.99)

        # p=max(self.moving_collision_rate.val,self.moving_out_of_scope.val,1-self.superior_A_model_moving_rate.val)**2.0
        p = (1 - self.superior_A_model_moving_rate.val)#**2
        # if not altered_objects_elevation:
        #     ref_pose=ref_pose.permute(0, 2, 3, 1)
        #     correction_ratios_ = (2 * (1 / (1 + torch.rand_like(ref_pose[0,:,:,-2][mask] * (1 - p)) / (
        #                 torch.rand_like(ref_pose[0,:,:,-2][mask]) * p)))) ** 2
        #     ref_pose[0,:,:,-2][mask]*=(1-correction_ratios_)+correction_ratios_*(torch.abs(torch.from_numpy(pc[:, -1] - pc[~bin_mask][:, -1].min()).cuda()*10)**0.3)
        #
        #     correction_ratios_ = (2 * (1 / (1 + torch.rand_like(ref_pose[0, :, :, -1][mask] * (1 - p)) / (
        #             torch.rand_like(ref_pose[0, :, :, -1][mask]) * p)))) ** 2
        #     ref_pose[0,:,:,-1][mask]=(1-correction_ratios_)*ref_pose[0,:,:,-1][mask]+correction_ratios_
        #
        #     correction_ratios_ = (2 * (1 / (1 + torch.rand_like(ref_pose[0, :, :, 0:3][mask] * (1 - p)) / (
        #             torch.rand_like(ref_pose[0, :, :, 0:3][mask]) * p)))) ** 2
        #
        #     ref_approach=torch.zeros_like(ref_pose[0, :, :, 0:3][mask])
        #     ref_approach[:,-1]+=1.0
        #     ref_pose[0, :, :, 0:3][mask] = (1 - correction_ratios_) * ref_pose[0, :, :, 0:3][
        #         mask] + correction_ratios_*ref_approach
        #
        #     ref_pose=ref_pose.permute(0, 3, 1, 2)

        sampled_pose = self.generate_random_beta_dist_widh(ref_pose[:, 0, ...].numel())
        sampled_pose = reshape_for_layer_norm(sampled_pose, camera=camera, reverse=True)

        print(Fore.CYAN, f'Noise ratio p={p}', Fore.RESET)

        # sampling_ratios= (2*(1/(1+(torch.rand_like(ref_pose)*(1-p))/(torch.rand_like(ref_pose)*p))))#**2
        # sampling_ratios = torch.rand_like(ref_pose) * p
        sampling_ratios = 1/(1+((1-p)*torch.rand_like(ref_pose)) /(p*torch.rand_like(ref_pose)))
        # sampling_ratios=sampling_ratios*p+(1-p)*torch.rand_like(ref_pose)

        sampled_pose = sampled_pose.detach().clone() * sampling_ratios + (1 - sampling_ratios) * ref_pose

        sampled_pose[:, 3:5, ...] = F.normalize(sampled_pose[:, 3:5, ...], dim=1)
        sampled_pose[:, 0:3, ...] = F.normalize(sampled_pose[:, 0:3, ...], dim=1)

        # sampled_pose[:, 5, ...]+=0.01

        sampled_pose[:, 5:, ...] = torch.clamp(sampled_pose[:, 5:, ...], 0.01, 0.99)
        return sampled_pose

    def initialize(self, n_samples=None):
        self.n_samples = n_samples
        self.prepare_data_loader()

        '''Moving rates'''

        self.gradient_moving_rate = MovingRate(action_module_with_GAN_key + '_gradient', decay_rate=0.01, initial_val=1000.)

        self.moving_collision_rate = MovingRate(action_module_with_GAN_key + '_collision', decay_rate=0.01,
                                                initial_val=1.)
        self.moving_firmness = MovingRate(action_module_with_GAN_key + '_firmness', decay_rate=0.01, initial_val=0.)
        self.moving_out_of_scope = MovingRate(action_module_with_GAN_key + '_out_of_scope', decay_rate=0.01,
                                              initial_val=1.)
        self.relative_sampling_timing = MovingRate(action_module_with_GAN_key + '_relative_sampling_timing',
                                                   decay_rate=0.01,
                                                   initial_val=1.)
        self.superior_A_model_moving_rate = MovingRate(action_module_with_GAN_key + '_superior_A_model',
                                                       decay_rate=0.001,
                                                       initial_val=0.)
        self.moving_anneling_factor = MovingRate(action_module_with_GAN_key + '_anneling_factor', decay_rate=0.1,
                                                 initial_val=0.)

        # self.moving_scores_std = MovingRate(action_module_with_GAN_key + '_scores_std', decay_rate=0.01,
        #                                     initial_val=.01)

        '''initialize statistics records'''
        self.suction_head_statistics = TrainingTracker(name=action_module_with_GAN_key + '_suction_head',
                                                       track_label_balance=True)
        self.bin_collision_statistics = TrainingTracker(name=action_module_with_GAN_key + '_bin_collision',
                                                        track_label_balance=True)
        self.objects_collision_statistics = TrainingTracker(name=action_module_with_GAN_key + '_objects_collision',
                                                            track_label_balance=True)
        self.shift_head_statistics = TrainingTracker(name=action_module_with_GAN_key + '_shift_head',
                                                     track_label_balance=True)

        self.suction_sampler_statistics = TrainingTracker(name=action_module_with_GAN_key + '_suction_sampler',
                                                          track_label_balance=False)

        self.background_detector_statistics = TrainingTracker(name=action_module_with_GAN_key + '_background_detector',
                                                              track_label_balance=False)


        self.gripper_sampler_statistics = TrainingTracker(name=action_module_with_GAN_key + '_gripper_sampler',
                                                          track_label_balance=False)

        self.critic_statistics = TrainingTracker(name=action_module_with_GAN_key + '_critic',
                                                 track_label_balance=False)

        self.data_tracker = DataTracker(name=gripper_grasp_tracker)

    def prepare_data_loader(self):
        file_ids = training_buffer.get_indexes()

        print(Fore.CYAN, f'Buffer size = {len(file_ids)}', Fore.RESET)
        dataset = GraspGANDataset2(data_pool=training_buffer, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers,
                                                  shuffle=True)
        self.size = len(dataset)
        self.data_loader = data_loader

    def prepare_model_wrapper(self):
        '''load  models'''
        gan = GANWrapper(action_module_with_GAN_key, ActionNet, D)
        gan.ini_models(train=True)


        gan.critic_adamW_optimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.999,weight_decay_=0)
        # gan.critic_sgd_optimizer(learning_rate=self.learning_rate*1,momentum=0.)
        gan.generator_adamW_optimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999)
        # gan.generator_sgd_optimizer(learning_rate=self.learning_rate*10,momentum=0., weight_decay_=0.)

        return gan

    def simulate_elevation_variations(self, original_depth, objects_mask, max_elevation=0.15, exponent=2.0):
        '''Elevation-based Augmentation'''
        shift_entities_mask = objects_mask & (original_depth > 0.0001) if np.random.random() > 0.5 else (
                    original_depth > 0.0001)
        new_depth = original_depth.clone().detach()
        new_depth[shift_entities_mask] -= max_elevation * (np.random.rand() ** exponent) * camera.scale

        return new_depth

    def analytical_bin_mask(self, pc, file_ids):
        try:
            bin_mask, floor_elevation = bin_planes_detection(pc, sides_threshold=0.005, floor_threshold=0.0015,
                                                             view=False,
                                                             file_index=file_ids[0], cache_name='bin_planes2')
            return bin_mask, floor_elevation
        except Exception as error_message:
            print('bin mask generation error:', file_ids[0])
            print(error_message)
            return None, None
    def step_discriminator_learning(self, depth, pc, mask, bin_mask, gripper_pose, gripper_pose_ref,
                                    altered_objects_elevation, valid_pose_index, floor_elevation):
        '''self supervised critic learning'''
        with torch.no_grad():
            generated_grasps_cat = torch.cat([gripper_pose, gripper_pose_ref], dim=0)
            # depth_cat = depth.repeat(2, 1, 1, 1)
        critic_score, margin_params = self.gan.critic(depth.clone(), generated_grasps_cat,mask[None,None,...])

        # cuda_memory_report()

        tracked_indexes = []

        gripper_pose2 = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask]
        ref_gripper_pose2 = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :][mask]
        gen_scores_ = critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        ref_scores_ = critic_score.permute(0, 2, 3, 1)[1, :, :, 0][mask]

        # mean_seperation = torch.abs(critic_score[0] - critic_score[1])
        # self.moving_scores_std.update(mean_seperation.std().item() * 2 + mean_seperation.mean().item())
        # tou = 1 - self.superior_A_model_moving_rate.val
        # margin_factor = (1 - tou ** 2) * self.moving_scores_std.val + tou ** 2

        # margin_params= margin_params.permute(0, 2, 3, 1)[0, :, :, 0][mask]

        selection_mask = (~bin_mask) & (ref_scores_ != 0)

        # score_delta_mean=torch.abs(ref_scores_-gen_scores_)[selection_mask].mean().item()
        # print(f'score_delta_mean={score_delta_mean}')

        if self.sampling_centroid is None:
            selection_p = torch.rand_like(ref_gripper_pose2[:, 0])
        else:
            ref_gripper_pose2_=ref_gripper_pose2.detach().clone()
            ref_gripper_pose2_[:,5:]=torch.clamp(ref_gripper_pose2_[:,5:],0.,1.)
            gripper_pose2_=gripper_pose2.detach().clone()
            gripper_pose2_[:,5:]=torch.clamp(gripper_pose2_[:,5:],0.,1.)

            def norm_(gamma,expo_=1.0):
                gamma = (gamma - gamma.min()) / (
                            gamma.max() - gamma.min())
                gamma = gamma ** expo_
                return gamma

            gamma_dive = norm_((1.001 - F.cosine_similarity(gripper_pose2_[:,3:5],
                                               self.sampling_centroid[None, 3:5], dim=-1))/2,1)


            # gamma_firmness=torch.clamp(ref_gripper_pose2_[:,-2].detach().clone(),0.001,0.99)**0.3
            gamma_rand=torch.rand_like(gamma_dive)

            gamma_dive=norm_(gamma_dive)

            range=pc[~bin_mask][:,-1].max()-pc[~bin_mask][:,-1].min()
            d_gamma=norm_(1-torch.abs((pc[:,-1]-pc[~bin_mask][:,-1].min())/range),2.0)


            selection_p = (gamma_dive *   gamma_rand * d_gamma) ** (1 / 4)


            assert not torch.isnan(selection_p).any(), f'selection_p: {ref_gripper_pose2}'

        avaliable_iterations = selection_mask.sum()

        n = int(min(max_n, avaliable_iterations))

        loss_terms_counters = [0, 0, 0, 0, 0]
        col_loss_counter = 0
        firm_loss_counter = 0

        counted_superior = []
        counted_inferior = []
        margin_lis = []
        # instance_losses = []
        t = 0
        while True:
            if t > n: break
            dist = MaskedCategorical(probs=selection_p, mask=selection_mask)
            if t == 0 and valid_pose_index is not None:
                target_index = valid_pose_index
            else:
                target_index = dist.sample().item()

            selection_mask[target_index] = False
            avaliable_iterations -= 1
            target_point = pc[target_index]

            target_generated_pose = gripper_pose2[target_index]
            target_ref_pose = ref_gripper_pose2[target_index]
            label_ = ref_scores_[target_index]
            prediction_ = gen_scores_[target_index]

            c_, s_, f_, q_ = evaluate_grasps3(target_point, target_generated_pose, target_ref_pose, pc, bin_mask,
                                              visualize=False, floor_elevation=floor_elevation)

            if (s_[0] == 0) and ((c_[0] == 0 and f_[0] > 0) or (s_[1] == 0 and c_[1] == 0 and f_[1] > 0)):
                self.moving_collision_rate.update(int(c_[0] > 0))
            if sum(s_) == 0 and sum(c_) == 0:
                self.moving_firmness.update(int(f_[0] > f_[1]))

            if s_[1] == 0 and c_[1] == 0 and f_[1] > 0:
                self.moving_out_of_scope.update(int(s_[0] > 0))

            # prediction_uniqness=  ((1 - F.cosine_similarity(target_generated_pose.detach().clone()[None, :],
            #                                                                 self.sampling_centroid[None, :], dim=-1)) / 2).item()**0.5    if      self.sampling_centroid is not None else 0.
            prediction_uniqness = 1.0
            dist_factor = ((1 + F.cosine_similarity(target_generated_pose.detach().clone()[None, 0:5],
                                                    target_ref_pose.detach().clone()[None, 0:5],
                                                    dim=-1)) / 2).item() ** 0.5

            # dist_factor=4*dist_factor*(1-dist_factor)

            if (s_[0] > 0. or c_[0] > 0. or f_[0] == 0) and s_[1] == 0 and c_[1] == 0 and f_[1] > 0:
                self.superior_A_model_moving_rate.update(0.)
            elif (s_[1] > 0. or c_[1] > 0. or f_[1] == 0) and s_[0] == 0 and c_[0] == 0 and f_[0] > 0:
                self.superior_A_model_moving_rate.update(1.0)

            counted, superior, inferior, margin, margin2, loss = collision_loss(c_, s_, f_, q_, prediction_,
                                                                                label_, prediction_uniqness,
                                                                                dist_factor)

            if counted:
                col_loss_counter += 1

                if t == 0 and valid_pose_index is not None:
                    print(Fore.GREEN)

                print(target_ref_pose.detach().cpu().numpy(), '--', target_generated_pose.detach().cpu().numpy(), 'pred=',
                      prediction_.item(), 'lable=', label_.item())

                if t == 0 and valid_pose_index is not None:
                    print(Fore.RESET)

            if not counted:
                '''firmness loss'''
                counted, superior, inferior, margin, margin2, loss = firmness_loss(c_, s_, f_, q_, prediction_,
                                                                                   label_, prediction_uniqness,
                                                                                   dist_factor)
                if counted:
                    firm_loss_counter += 1
                    print(Fore.LIGHTGREEN_EX, target_ref_pose.detach().cpu().numpy(), '--',
                          target_generated_pose.detach().cpu().numpy(), 'pred=', prediction_.item(), 'lable=', label_.item(),
                          Fore.RESET)

            if counted:
                t = 1
                counted_inferior.append(inferior)
                counted_superior.append(superior)
                margin_lis.append(margin)
                # instance_losses.append(loss)

                hh = ((col_loss_counter + firm_loss_counter) / batch_size) ** 2
                n = int(min(hh * max_n + n, avaliable_iterations))

                avoid_collision = (s_[0] > 0. or c_[0] > 0.)
                A_is_collision_free = None
                A_is_more_firm = None

                if (s_[0] > 0. or c_[0] > 0. or f_[0] == 0 or q_[0] < 0.5) and s_[1] == 0 and c_[1] == 0:
                    A_is_collision_free = False
                elif (s_[1] > 0. or c_[1] > 0.) and s_[0] == 0 and c_[0] == 0:
                    A_is_collision_free = True
                elif sum(c_) == 0 and sum(s_) == 0:
                    if f_[1] > f_[0]:
                        A_is_more_firm = False
                    elif f_[0] > f_[1]:
                        A_is_more_firm = True
                tracked_indexes.append((target_index, avoid_collision, A_is_collision_free, A_is_more_firm, margin))

                if self.sampling_centroid is None:
                    self.sampling_centroid = target_generated_pose.detach().clone()
                else:
                    diffrence = ((1 - F.cosine_similarity(target_generated_pose.detach().clone()[None, :],
                                                          self.sampling_centroid[None, :], dim=-1)) / 2) ** 2.0
                    self.diversity_momentum = self.diversity_momentum * 0.99 + diffrence.item() * 0.01
                    self.sampling_centroid = self.sampling_centroid * 0.99 + target_generated_pose.detach().clone() * 0.01

            if col_loss_counter + firm_loss_counter == batch_size:
                self.relative_sampling_timing.update((t + 1) / n)
                break

            t += 1

        if col_loss_counter + firm_loss_counter == batch_size:
            print(loss_terms_counters)

            # neagtives=torch.clamp(1+torch.stack(counted_inferior),0.).mean()
            # positives=torch.clamp(1-torch.stack(counted_superior),0.).mean()
            # loss=(neagtives+positives)
            margins = torch.tensor(margin_lis).cuda()
            # margins = margins / margins.sum()
            inferiors = torch.stack(counted_inferior) #* margins
            superiors = torch.stack(counted_superior) #* margins

            # loss = (inferiors.sum() - superiors.sum())


            # loss=loss+gp
            # loss=((torch.clamp(inferiors.mean()-superiors+1,0.))*margins).sum()+((torch.clamp(inferiors-superiors.mean()+1,0.))*margins).sum()

            loss=(torch.clamp(inferiors - superiors + margins, 0.)**2).mean()
            # loss=((torch.clamp(inferiors-superiors+margins,0.))).mean()

            # margins=torch.tensor(margin_lis).cuda()#*0.5
            # curc_mask=(superiors-inferiors>1.0).float()
            # inferiors = torch.stack(counted_inferior).detach()
            # loss=hinge_contrastive_loss(positive_score=superiors,negative_score=inferiors,margin=margins,curc_mask=curc_mask).mean()

            # if loss > -0.3:
            # cuda_memory_report()
            print(Fore.LIGHTYELLOW_EX, f'c_loss={loss.item()}', '  scores mean =', inferiors.mean().item(), Fore.RESET)

            loss.backward()

            self.critic_statistics.loss = loss.item()

            norm = torch.nn.utils.clip_grad_norm_(self.gan.critic.parameters(), max_norm=1.)
            print(Fore.LIGHTGREEN_EX, f' D  norm : {norm}', Fore.RESET)

            self.gan.critic_optimizer.step()
            self.gan.critic_optimizer.zero_grad(set_to_none=True)
            # if (loss_terms_counters[0] + loss_terms_counters[1] + loss_terms_counters[3] == batch_size) or (loss_terms_counters[2]*w2+loss_terms_counters[4]*w2==batch_size):
            return True, tracked_indexes
            # else:
            #     return False, tracked_indexes
        else:
            print('pass, counter/Batch_size=', loss_terms_counters, '/', batch_size)
            self.gan.critic_optimizer.zero_grad(set_to_none=True)
            return False, tracked_indexes

    def soft_entropy_loss(self, values, bins=40, sigma=0.1, min_val=None, max_val=None):
        """
        Differential approximation of entropy oiver scaler output
        """
        N = values.size(0)

        '''create bin centers'''
        if min_val is None and max_val is None:
            min_val, max_val = values.min().item(), values.max().item()
        bin_centers = torch.linspace(min_val, max_val, steps=bins, device=values.device)
        bin_centers = bin_centers.view(1, -1)

        '''compute soft assignment via Gaussion kernel'''
        dists = (values - bin_centers).pow(2)  # [B, num_bins]
        soft_assignments = torch.exp(-dists / (2 * sigma ** 2))  # [B, num_bins]

        '''normalize per sample and then sum over batch'''
        probs = soft_assignments / (soft_assignments.sum(dim=1, keepdim=True) + 1e-8)
        avg_probs = probs.mean(dim=0)  # [num_bins], average histogram over batch
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
        # return -entropy
        max_entropy = torch.log(torch.tensor([bins], device=entropy.device).float())
        return (max_entropy - entropy) / max_entropy  # maximize entropy

    def ref_reg_loss(self, gripper_pose_ref2, objects_mask):
        width_scope_loss = (torch.clamp(gripper_pose_ref2[:, 6:7][objects_mask] - 1, min=0.) ** 2).mean()
        dist_scope_loss = (torch.clamp(-gripper_pose_ref2[:, 5:6][objects_mask] + 0.01, min=0.) ** 2).mean()
        width_scope_loss += (torch.clamp(-gripper_pose_ref2[:, 6:7][objects_mask], min=0.) ** 2).mean()
        dist_scope_loss += (torch.clamp(gripper_pose_ref2[:, 5:6][objects_mask] - 1, min=0.) ** 2).mean()

        # beta_entropy = self.soft_entropy_loss(gripper_pose_ref2[:, 3:4][objects_mask], bins=36, sigma=0.1, min_val=-1,
        #                                       max_val=1) ** 2
        # beta_entropy += self.soft_entropy_loss(gripper_pose_ref2[:, 4:5][objects_mask], bins=36, sigma=0.1, min_val=-1,
        #                                        max_val=1) ** 2

        beta_std = 0.5 - ((gripper_pose_ref2[:, 3:4][objects_mask] + 1) / 2).std()
        beta_std += 0.5 - ((gripper_pose_ref2[:, 4:5][objects_mask] + 1) / 2).std()

        # beta_entropy = torch.tensor(0., device=gripper_pose_ref2.device) if torch.isnan(beta_entropy) else beta_entropy

        loss = width_scope_loss * 100 + dist_scope_loss * 100 + (beta_std ** 2)

        return loss

    def get_generator_loss(self, depth, mask, gripper_pose, gripper_pose_ref, tracked_indexes, objects_mask):

        generated_grasps_cat = torch.cat([gripper_pose, gripper_pose_ref], dim=0)
        # depth_cat = depth.repeat(2, 1, 1, 1)
        critic_score, _ = self.gan.critic(depth.clone(), generated_grasps_cat,mask[None,None,...], detach_backbone=True)
        pred_scores_ = critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        ref_scores_ = critic_score.permute(0, 2, 3, 1)[1, :, :, 0][mask]

        gripper_pose2 = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask]
        gripper_pose_ref2=gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :][mask]

        # gripper_sampling_loss = torch.tensor(0., device=depth.device) * ref_scores_.mean() * 0.0
        direct_loss = torch.tensor(0., device=depth.device)

        # ref_generator_stockhastic_loss=(torch.clamp(pred_scores_.detach().clone()-ref_scores_,0.)[objects_mask]**2).mean()
        generator_stockhastic_loss = (
                torch.clamp(ref_scores_.detach().clone() - pred_scores_, 0.)[objects_mask] ** generator_expo).mean()
        # generator_stockhastic_loss=-(pred_scores_[objects_mask]).mean()

        counter = [0., 0., 0., 0.]
        l1 = torch.tensor(0., device=depth.device)
        l2 = torch.tensor(0., device=depth.device)
        l3 = torch.tensor(0., device=depth.device)
        l4 = torch.tensor(0., device=depth.device)

        pred_list = []
        label_list = []
        margin_list = []
        for j in range(len(tracked_indexes)):
            A_is_collision_free = tracked_indexes[j][2]

            if A_is_collision_free is not None:
                if  (counter[1] + counter[3] < batch_size):
                    target_index = tracked_indexes[j][0]
                    margin = tracked_indexes[j][4]
                    label = ref_scores_[target_index]
                    pred_ = pred_scores_[target_index]
                    pred_pose = gripper_pose2[target_index]
                    label_pose = gripper_pose_ref2[target_index]
                    # l1_loss = (torch.clamp(label.detach().clone() - pred_, 0.) ** 1)
                    # l1_loss = hinge_contrastive_loss(positive_score=pred_, negative_score=label.detach().clone(),
                    #                                  margin=0.)
                    # print('test--------------pred_=',pred_.item(),'--------label=',label.item(),'---------l=',l1_loss.item())

                    # l2 += (l1_loss ** generator_expo)  # +self.get_scope_loss(target_pose)
                    pred_list.append(pred_)
                    label_list.append(label)
                    # direct_loss+=(((pred_pose[-2]-label_pose[-2])**2.)+((pred_pose[-1]-label_pose[-1])**2.)*0.16+((1.00 - F.cosine_similarity(pred_pose[3:5],
                    #                        label_pose[3:5], dim=0)))*0.047+((1.00 - F.cosine_similarity(pred_pose[0:3],
                    #                        label_pose[0:3], dim=0)))*0.42)/batch_size

                    margin_list.append(torch.tensor([margin], device=pred_.device))
                    # # l2-=pred_
                    counter[1] += 1.

            for j in range(len(tracked_indexes)):
                A_is_more_firm = tracked_indexes[j][3]


                if A_is_more_firm is not None:


                    if (counter[1] + counter[3] < batch_size):
                        target_index = tracked_indexes[j][0]
                        label = ref_scores_[target_index]
                        pred_ = pred_scores_[target_index]
                        margin = tracked_indexes[j][4]
                        pred_pose = gripper_pose2[target_index]
                        label_pose = gripper_pose_ref2[target_index]
                        # l1_loss = (torch.clamp(label.detach().clone() - pred_, 0.) ** 1)
                        # l1_loss = hinge_contrastive_loss(positive_score=pred_, negative_score=label.detach().clone(),
                        #                                  margin=0.)

                        # l4 += (l1_loss ** generator_expo)  # +self.get_scope_loss(target_pose)
                        # l4 -= pred_
                        pred_list.append(pred_)
                        label_list.append(label)
                        margin_list.append(torch.tensor([margin], device=pred_.device))
                        # direct_loss +=  (((pred_pose[-2] - label_pose[-2]) ** 2.) + (
                        #             (pred_pose[-1] - label_pose[-1]) ** 2.) * 0.16 + (
                        #                          (1.00 - F.cosine_similarity(pred_pose[3:5],
                        #                                                      label_pose[3:5], dim=0))) * 0.047 + (
                        #                          (1.00 - F.cosine_similarity(pred_pose[0:3],
                        #                                                      label_pose[0:3],
                        #                                                      dim=0))) * 0.42) / batch_size
                        # weight_list.append(torch.tensor([weight], device=pred_.device))

                        counter[3] += 1.

        # if counter[0]+counter[2]==batch_size:
        # if counter[0] >0: gripper_sampling_loss += l1 / (counter[0])
        # if counter[2] >0: gripper_sampling_loss += l3 / counter[2]

        # if counter[1]==G_batch_size:
        #     gripper_sampling_loss += l2 / (counter[1])
        # weights=torch.concatenate(weight_list)

        # gripper_sampling_loss=-(torch.stack(pred_list)*weights).sum()/weights.sum()
        # gripper_sampling_loss = -pred_scores_[objects_mask].mean()
        # margins = torch.tensor(margin_list).cuda()
        # margins = margins / margins.sum()
        # gripper_sampling_loss = -(torch.stack(pred_list) * margins).sum()
        pred=torch.stack(pred_list)
        label=torch.stack(label_list)
        # assert len(label_list)==batch_size, f'{len(label_list)},  {len(tracked_indexes)}'
        #
        # gripper_sampling_loss = (torch.clamp(label - pred+margins ,0.)).mean()#+direct_loss
        gripper_sampling_loss = (torch.clamp(label - pred ,0.)**1).mean()#+direct_loss

        # gripper_sampling_loss = (((torch.clamp(label.mean() - pred +  1,0.))*margins).sum()+
        #                          ((torch.clamp(label - pred.mean() +  1,0.))*margins).sum())

        # if counter[1]+counter[3]==batch_size:
        #     if counter[1] >0:
        #         gripper_sampling_loss += l2 / (counter[1])
        #         # print('----------------------',gripper_sampling_loss.item())
        #     if counter[3] >0:
        #         gripper_sampling_loss += l4 / counter[3]
        # pred=torch.stack(pred_list)
        # label=torch.stack(label_list)
        # # margins=torch.stack(margin_list)#*0.5
        # gripper_sampling_loss = hinge_contrastive_loss(positive_score=pred, negative_score=label,margin=0).mean()

        # if counter[1] ==G_batch_size: gripper_sampling_loss += l2 / (counter[1])
        # elif counter[1]+counter[3]==G_batch_size: gripper_sampling_loss += (l2+l4) / (counter[1]+counter[3])
        # elif counter[3] ==G_batch_size: gripper_sampling_loss += l4 / (counter[3])

        # print(f'G loss weights: {counter}, gripper_sampling_loss={gripper_sampling_loss.item()}')

        # l = avoid_collision * (
        #         torch.clamp(label - pred_, 0.) ** 2)
        # l=smooth_l1_loss(l,torch.zeros_like(l))
        # gripper_sampling_loss += l / len(tracked_indexes)

        return gripper_sampling_loss, generator_stockhastic_loss

    def pixel_to_point_index(self, depth, mask, gripper_pixel_index):
        tmp = torch.zeros_like(depth)[0, 0]
        tmp[gripper_pixel_index[0], gripper_pixel_index[1]] = 1.0
        tmp = tmp[mask]

        point_index = (tmp == 1).nonzero(as_tuple=False)
        return point_index.item()

    def pick_trainable_grasp_parameters(self):
        if np.random.random() > 0.5:
            v_ = np.random.random()
            self.freeze_approach = False if v_ < 0.35 else True
            self.freeze_beta = False if 0.75 > v_ > 0.35 else True
            self.freeze_width = False if v_ > 0.75 else True
        else:
            self.freeze_approach = False
            self.freeze_beta = False
            self.freeze_width = False

        print(Fore.CYAN,
              f'freeze_approach={self.freeze_approach}, freeze_beta={self.freeze_beta}, freeze_width={self.freeze_width}',
              Fore.RESET)

    def get_scope_loss(self, pose):
        dist_scope_loss = (torch.clamp(pose[-2] - 0.99, min=0.) ** 2)
        width_scope_loss = (torch.clamp(pose[-1] - 0.99, min=0.) ** 2)
        dist_scope_loss += (torch.clamp(-pose[-2] + 0.01, min=0.) ** 2)
        width_scope_loss += (torch.clamp(-pose[-1] + 0.01, min=0.) ** 2)

        return width_scope_loss + dist_scope_loss

    def multi_head_loss(self,pc,depth,mask,file_ids,predicted_normal,pred_griper_collision_classifier,pred_gripper_poses
                        ,suction_quality_classifier,shift_affordance_classifier,background_class):
        '''background detection head'''
        bin_mask, floor_elevation_ = self.analytical_bin_mask(pc, file_ids)
        if bin_mask is None: return None,None,None,None,None
        objects_mask = bin_mask <= 0.5
        # objects_mask = torch.from_numpy(objects_mask_numpy).cuda()
        objects_mask_pixel_form = torch.ones_like(depth)
        objects_mask_pixel_form[0, 0][mask] = objects_mask_pixel_form[0, 0][mask] * objects_mask
        objects_mask_pixel_form = objects_mask_pixel_form > 0.5

        '''loss computation'''
        suction_seal_loss = torch.tensor(0., device=depth.device)
        gripper_collision_loss = torch.tensor(0., device=depth.device)
        shift_appealing_loss = torch.tensor(0., device=depth.device)
        background_detection_loss = torch.tensor(0., device=depth.device)
        suction_sampling_loss = torch.tensor(0., device=depth.device)

        suction_sampling_loss += suction_sampler_loss(pc, predicted_normal.permute(0, 2, 3, 1)[0][mask], objects_mask,
                                                      file_index=file_ids[0])

        gripper_poses1 = pred_gripper_poses[0].permute(1, 2, 0)[mask].detach()

        suction_head_predictions = suction_quality_classifier[0, 0][mask]
        gripper_head_predictions1 = pred_griper_collision_classifier[0, :].permute(1, 2, 0)[mask]
        shift_head_predictions = shift_affordance_classifier[0, 0][mask]
        background_class_predictions = background_class.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        normals = predicted_normal[0].permute(1, 2, 0)[mask].detach().cpu().numpy()

        label = bin_mask.float()
        background_detection_loss += bce_loss(background_class_predictions, label)
        self.background_detector_statistics.update_confession_matrix(label,
                                                                     background_class_predictions.detach())

        for k in range(batch_size2):
            '''gripper-object collision - 1'''
            while True:
                gripper_target_index = balanced_sampling(gripper_head_predictions1[:, 0], mask=objects_mask,
                                                         exponent=10.0,
                                                         balance_indicator=self.objects_collision_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = gripper_head_predictions1[gripper_target_index]
                gripper_target_pose = gripper_poses1[gripper_target_index]
                loss, counted = gripper_object_collision_loss(gripper_target_pose, gripper_target_point, pc,
                                                              objects_mask, gripper_prediction_,
                                                              self.objects_collision_statistics)
                if counted: break
            gripper_collision_loss += loss / batch_size2

        for k in range(batch_size2):
            '''gripper-bin collision - 1'''
            while True:
                gripper_target_index = balanced_sampling(gripper_head_predictions1[:, 1], mask=objects_mask,
                                                         exponent=10.0,
                                                         balance_indicator=self.bin_collision_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = gripper_head_predictions1[gripper_target_index]
                gripper_target_pose = gripper_poses1[gripper_target_index]
                loss, counted = gripper_bin_collision_loss(gripper_target_pose, gripper_target_point, pc,
                                                           objects_mask, gripper_prediction_,
                                                           self.bin_collision_statistics, floor_elevation_)

                if counted: break
            gripper_collision_loss += loss / batch_size2

        for k in range(batch_size2):
            '''suction seal head'''
            suction_target_index = balanced_sampling(suction_head_predictions, mask=objects_mask,
                                                     exponent=10.0,
                                                     balance_indicator=self.suction_head_statistics.label_balance_indicator)
            suction_prediction_ = suction_head_predictions[suction_target_index]
            suction_seal_loss += get_suction_seal_loss(pc.cpu().numpy(), normals, suction_target_index, suction_prediction_,
                                                       self.suction_head_statistics, objects_mask) / batch_size2

        for k in range(batch_size2):
            '''shift affordance head'''
            shift_target_index = balanced_sampling(shift_head_predictions, mask=None, exponent=10.0,
                                                   balance_indicator=self.shift_head_statistics.label_balance_indicator)
            shift_target_point = pc[shift_target_index]
            shift_prediction_ = shift_head_predictions[shift_target_index]
            shift_appealing_loss += shift_affordance_loss(pc.cpu().numpy(), shift_target_point.cpu().numpy(), objects_mask.cpu().numpy(),
                                                          self.shift_head_statistics, shift_prediction_) / batch_size2

        return suction_seal_loss, gripper_collision_loss, shift_appealing_loss, background_detection_loss, suction_sampling_loss

    def visualize(self, depth):
        with torch.no_grad():
            '''get parameters'''
            pc, mask = depth_to_point_clouds(depth[0, 0], camera)
            pc = transform_to_camera_frame_torch(pc, reverse=True)

            # downsampling_mask=np.random.random(pc.shape[0])>0.5

            elevation_mask = (pc[:, -1] < 0.15)  # & downsampling_mask
            cloned_mask=mask.clone()
            mask[cloned_mask] = (mask[cloned_mask]) & elevation_mask
            pc = pc[elevation_mask]

            '''generated grasps'''
            gripper_pose, suction_direction, griper_collision_classifier, suction_quality_classifier, shift_affordance_classifier, background_class = self.gan.generator(
                depth.clone(),mask[None,None,...], detach_backbone=True)

            # print(self.gan.generator.gripper_sampler.dist_biased_tanh.b.data)
            # # print(self.gan.generator.gripper_sampler.dist_biased_tanh.k.data)
            #
            # print(self.gan.generator.gripper_sampler.width_biased_tanh.b.data)
            # # print(self.gan.generator.gripper_sampler.width_biased_tanh.k.data)
            #
            # exit()
            # bin_mask = bin_planes_detection(pc, sides_threshold=0.005, floor_threshold=0.0015, view=True,
            #                                 file_index=file_ids[0], cache_name='bin_planes2')

            suction_head_predictions = suction_quality_classifier[0, 0][mask]
            shift_head_predictions = shift_affordance_classifier[0, 0][mask].detach().cpu().numpy()
            background_class_predictions = background_class.permute(0, 2, 3, 1)[0, :, :, 0][mask]
            objects_mask = background_class_predictions.detach().cpu().numpy() <= .5
            gripper_poses = gripper_pose[0].permute(1, 2, 0)[mask]  # .cpu().numpy()
            collision_with_objects_predictions = griper_collision_classifier[0, 0][mask]
            collision_with_bin_predictions = griper_collision_classifier[0, 1][mask]
            gripper_sampling_mask = (collision_with_objects_predictions < .7)

            # normal_view_mask=objects_mask& (suction_head_predictions.cpu().numpy()>0.5)
            # normals=suction_direction[0].permute(1,2,0)[mask].cpu().numpy()

            # colors=np.ones_like(pc)*[0.52, 0.8, 0.92]
            # custom_normal_open3d_view(pc=pc, normals=normals,normal_mask=normal_view_mask,color=colors,view_coordinate=False)

            # print(pc[objects_mask].min())
            # print(pc[~objects_mask].min())
            # print(pc[objects_mask].max())
            # print(pc[~objects_mask].max())
            # exit()
            # view_image(depth[0,0].cpu().numpy().astype(np.float64))

            # view_image(background_class[0, 0].detach().cpu().numpy().astype(np.float64))
            # sampling_p=1.-torch.sqrt(collision_with_objects_predictions*collision_with_bin_predictions)
            sampling_p = 1. - collision_with_objects_predictions #** 2.0
            # gripper_poses[:,-2]*=0
            # gripper_poses[:,-1]=1
            dense_grasps_visualization(pc, gripper_poses,
                                       view_mask=(gripper_sampling_mask & torch.from_numpy(objects_mask).cuda()),
                                       sampling_p=sampling_p, view_all=False,exclude_collision=True)

            dense_grasps_visualization(pc, gripper_poses,
                                       view_mask=( torch.from_numpy(objects_mask).cuda()),
                                       sampling_p=None, view_all=False, exclude_collision=True)

            # gripper_poses[:,3:5]*=-1
            # dense_grasps_visualization(pc, gripper_poses, view_mask=gripper_sampling_mask&torch.from_numpy(objects_mask).cuda(),sampling_p=sampling_p,view_all=False)

            suction_head_predictions[~torch.from_numpy(objects_mask).cuda()] *= 0.

            # view_mask(pc, background_class_predictions, pivot=0.5)
            # view_mask(pc, suction_head_predictions, pivot=0.5)
            # view_mask(pc, shift_head_predictions, pivot=0.5)
            # view_mask(pc, collision_with_objects_predictions, pivot=0.5)
            # view_mask(pc, collision_with_bin_predictions, pivot=0.5)

    def begin(self):

        pi = progress_indicator('Begin new training round: ', max_limit=len(self.data_loader))
        gripper_pose = None
        for i, batch in enumerate(self.data_loader, 0):
            depth, file_ids, pose_7, valid_gripper_pose, gripper_pixel_index = batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pose_7 = pose_7.cuda().float()[0]
            gripper_pixel_index = gripper_pixel_index[0]
            valid_gripper_pose = valid_gripper_pose[0]
            valid_pose_index = None
            pose_7[-2] = pose_7[-2] * (np.random.rand() ** 2)

            # self.visualize(depth)
            # continue

            pi.step(i)

            pc, mask = depth_to_point_clouds(depth[0, 0], camera)
            pc = transform_to_camera_frame_torch(pc, reverse=True)

            if valid_gripper_pose:
                valid_pose_index = self.pixel_to_point_index(depth, mask, gripper_pixel_index)

            '''background detection head'''
            bin_mask, floor_elevation = self.analytical_bin_mask(pc, file_ids)
            if bin_mask is None: continue
            objects_mask = bin_mask <= 0.5
            # objects_mask = torch.from_numpy(objects_mask_numpy).cuda()
            objects_mask_pixel_form = torch.ones_like(depth)
            objects_mask_pixel_form[0, 0][mask] = objects_mask_pixel_form[0, 0][mask] * objects_mask
            objects_mask_pixel_form = objects_mask_pixel_form > 0.5

            # view_features(reshape_for_layer_norm(objects_mask_pixel_form, camera=camera, reverse=False))

            '''Elevation-based augmentation'''
            if np.random.rand() > 1.:
                depth = self.simulate_elevation_variations(depth, objects_mask_pixel_form, exponent=5.0)
                pc, mask = depth_to_point_clouds(depth[0, 0], camera)
                pc = transform_to_camera_frame(pc, reverse=True)
                altered_objects_elevation = True
            else:
                altered_objects_elevation = False

            for k in range(max_samples_per_image):
                # cuda_memory_report()
                if k > 0: valid_pose_index = None
                with torch.no_grad():
                    gripper_pose,_,_,_,_,_ = self.gan.generator(
                        depth.clone(),mask[None,None,...], detach_backbone=True)  # [1,7,h,w]

                gripper_pose_ref = self.pose_noisfication(gripper_pose, objects_mask, pc, bin_mask,
                                                          altered_objects_elevation, mask)

                # print(gripper_pose_ref[:,0:3],gripper_pose[:,0:3])
                # exit()
                if self.freeze_approach: gripper_pose_ref[:, 0:3] = gripper_pose[:, 0:3]
                if self.freeze_beta: gripper_pose_ref[:, 3:5] = gripper_pose[:, 3:5]
                if self.freeze_distance:
                    gripper_pose_ref[:, 5:6] = gripper_pose[:, 5:6]
                if self.freeze_width:
                    gripper_pose_ref[:, 6:7] = gripper_pose[:, 6:7]

                if valid_gripper_pose:
                    pixels_A = gripper_pixel_index[0]
                    pixels_B = gripper_pixel_index[1]
                    if not self.freeze_approach: gripper_pose_ref[0, 0:3, pixels_A, pixels_B] = pose_7[0:3]
                    if not self.freeze_beta: gripper_pose_ref[0, 3:5, pixels_A, pixels_B] = pose_7[3:5]
                    if not self.freeze_distance: gripper_pose_ref[0, 5:6, pixels_A, pixels_B] = pose_7[5:6]
                    if not self.freeze_width: gripper_pose_ref[0, 6:7, pixels_A, pixels_B] = pose_7[6:7]
                    print(Fore.GREEN, 'inject label pose', Fore.RESET)

                if i % int(600 / batch_size) == 0 and i != 0 and k == 0:

                    try:
                        self.export_check_points()
                        self.save_statistics()
                        # self.load_action_model()
                    except Exception as e:
                        print(Fore.RED, str(e), Fore.RESET)
                if i % 10 == 0 and k == 0:
                    gripper_poses = gripper_pose[0].permute(1, 2, 0)[mask].detach()  # .cpu().numpy()
                    self.view_result(gripper_poses)
                    gripper_pose2 = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask].detach()
                    beta_angles = torch.atan2(gripper_pose2[:, 3], gripper_pose2[:, 4])
                    print(f'beta angles variance = {beta_angles.std()}')
                    subindexes = torch.randperm(gripper_pose2.size(0))
                    if subindexes.shape[0] > 1000: subindexes = subindexes[:1000]
                    all_values = gripper_pose2[subindexes].detach()
                    dist_values = torch.clamp((all_values[:, 5:6]), 0., 1.).clone()
                    width_values = torch.clamp((all_values[:, 6:7]), 0., 1.).clone()
                    beta_values = (all_values[:, 3:5]).clone()
                    beta_diversity = (1.001 - F.cosine_similarity(beta_values[None, ...], beta_values[:, None, :],
                                                                  dim=-1)).mean() / 2
                    print(f'Mean separation for beta : {beta_diversity}')
                    print(f'Mean separation for width : {torch.cdist(width_values, width_values, p=1).mean()}')
                    print(f'Mean separation for distance : {torch.cdist(dist_values, dist_values, p=1).mean()}')
                    self.moving_anneling_factor.update(beta_diversity.item())


                '''zero grad'''
                self.gan.critic.zero_grad(set_to_none=True)
                self.gan.generator.zero_grad(set_to_none=True)

                train_generator, tracked_indexes = self.step_discriminator_learning(depth, pc, mask, bin_mask,
                                                                                    gripper_pose,
                                                                                    gripper_pose_ref.detach().clone(),
                                                                                    altered_objects_elevation,
                                                                                    valid_pose_index, floor_elevation)

                # self.sample_from_latent = False
                if not train_generator:
                    # self.sample_from_latent = not self.sample_from_latent
                    break

                '''zero grad'''
                self.gan.critic.zero_grad(set_to_none=True)
                self.gan.generator.zero_grad(set_to_none=True)
                # self.gan.critic.eval()
                '''generated grasps'''
                gripper_pose,predicted_normal, pred_griper_collision_classifier, suction_quality_classifier, shift_affordance_classifier, background_class = self.gan.generator(depth.clone(),mask[None,None,...], detach_backbone=freeze_backbone)

                suction_seal_loss ,gripper_collision_loss,shift_appealing_loss,background_detection_loss ,suction_sampling_loss =self.multi_head_loss( pc, depth, mask, file_ids, predicted_normal, pred_griper_collision_classifier,
                                gripper_pose.detach().clone()
                                , suction_quality_classifier, shift_affordance_classifier, background_class)

                if suction_seal_loss is None: break

                gripper_sampling_loss, generator_stockhastic_loss = self.get_generator_loss(
                    depth, mask, gripper_pose, gripper_pose_ref,
                    tracked_indexes, objects_mask)

                assert not torch.isnan(gripper_sampling_loss).any(), f'{gripper_sampling_loss}'

                # gripper_sampling_loss-=weight*pred_/m

                print(Fore.LIGHTYELLOW_EX,
                      f'g_loss={gripper_sampling_loss.item()}, generator_stockhastic_loss={generator_stockhastic_loss.item()}',
                      Fore.RESET)

                loss = gripper_sampling_loss*10  +suction_seal_loss +gripper_collision_loss+shift_appealing_loss+background_detection_loss*30 +suction_sampling_loss*100
                # loss = generator_stockhastic_loss

                with torch.no_grad():
                    self.gripper_sampler_statistics.loss = gripper_sampling_loss.item()
                # if abs(loss.item())>0.0:
                # try:
                loss.backward()

                norm = torch.nn.utils.clip_grad_norm_(self.gan.generator.parameters(), max_norm=5.0)
                print(Fore.LIGHTGREEN_EX, f' G  norm : {norm}', Fore.RESET)

                self.gan.generator_optimizer.step()
                # self.ref_generator.optimizer.step()
                # except Exception as e:
                #     print(str(e))
                #     print(loss)
                # else:
                #     loss.detach_()
                #     del loss

                # self.ref_generator.model.zero_grad(set_to_none=True)
                # self.ref_generator.optimizer.zero_grad(set_to_none=True)
                self.gan.generator.zero_grad(set_to_none=True)
                self.gan.critic.zero_grad(set_to_none=True)
                self.gan.generator_optimizer.zero_grad(set_to_none=True)
                self.gan.critic_optimizer.zero_grad(set_to_none=True)

                # self.gan.critic.train()

                with torch.no_grad():
                    self.suction_sampler_statistics.loss = suction_sampling_loss.item()
                    self.suction_head_statistics.loss = suction_seal_loss.item()
                    self.shift_head_statistics.loss = shift_appealing_loss.item()
                    self.background_detector_statistics.loss = background_detection_loss.item()

                # def log_gradient_norms(model):
                #     max_norm=0
                #     max_norm_name=''
                #     for name, param in model.named_parameters():
                #         if param.grad is not None:
                #             grad_norm = param.grad.norm(2).item()  # L2 norm
                #             if grad_norm>max_norm:
                #                 max_norm=grad_norm
                #                 max_norm_name=name
                #     print(f"Layer: {max_norm_name} | Gradient Norm: {max_norm:.6f}")
                #
                # log_gradient_norms(self.gan.generator)

        pi.end()

        self.export_check_points()
        self.clear()

    def view_result(self, values):
        with torch.no_grad():

            self.gripper_sampler_statistics.print()
            self.critic_statistics.print()

            # values = gripper_pose.permute(1, 0, 2, 3).flatten(1).detach()
            try:
                print(f'gripper_pose sample = {values[np.random.randint(0, values.shape[0])].cpu()}')
            except Exception as e:
                print('result view error', str(e))
            values[:, 3:5] = F.normalize(values[:, 3:5], dim=1)
            print(f'gripper_pose std = {torch.std(values, dim=0).cpu()}')
            print(f'gripper_pose mean = {torch.mean(values, dim=0).cpu()}')
            print(f'gripper_pose max = {torch.max(values, dim=0)[0].cpu()}')
            print(f'gripper_pose min = {torch.min(values, dim=0)[0].cpu()}')

            self.moving_collision_rate.view()
            self.moving_firmness.view()
            self.moving_out_of_scope.view()
            self.relative_sampling_timing.view()
            self.moving_anneling_factor.view()
            # self.moving_scores_std.view()
            self.superior_A_model_moving_rate.view()


            self.suction_sampler_statistics.print()
            self.suction_head_statistics.print()
            self.bin_collision_statistics.print()
            self.objects_collision_statistics.print()

            self.shift_head_statistics.print()
            self.background_detector_statistics.print()

            self.gradient_moving_rate.view()

    def save_statistics(self):
        self.moving_collision_rate.save()
        self.moving_firmness.save()
        self.moving_out_of_scope.save()
        self.relative_sampling_timing.save()
        self.moving_anneling_factor.save()
        # self.moving_scores_std.save()
        self.superior_A_model_moving_rate.save()

        self.critic_statistics.save()
        self.gripper_sampler_statistics.save()

        self.data_tracker.save()

        self.gradient_moving_rate.save()

        self.suction_head_statistics.save()
        self.bin_collision_statistics.save()
        self.objects_collision_statistics.save()
        self.shift_head_statistics.save()
        self.background_detector_statistics.save()
        self.suction_sampler_statistics.save()


    def export_check_points(self):
        self.gan.export_models()
        self.gan.export_optimizers()


    def clear(self):
        self.critic_statistics.clear()

        self.suction_head_statistics.clear()
        self.bin_collision_statistics.clear()
        self.objects_collision_statistics.clear()
        self.shift_head_statistics.clear()
        self.gripper_sampler_statistics.clear()
        self.suction_sampler_statistics.clear()
        self.background_detector_statistics.clear()

def train_N_grasp_GAN(n=1):
    lr = 1e-5
    Train_grasp_GAN = TrainGraspGAN(n_samples=None, learning_rate=lr)
    torch.cuda.empty_cache()
    # torch.autograd.set_detect_anomaly(True)

    for i in range(n):
        # try:
            cuda_memory_report()
            Train_grasp_GAN.initialize(n_samples=None)
            Train_grasp_GAN.begin()
        # except Exception as e:
        #     print(Fore.RED, str(e), Fore.RESET)
        #     del Train_grasp_GAN
        #     Train_grasp_GAN = TrainGraspGAN(n_samples=None, learning_rate=lr)

    # del Train_grasp_GAN


if __name__ == "__main__":
    train_N_grasp_GAN(n=10000)
