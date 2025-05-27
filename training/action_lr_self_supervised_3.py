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
from Configurations.config import workers, distance_scope
from Online_data_audit.data_tracker import sample_positive_buffer, gripper_grasp_tracker, DataTracker
from analytical_suction_sampler import estimate_suction_direction
from check_points.check_point_conventions import GANWrapper, ModelWrapper
from dataloaders.action_dl import ActionDataset, ActionDataset2
from lib.IO_utils import custom_print, load_pickle, save_pickle
from lib.Multible_planes_detection.plane_detecttion import bin_planes_detection, cache_dir
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data2
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds, point_clouds_to_depth
from lib.image_utils import view_image
from lib.loss.D_loss import smooth_l1_loss
from lib.loss.balanced_bce_loss import BalancedBCELoss
from lib.models_utils import reshape_for_layer_norm, view_grad_val
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from models.action_net import ActionNet, Critic, action_module_key3
from models.scope_net import scope_net_vanilla, gripper_scope_module_key
from records.training_satatistics import TrainingTracker, MovingRate, truncate
from registration import camera
from training.learning_objectives.gripper_collision import gripper_collision_loss, evaluate_grasps3, \
    gripper_object_collision_loss, gripper_bin_collision_loss
from training.learning_objectives.shift_affordnace import shift_affordance_loss
from training.learning_objectives.suction_seal import suction_seal_loss
from visualiztion import view_o3d, view_npy_open3d

detach_backbone = False
lock = FileLock("file.lock")

initlize_training = True
view_mode = False
max_n = 500
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

def suction_sampler_loss(pc, target_normal, file_index):
    file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
    if os.path.exists(file_path):
        labels = load_pickle(file_path)
    else:
        labels = estimate_suction_direction(pc,
                                            view=False)  # inference time on local computer = 1.3 s        if file_index is not None:
        file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
        save_pickle(file_path, labels)

    labels = torch.from_numpy(labels).to('cuda')

    return ((1 - cos(target_normal, labels.squeeze())) ** 2).mean()

def balanced_sampling(values, mask=None, exponent=2.0, balance_indicator=1.0):
    max_ = values.max().item()
    min_ = values.min().item()
    range_ = max_ - min_

    assert range_>0.

    pivot_point = np.sqrt(np.abs(balance_indicator)) * np.sign(balance_indicator)
    xa = ((max_ - values) / range_) * pivot_point
    selection_probability = ((1 - pivot_point) / 2 + xa + 0.5 * (1 - abs(pivot_point)))
    selection_probability = selection_probability ** exponent

    if mask is None:
        dist = Categorical(probs=selection_probability)
    else:
        dist = MaskedCategorical(probs=selection_probability, mask=torch.from_numpy(mask).cuda())

    target_index = dist.sample()

    return target_index

def firmness_loss(c_,s_,f_,prediction_,label_,loss_terms_counters):
    if (loss_terms_counters[3]+loss_terms_counters[4]) >= batch_size:
        return 0.0, False, loss_terms_counters
    if sum(s_) == 0 and sum(c_) == 0:
        '''improve firmness'''
        # print(f'f____{sum(c_)} , {sum(s_) }')
        if (f_[1] - f_[0] > 0.0) and f_[1] > 0.0 :
            margin = abs(f_[1] - f_[0])
            loss_terms_counters[3] += 1.0
            return (torch.clamp(prediction_ - label_ +margin, 0.) ** firmness_expo) , True, loss_terms_counters
        elif (  f_[0]-f_[1] > 0.0) and f_[0] > 0.0 and loss_terms_counters[4] < loss_terms_counters[3]:
            margin = abs(f_[1] - f_[0])
            loss_terms_counters[4] += 1.0
            return (torch.clamp(label_-prediction_ +margin , 0.) ** firmness_expo)*0. , True, loss_terms_counters
        else:
            return 0.0, False, loss_terms_counters
    else:
        return 0.0, False, loss_terms_counters

def collision_loss(c_,s_,f_,prediction_,label_,loss_terms_counters):
    if  (loss_terms_counters[0]+loss_terms_counters[1]+loss_terms_counters[2]) >= batch_size:
        return 0.0, False,loss_terms_counters
    if  s_[0] > 0:
        if  s_[1] == 0 and c_[1]==0 and f_[1]>0.0 :
            margin=s_[0]+f_[1]+1
            loss_terms_counters[0]+=1
            return (torch.clamp(prediction_ - label_ +discrepancy_distance, 0.)**collision_expo), True,loss_terms_counters
        else:
            return 0.0, False,loss_terms_counters
    elif  sum(s_) == 0 and sum(c_)>0:
        if  c_[1]==0 and f_[1]>0.0:
            margin = f_[1]+c_[0]
            loss_terms_counters[1] += 1
            return (torch.clamp(prediction_ - label_ +discrepancy_distance, 0.)**collision_expo), True,loss_terms_counters
        elif  c_[0]==0 and f_[0]>0.0 and loss_terms_counters[2]<loss_terms_counters[1]:
            margin = f_[1]+c_[0]
            loss_terms_counters[2] += 1
            return (torch.clamp(label_-prediction_  +discrepancy_distance, 0.)**collision_expo)*0., True,loss_terms_counters
        else:
            return 0.0, False,loss_terms_counters
    else:
        return 0.0, False, loss_terms_counters


class TrainActionNet:
    def __init__(self, n_samples=None, epochs=1, learning_rate=5e-5):
        self.n_samples = n_samples
        self.size = n_samples
        self.epochs = epochs
        self.learning_rate = learning_rate

        '''model wrapper'''
        self.gan = self.prepare_model_wrapper()
        self.ref_generator = self.initialize_ref_generator()
        self.data_loader = None

        '''Moving rates'''
        self.moving_collision_rate = None
        self.moving_firmness = None
        self.moving_out_of_scope = None
        self.relative_sampling_timing = None
        self.superior_A_model_moving_rate = None

        self.moving_anneling_factor = None

        '''initialize statistics records'''
        self.bin_collision_statistics = None
        self.objects_collision_statistics = None
        self.suction_head_statistics = None
        self.shift_head_statistics = None
        self.gripper_sampler_statistics = None
        self.suction_sampler_statistics = None
        self.critic_statistics = None
        self.background_detector_statistics = None
        self.data_tracker = None

        self.sampling_centroid = None
        self.diversity_momentum = 1.0

    def initialize_ref_generator(self):
        model_wrapper = ModelWrapper(model=copy.deepcopy(self.gan.generator), module_key='ref_generator')
        model_wrapper.optimizer = torch.optim.Adam(model_wrapper.model.parameters(), lr=self.learning_rate,
                                                   betas=(0.9, 0.999), eps=1e-8)
        # model_wrapper.optimizer = torch.optim.SGD(model_wrapper.model.parameters(), lr=self.learning_rate)
        model_wrapper.model.train(True)
        return model_wrapper

    def generate_random_beta_dist_widh(self, size):
        sampled_beta = (torch.rand((size, 2), device='cuda') - 0.5) * 2
        sampled_beta = F.normalize(sampled_beta, dim=1)

        sampled_dist = torch.distributions.LogNormal(loc=-1.337, scale=0.791)
        sampled_dist = sampled_dist.sample((size, 1)).cuda()

        # sampled_dist = torch.rand((size, 1), device='cuda') ** 5
        # sampled_dist[sampled_dist > 0.7] /= 2
        # sampled_dist[sampled_dist < 0.1] += 0.1
        sampled_width = torch.distributions.LogNormal(loc=-1.312, scale=0.505)
        sampled_width = 1. - sampled_width.sample((size, 1)).cuda()
        # sampled_width = 1 - torch.rand((size, 1), device='cuda') ** 5
        # sampled_width[sampled_width < 0.3] *= 3

        sampled_pose = torch.cat([sampled_beta, sampled_dist, sampled_width], dim=1)

        return sampled_pose

    def step_ref_generator_training(self, depth, mask, gripper_pose, objects_mask, unvalid_mask, valid_mask,
                                    latent_vector):
        assert objects_mask.sum() > 0
        approach_direction = gripper_pose[:, 0:3, ...].detach().clone()
        p=self.moving_collision_rate.val+self.moving_out_of_scope.val
        if p>np.random.rand():
            beta_dist_width=self.generate_random_beta_dist_widh(gripper_pose[:, 0, ...].numel())
            beta_dist_width = reshape_for_layer_norm(beta_dist_width, camera=camera, reverse=True)
            sampled_pose=torch.cat([approach_direction,beta_dist_width],dim=1)
            return True, sampled_pose

        self.ref_generator.model.zero_grad()
        gripper_pose_ref = self.ref_generator.model.ref_generator_forward(depth.clone(), latent_vector,
                                                                          approach_direction, randomization_factor=0.)
        assert not torch.isnan(gripper_pose_ref).any()

        gripper_pose_ref2 = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :][mask]
        gripper_pose2 = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask].detach().clone()

        models_separation = torch.abs(
            gripper_pose_ref2[:, 3:][objects_mask] - gripper_pose2[:, 3:][objects_mask]).mean()
        # print(gripper_pose_ref[:,3:,...])
        # print(gripper_pose[:,3:,...])
        print(f'models_separation={models_separation}')

        if models_separation > 0.01:
            return False, gripper_pose_ref
        else:
            beta_dist_width = self.generate_random_beta_dist_widh(gripper_pose[:, 0, ...].numel())
            beta_dist_width = reshape_for_layer_norm(beta_dist_width, camera=camera, reverse=True)
            sampled_pose = torch.cat([approach_direction, beta_dist_width], dim=1)
            return True, sampled_pose

        # if models_separation > 0.5:
        #     self.ref_generator = self.initialize_ref_generator()

        # if models_separation > 0.35:
        #     return True, gripper_pose_ref.detach()

        width_scope_loss = (torch.clamp(gripper_pose_ref2[:, 6:7][objects_mask] - 1, min=0.) ** 2).mean()
        dist_scope_loss = (torch.clamp(-gripper_pose_ref2[:, 5:6][objects_mask] + 0.01, min=0.) ** 2).mean()
        width_scope_loss += (torch.clamp(-gripper_pose_ref2[:, 6:7][objects_mask], min=0.) ** 2).mean()
        dist_scope_loss += (torch.clamp(gripper_pose_ref2[:, 5:6][objects_mask] - 1, min=0.) ** 2).mean()

        # partially_detached_pose_ref2=gripper_pose_ref2.clone()
        # partially_detached_pose_ref2[valid_mask]=gripper_pose_ref2[valid_mask].detach()

        # positive_dist = gripper_pose_ref2[:, -2] > 0.001
        # valid_width = gripper_pose_ref2[:, -1] < 1.001
        # width_mean_loss = torch.abs(
        #     torch.log(1. - gripper_pose_ref2[:, -1][objects_mask & valid_width]).mean() - (-1.312)) ** 2
        # dist_mean_loss = torch.abs(
        #     torch.log(gripper_pose_ref2[:, -2][objects_mask & positive_dist]).mean() - (-1.220)) ** 2
        # width_mean_loss=torch.abs(torch.log(1.-partially_detached_pose_ref2[:,-1][all_mask ]).mean()-(-1.312))**loss_exponent
        # dist_mean_loss=torch.abs(torch.log(partially_detached_pose_ref2[:,-2][all_mask ]).mean()-(-1.220))**loss_exponent

        dist_reset_mask = unvalid_mask & (gripper_pose_ref2[:, -2] > 0.1) & (torch.rand_like(gripper_pose_ref2[:, -2])>0.5)
        if dist_reset_mask.sum() > 0.:
            dist_reduction_label = torch.clamp(gripper_pose_ref2[:, -2][dist_reset_mask] - 0.1,
                                               min=0.1).detach().clone()
            # dist_reduction_label=torch.zeros_like(gripper_pose_ref2[:, -2][dist_reset_mask] )+0.01
            dist_reset_loss = (torch.abs(gripper_pose_ref2[:, -2][dist_reset_mask] - dist_reduction_label) ** 2).mean()
            assert not torch.isnan(dist_reset_loss).any()

        else:
            dist_reset_loss = torch.tensor(0, device=gripper_pose_ref2.device)
        #
        # width_reset_mask = unvalid_mask & (gripper_pose_ref2[:, -1] < 0.8)
        # if width_reset_mask.sum() > 0.:
        #     width_reduction_label = torch.clamp(gripper_pose_ref2[:, -1][width_reset_mask] + 0.1,
        #                                         max=1.0).detach().clone()
        #     width_reset_loss = (
        #                 torch.abs(gripper_pose_ref2[:, -1][width_reset_mask] - width_reduction_label) ** 2).mean()
        #     assert not torch.isnan(width_reset_loss).any()
        #
        # else:
        #     width_reset_loss = torch.tensor(0, device=gripper_pose_ref2.device)
        #
        # firmness_mask = valid_mask & (gripper_pose_ref2[:, -2] < 0.3)
        # if firmness_mask.sum() > 0.:
        #     firmness_label = torch.clamp(gripper_pose_ref2[:, -2][firmness_mask] + 0.1, max=1.).detach().clone()
        #     firmness_regulator = (torch.abs(gripper_pose_ref2[:, -2][firmness_mask] - firmness_label) ** 2).mean()
        #     assert not torch.isnan(firmness_regulator).any()
        # else:
        #     firmness_regulator = torch.tensor(0, device=gripper_pose_ref2.device)

        random_beta_dist_width=self.generate_random_beta_dist_widh(gripper_pose_ref2.size(0))
        noise=(torch.abs(gripper_pose_ref2[:,3:][objects_mask]-random_beta_dist_width[objects_mask])**2).mean()

        # dist_std=torch.abs(gripper_pose_ref2[:,5:6][unvalid_mask].std()-0.16)**loss_exponent
        # width_std=torch.abs((1.-gripper_pose_ref2[:,5:6][unvalid_mask]).std()-0.14)**loss_exponent

        # beta_angles = torch.atan2(gripper_pose_ref2[:, 3][objects_mask], gripper_pose_ref2[:, 4][objects_mask])
        # beta_entropy=self.soft_entropy_loss(beta_angles[:,None],bins=36,sigma=0.1,min_val=0.,max_val=2*torch.pi)**2

        beta_entropy = self.soft_entropy_loss(gripper_pose_ref2[:, 3:4][objects_mask], bins=36, sigma=0.1, min_val=-1,
                                              max_val=1) ** 2
        beta_entropy += self.soft_entropy_loss(gripper_pose_ref2[:, 4:5][objects_mask], bins=36, sigma=0.1, min_val=-1,
                                               max_val=1) ** 2

        # beta_contrast = (torch.abs(
        #     gripper_pose_ref2[:, 3:5][objects_mask] + gripper_pose2[:, 3:5][objects_mask]) ** 2).mean()

        # beta_x_entropy_loss=self.soft_entropy_loss(gripper_pose_ref2[:,3:4][unvalid_mask],bins=60,sigma=0.1)**loss_exponent
        # beta_y_entropy_loss=self.soft_entropy_loss(gripper_pose_ref2[:,4:5][unvalid_mask],bins=60,sigma=0.1)**loss_exponent

        # max_std=0.5
        # width_std_loss =torch.clamp(max_std-gripper_pose_ref2[:, -1][objects_mask].std(),0.)** 2
        # dist_std_loss =torch.clamp(max_std-gripper_pose_ref2[:, -2][objects_mask].std(),0.)** 2
        # print(gripper_pose_ref)
        # print(f' {all_mask.sum()},c {width_scope_loss}, d {dist_scope_loss}, {beta_entropy}, {dist_reset_loss},{width_reset_loss},{firmness_regulator} ,{dist_mean_loss},{width_mean_loss}')
        # width_scope_loss=torch.tensor(0.,device=gripper_pose_ref.device) if torch.isnan(width_scope_loss) else width_scope_loss
        # dist_scope_loss=torch.tensor(0.,device=gripper_pose_ref.device) if torch.isnan(dist_scope_loss) else dist_scope_loss
        beta_entropy = torch.tensor(0., device=gripper_pose_ref.device) if torch.isnan(beta_entropy) else beta_entropy
        # dist_mean_loss = torch.tensor(0., device=gripper_pose_ref.device) if torch.isnan(
        #     dist_mean_loss) else dist_mean_loss
        # width_mean_loss = torch.tensor(0., device=gripper_pose_ref.device) if torch.isnan(
        #     width_mean_loss) else width_mean_loss

        loss =  width_scope_loss * 10 + dist_scope_loss * 10  + beta_entropy +noise+dist_reset_loss
        assert not torch.isnan(
            loss), f'{objects_mask.sum()}, {dist_scope_loss},{width_scope_loss}, {beta_entropy}'
        loss.backward()
        self.ref_generator.optimizer.step()
        self.ref_generator.model.zero_grad()

        return models_separation > 0.1, gripper_pose_ref.detach()

    def initialize(self, n_samples=None):
        self.n_samples = n_samples
        self.prepare_data_loader()

        '''Moving rates'''
        self.moving_collision_rate = MovingRate(action_module_key3 + '_collision', decay_rate=0.0001, initial_val=1.)
        self.moving_firmness = MovingRate(action_module_key3 + '_firmness', decay_rate=0.0001, initial_val=0.)
        self.moving_out_of_scope = MovingRate(action_module_key3 + '_out_of_scope', decay_rate=0.0001, initial_val=1.)
        self.relative_sampling_timing = MovingRate(action_module_key3 + '_relative_sampling_timing', decay_rate=0.0001,
                                                   initial_val=1.)
        self.superior_A_model_moving_rate=MovingRate(action_module_key3 + '_superior_A_model', decay_rate=0.0001,
                                                   initial_val=1.)
        self.moving_anneling_factor = MovingRate(action_module_key3 + '_anneling_factor', decay_rate=0.0001,
                                                 initial_val=0.5)

        '''initialize statistics records'''
        self.suction_head_statistics = TrainingTracker(name=action_module_key3 + '_suction_head',
                                                       iterations_per_epoch=len(self.data_loader),
                                                       track_label_balance=True)
        self.bin_collision_statistics = TrainingTracker(name=action_module_key3 + '_bin_collision',
                                                        iterations_per_epoch=len(self.data_loader),
                                                        track_label_balance=True)
        self.objects_collision_statistics = TrainingTracker(name=action_module_key3 + '_objects_collision',
                                                            iterations_per_epoch=len(self.data_loader),
                                                            track_label_balance=True)
        self.shift_head_statistics = TrainingTracker(name=action_module_key3 + '_shift_head',
                                                     iterations_per_epoch=len(self.data_loader),
                                                     track_label_balance=True)
        self.gripper_sampler_statistics = TrainingTracker(name=action_module_key3 + '_gripper_sampler',
                                                          iterations_per_epoch=len(self.data_loader),
                                                          track_label_balance=False)
        self.suction_sampler_statistics = TrainingTracker(name=action_module_key3 + '_suction_sampler',
                                                          iterations_per_epoch=len(self.data_loader),
                                                          track_label_balance=False)
        self.critic_statistics = TrainingTracker(name=action_module_key3 + '_critic',
                                                 iterations_per_epoch=len(self.data_loader), track_label_balance=False)
        self.background_detector_statistics = TrainingTracker(name=action_module_key3 + '_background_detector',
                                                              iterations_per_epoch=len(self.data_loader),
                                                              track_label_balance=False)

        self.data_tracker = DataTracker(name=gripper_grasp_tracker)

        gripper_scope = ModelWrapper(model=scope_net_vanilla(in_size=6), module_key=gripper_scope_module_key)
        gripper_scope.ini_model(train=False)
        self.gripper_arm_reachability_net = gripper_scope.model

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
        gan = GANWrapper(action_module_key3, ActionNet, Critic)
        gan.ini_models(train=True)
        # gan.critic_sgd_optimizer(learning_rate=self.learning_rate * 10)
        gan.critic_adam_optimizer(learning_rate=self.learning_rate*10,beta1=0.9)
        gan.generator_adam_optimizer(learning_rate=self.learning_rate, beta1=0.9)
        # gan.generator_sgd_optimizer(learning_rate=self.learning_rate)
        return gan

    def simulate_elevation_variations(self, original_depth, seed, max_elevation=0.2, exponent=2.0):
        with torch.no_grad():
            _, _, _, _, _, background_class_3, _ = self.gan.generator(
                original_depth.clone(),None, seed=seed)

            '''Elevation-based Augmentation'''
            objects_mask = background_class_3 <= 0.5
            shift_entities_mask = objects_mask & (original_depth > 0.0001)
            new_depth = original_depth.clone().detach()
            new_depth[shift_entities_mask] -= max_elevation * (np.random.rand() ** exponent) * camera.scale

            return new_depth

    def analytical_bin_mask(self, pc, file_ids):
        try:
            bin_mask = bin_planes_detection(pc, sides_threshold=0.005, floor_threshold=0.0015, view=False,
                                            file_index=file_ids[0], cache_name='bin_planes2')
        except Exception as error_message:
            print(file_ids[0])
            print(error_message)
            bin_mask = None
        return bin_mask

    def step_discriminator_learning(self, depth, pc, mask, bin_mask, gripper_pose, gripper_pose_ref,
                                    griper_collision_classifier_2,is_noise_label):
        '''self supervised critic learning'''
        with torch.no_grad():
            generated_grasps_cat = torch.cat([gripper_pose, gripper_pose_ref], dim=0)
            depth_cat = depth.repeat(2, 1, 1, 1)
        critic_score = self.gan.critic(depth_cat, generated_grasps_cat)
        # counter = 0
        tracked_indexes = []
        collide_with_objects_p = griper_collision_classifier_2[0, 0][mask].detach()
        collide_with_bins_p = griper_collision_classifier_2[0, 1][mask].detach()

        gripper_pose2 = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask]
        ref_gripper_pose2 = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :][mask]
        gen_scores_ = critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        ref_scores_ = critic_score.permute(0, 2, 3, 1)[1, :, :, 0][mask]


        gamma_d=torch.from_numpy(pc[:,-1]).cuda()
        gamma_d=1.0-(gamma_d-gamma_d.min())/(gamma_d.max()-gamma_d.min())
        gamma_d=0.1+gamma_d*0.9

        gamma_col = 1.0 - torch.sqrt(collide_with_objects_p.detach().clone() * collide_with_bins_p.detach().clone())


        if self.sampling_centroid is None:
            selection_p = torch.rand_like(collide_with_objects_p)
        else:
            exp = 1 + max(0., 10 * (1 - self.diversity_momentum))
            dist = 1.001 - F.cosine_similarity(ref_gripper_pose2.detach().clone(),
                                               self.sampling_centroid[None, :], dim=-1)
            binned_tensor=torch.floor(dist*10)/10
            unique_vals,counts=torch.unique(torch.floor(dist*10)/10,return_counts=True)

            probs=counts.float()#/dist.numel()
            probs=(probs-probs.min())/(probs.max()-probs.min())
            probs=1-probs+1e-3
            gamma_occurance=binned_tensor
            for k in range(unique_vals.shape[0]):
                val=unique_vals[k]
                gamma_occurance[gamma_occurance==val]=probs[k]

            assert not torch.isnan(gamma_occurance).any(), f'{unique_vals},{counts},{gamma_occurance},{probs}'

            gamma_dive = (dist / 2.) ** exp

            gamma_d=gamma_d**2.0
            gamma_col=gamma_col**0.5
            gamma_rand=torch.rand_like(gamma_col)
            selection_p=(gamma_d*gamma_col*gamma_occurance*gamma_dive*gamma_rand)**(1/5.5)
        # selection_p=torch.rand_like(gen_scores_)

        assert torch.isnan(selection_p).any() == False
        assert torch.isinf(selection_p).any() == False

        selection_mask = torch.from_numpy(bin_mask <= 0.5).cuda()  # & (collide_with_objects_p<0.9)
        n = int(min(max_n, selection_mask.sum()))

        # speculated_generator_loss = 0.
        loss_terms_counters = [0, 0, 0, 0, 0]
        l_collision = torch.tensor(0., device=gripper_pose.device)
        l_firmness = torch.tensor(0., device=gripper_pose.device)
        col_loss_counter=0
        firm_loss_counter=0
        for t in range(n):
            dist = MaskedCategorical(probs=selection_p, mask=selection_mask)

            target_index = dist.sample()
            selection_mask[target_index] = False
            target_point = pc[target_index]

            target_generated_pose = gripper_pose2[target_index]
            target_ref_pose = ref_gripper_pose2[target_index]
            label_ = ref_scores_[target_index]
            prediction_ = gen_scores_[target_index]
            c_, s_, f_ = evaluate_grasps3(target_point, target_generated_pose, target_ref_pose, pc, visualize=view_mode)

            if (s_[0] == 0) and (c_[0] == 0 or (s_[1] == 0 and c_[1] == 0) or (
                    collide_with_bins_p[target_index] <= 0.5 and collide_with_objects_p[target_index] <= 0.5)):
                self.moving_collision_rate.update(int(c_[0] > 0))
            if sum(s_) == 0 and sum(c_) == 0:
                self.moving_firmness.update(int(f_[0] > f_[1]))
            self.moving_out_of_scope.update(int(s_[0] > 0))

            counted=False
            if col_loss_counter<batch_size:
                l_c, counted, loss_terms_counters = collision_loss(c_, s_, f_, prediction_,
                                                                   label_, loss_terms_counters)
            if counted:
                print(target_ref_pose[3:].detach(), '--', target_generated_pose[3:].detach(), 'c--', c_, 's--', s_,
                      'f--', f_)
                l_collision += l_c
                col_loss_counter+=1


            if not counted and firm_loss_counter<batch_size:
                '''firmness loss'''
                l_f, counted, loss_terms_counters = firmness_loss(c_, s_, f_, prediction_,
                                                                  label_, loss_terms_counters)
                if counted:
                    # print(target_ref_pose[3:].detach(), '--', target_generated_pose[3:].detach(), 'c--', c_, 's--', s_,
                    #       'f--', f_)
                    l_firmness += l_f#*0.3
                    firm_loss_counter+=1

            if counted:
                avoid_collision = (s_[0] > 0. or c_[0] > 0.)
                A_is_collision_free = None
                A_is_more_firm = None
                if (s_[0] > 0. or c_[0] > 0.) and s_[1] == 0 and c_[1] == 0:
                    A_is_collision_free = False
                    if not is_noise_label: self.superior_A_model_moving_rate.update(0.)
                elif (s_[1] > 0. or c_[1] > 0.) and s_[0] == 0 and c_[0] == 0:
                    A_is_collision_free = True
                    if not is_noise_label: self.superior_A_model_moving_rate.update(1.0)
                elif sum(c_) == 0 and sum(s_) == 0:
                    if f_[1] > f_[0]:
                        A_is_more_firm = False
                    elif f_[0] > f_[1]:
                        A_is_more_firm = True
                tracked_indexes.append((target_index, avoid_collision, A_is_collision_free, A_is_more_firm))

                if self.sampling_centroid is None:
                    self.sampling_centroid = target_ref_pose.detach().clone()
                else:
                    diffrence = ((1 - F.cosine_similarity(target_ref_pose.detach().clone()[None, :],
                                                          self.sampling_centroid[None, :], dim=-1)) / 2) ** 2.0
                    self.diversity_momentum = self.diversity_momentum * 0.9 + diffrence.item() * 0.1
                    self.sampling_centroid = self.sampling_centroid * 0.9 + target_ref_pose.detach().clone() * 0.1

            if col_loss_counter+firm_loss_counter==2* batch_size or ((col_loss_counter == batch_size) and (firm_loss_counter == 0)):
                self.relative_sampling_timing.update((t + 1) / n)
                break

        self.critic_statistics.loss = l_collision.item() + l_firmness.item()
        if col_loss_counter== batch_size and not view_mode:
            print(loss_terms_counters)
            loss = l_collision / (loss_terms_counters[0] + loss_terms_counters[1]+loss_terms_counters[2])
            if firm_loss_counter>0:
                loss += l_firmness / (loss_terms_counters[3]+loss_terms_counters[4])
            loss.backward()
            self.gan.critic_optimizer.step()
            self.gan.critic_optimizer.zero_grad()
            return True, tracked_indexes
        else:
            print('pass, counter/Batch_size=', loss_terms_counters, '/', batch_size)
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

    def soft_hist(self, values, min_val=0., max_val=1., bins=20, sigma=0.1):
        bin_centers = torch.linspace(min_val, max_val, steps=bins, device=values.device).unsqueeze(0)  # (1,B)

        '''compute gaussian kernel wieght'''
        weights = torch.exp(-0.5 * ((values - bin_centers) / sigma) ** 2)  # (N,B)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # normalize accros bins

        hist = weights.sum(dim=0)  # sum over all samples (B,)
        hist = hist / hist.sum()
        return hist

    def get_lognormal_target(self, min_val=0., max_val=1., bins=20, loc=0.0, scale=0.5):
        data = torch.distributions.LogNormal(loc=loc, scale=scale)
        data = data.sample((10000, 1))
        data = data[data > min_val]
        data = data[data > max_val]

        hist = torch.histc(data, bins=bins, min=min_val, max=max_val)

        # normalize to sum 1
        target_dist = hist / hist.size(0)

        return target_dist

    def entropy_loss_for_target_dist(self, hist, target_dist):

        # add small epsilon to avoid log(0)
        eps = 1e-8
        hist = hist + eps
        target_dist = target_dist + eps

        kl_div = F.kl_div(hist.log(), target_dist, reduction='batchmean')

        return kl_div

    def get_generator_loss(self, depth, mask, gripper_pose, gripper_pose_ref, tracked_indexes):
        gripper_sampling_loss = torch.tensor(0., device=depth.device)

        generated_grasps_cat = torch.cat([gripper_pose, gripper_pose_ref], dim=0)
        depth_cat = depth.repeat(2, 1, 1, 1)
        critic_score = self.gan.critic(depth_cat, generated_grasps_cat)
        pred_scores_=critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        ref_scores_=critic_score.permute(0, 2, 3, 1)[1, :, :, 0][mask]
        # '''gripper sampler loss'''
        # with torch.no_grad():
        #     ref_critic_score = self.gan.critic(depth.clone(), gripper_pose_ref)
        #     assert not torch.isnan(ref_critic_score).any(), f'{ref_critic_score}'
        #     ref_scores_ = ref_critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        #
        # generated_critic_score = self.gan.critic(depth.clone(), gripper_pose, detach_backbone=True)
        # pred_scores_ = generated_critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        counter = [0., 0., 0., 0.]
        for j in range(len(tracked_indexes)):
            target_index = tracked_indexes[j][0]
            avoid_collision = tracked_indexes[j][1]
            A_is_collision_free=tracked_indexes[j][2]
            A_is_more_firm=tracked_indexes[j][3]
            label = ref_scores_[target_index]
            pred_ = pred_scores_[target_index]
            l1=torch.tensor(0.,device=pred_.device)
            l2=torch.tensor(0.,device=pred_.device)
            l3=torch.tensor(0.,device=pred_.device)
            l4 = torch.tensor(0., device=pred_.device)

            if A_is_collision_free is not None:
                if A_is_collision_free:
                    l1+=  (torch.clamp(  pred_.detach().clone()-label, 0.) ** 2)
                    counter[0]+=1.
                    # print('-----------------------------',l)
                elif not A_is_collision_free:
                    l2 +=  (torch.clamp(label.detach().clone() - pred_, 0.) ** 2)
                    counter[1]+=1.

            elif A_is_more_firm is not None:
                if A_is_more_firm:
                    l3 +=  (torch.clamp(  pred_.detach().clone()-label, 0.) ** 2)
                    counter[2]+=1.

                    # print('-----------------------------',l)
                elif not A_is_more_firm:
                    l4 +=  (torch.clamp(label.detach().clone() - pred_, 0.) ** 2)
                    counter[3]+=1.

        if counter[0]>0:gripper_sampling_loss+= l1/counter[0]
        if counter[1]>0:gripper_sampling_loss+= l2/counter[1]
        if counter[2]>0:gripper_sampling_loss+= l3/counter[2]
        if counter[3]>0:gripper_sampling_loss+= l4/counter[3]

            # l = avoid_collision * (
            #         torch.clamp(label - pred_, 0.) ** 2)
            # l=smooth_l1_loss(l,torch.zeros_like(l))
            # gripper_sampling_loss += l / len(tracked_indexes)

        return gripper_sampling_loss

    def begin(self):

        pi = progress_indicator('Begin new training round: ', max_limit=len(self.data_loader))
        gripper_pose = None
        for i, batch in enumerate(self.data_loader, 0):
            depth, file_ids = batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pi.step(i)

            '''Elevation-based augmentation'''
            seed = np.random.randint(0, 5000)
            if np.random.rand()>0.7: depth=self.simulate_elevation_variations(depth,seed)

            '''get parameters'''
            pc, mask = depth_to_point_clouds(depth[0, 0].cpu().numpy(), camera)
            pc = transform_to_camera_frame(pc, reverse=True)

            latent_vector = torch.randn(size=(depth.numel(), 8), device=depth.device)

            with torch.no_grad():
                gripper_pose, suction_direction, griper_collision_classifier_2, _, _, background_class_2, _ = self.gan.generator(
                    depth.clone(), latent_vector, seed=seed)

            '''background detection head'''
            bin_mask = self.analytical_bin_mask(pc, file_ids)
            if bin_mask is None: continue
            objects_mask_numpy = bin_mask <= 0.5
            objects_mask = torch.from_numpy(objects_mask_numpy).cuda()

            '''shuffle mask'''
            objects_collision_class = griper_collision_classifier_2.permute(0, 2, 3, 1)[0, :, :, 0][mask].detach()
            bin_collision_class = griper_collision_classifier_2.permute(0, 2, 3, 1)[0, :, :, 1][mask].detach()
            unvalid_mask = (objects_collision_class > 0.5) | (bin_collision_class > 0.5)
            unvalid_mask = unvalid_mask & objects_mask
            valid_mask = (objects_collision_class < 0.5) | (bin_collision_class < 0.5)
            valid_mask = valid_mask & objects_mask

            ''''train ref generator'''
            with torch.enable_grad():
                is_noise_label, gripper_pose_ref = self.step_ref_generator_training(depth, mask, gripper_pose,
                                                                                         objects_mask, unvalid_mask,
                                                                                         valid_mask, latent_vector)
            # if not seperation_criteria: continue

            if i % 25 == 0 and i != 0:
                gripper_poses = gripper_pose[0].permute(1, 2, 0)[mask].detach()  # .cpu().numpy()
                self.view_result(gripper_poses[unvalid_mask])
                self.export_check_points()
                self.save_statistics()
                gripper_pose2 = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask][unvalid_mask].detach()
                beta_angles = torch.atan2(gripper_pose2[:, 3], gripper_pose2[:, 4])
                print(f'beta angles variance = {beta_angles.std()}')
                for p in range(3, 7):
                    subindexes = torch.randperm(gripper_pose2.size(0))
                    if subindexes.shape[0] > 1000: subindexes = subindexes[:1000]
                    values = gripper_pose2[subindexes, p:p + 1].detach().clone().cpu()
                    mean_dist = torch.cdist(values, values, p=1).mean()
                    print(f'Mean separation dist ({p}): {mean_dist}')
                    if p in (3, 4): self.moving_anneling_factor.update(mean_dist)

            '''zero grad'''
            self.gan.critic.zero_grad()
            self.gan.generator.zero_grad()

            train_generator, tracked_indexes = self.step_discriminator_learning(depth, pc, mask, bin_mask, gripper_pose,
                                                                                gripper_pose_ref.detach().clone(),
                                                                                griper_collision_classifier_2,is_noise_label)
            # if not train_generator:continue
            if view_mode: continue
            # if speculated_generator_loss<1e-4:
            #     print('pass generator training',f' critic loss: {loss.item()}')
            #     continue # train the discriminator

            '''zero grad'''
            self.gan.critic.zero_grad()
            self.gan.generator.zero_grad()

            '''generated grasps'''
            gripper_pose, suction_direction, griper_collision_classifier, suction_quality_classifier, shift_appealing_classifier, background_class, depth_features = self.gan.generator(
                depth.clone(), latent_vector, seed=seed, detach_backbone=detach_backbone)

            # assert (gripper_pose.detach()-early_sampled_poses).sum()==0, f'{(gripper_pose.detach()-early_sampled_poses).sum()==0}'

            assert not torch.isnan(gripper_pose).any(), f'{gripper_pose}'

            # if detach_beta: gripper_pose[:, 3:5] = gripper_pose[:, 3:5].detach()
            # if detach_dist: gripper_pose[:, -2] = gripper_pose[:, -2].detach()
            # if detach_width: gripper_pose[:, -1] = gripper_pose[:, -1].detach()

            '''loss computation'''
            suction_loss = suction_quality_classifier.mean() * 0.0
            gripper_loss = griper_collision_classifier.mean() * 0.0
            shift_loss = shift_appealing_classifier.mean() * 0.0
            background_loss = background_class.mean() * 0.0
            suction_sampling_loss = suction_direction.mean() * 0.0

            non_zero_background_loss_counter = 0

            gripper_sampling_loss = self.get_generator_loss(depth, mask, gripper_pose, gripper_pose_ref,
                                                            tracked_indexes)

            print(f'generator loss= {gripper_sampling_loss.item()}')


                # gripper_sampling_loss-=weight*pred_/m
            self.gripper_sampler_statistics.loss = gripper_sampling_loss.item()

            suction_sampling_loss += suction_sampler_loss(pc, suction_direction.permute(0, 2, 3, 1)[0][mask],
                                                          file_index=file_ids[0])

            # loss = gripper_sampling_loss+suction_sampling_loss
            # loss.backward()
            # self.gan.generator_optimizer.step()
            # self.gan.generator_optimizer.zero_grad()
            #
            # with torch.no_grad():
            #     self.gripper_sampler_statistics.loss = gripper_sampling_loss.item()
            #     self.suction_sampler_statistics.loss = suction_sampling_loss.item()
            #     self.suction_head_statistics.loss = suction_loss.item()
            #     self.shift_head_statistics.loss = shift_loss.item()
            #     self.background_detector_statistics.loss = background_loss.item()
            #
            # # print(f'c_loss={truncate(l_c)}, g_loss={(gripper_sampling_loss.item())},  c:{self.moving_collision_rate.val}, s:{self.moving_out_of_scope.val}, rk={r_k}, diversity_momentum={self.diversity_momentum}')
            #
            # continue

            gripper_poses = gripper_pose[0].permute(1, 2, 0)[mask].detach()  # .cpu().numpy()
            suction_head_predictions = suction_quality_classifier[0, 0][mask]
            gripper_head_predictions = griper_collision_classifier[0, :].permute(1, 2, 0)[mask]
            shift_head_predictions = shift_appealing_classifier[0, 0][mask]
            background_class_predictions = background_class.permute(0, 2, 3, 1)[0, :, :, 0][mask]

            with torch.no_grad():
                normals = suction_direction[0].permute(1, 2, 0)[mask].detach().cpu().numpy()

            if bin_mask is None:
                print(Fore.RED, f'Failed to generate label for background segmentation, file id ={file_ids[0]}',
                      Fore.RESET)
            else:
                label = torch.from_numpy(bin_mask).to(background_class_predictions.device).float()
                background_loss += bce_loss(background_class_predictions, label)
                self.background_detector_statistics.update_confession_matrix(label,
                                                                             background_class_predictions.detach())
                non_zero_background_loss_counter += 1

            # for k in range(m):
            #     '''gripper collision head'''
            #     sta=self.objects_collision_statistics if i%2==0 else self.bin_collision_statistics
            #     gripper_target_index=model_dependent_sampling(pc, gripper_head_predictions[:,i%2], gripper_head_max_score, gripper_head_score_range,objects_mask,probability_exponent=10,balance_indicator=sta.label_balance_indicator)
            #     gripper_target_point = pc[gripper_target_index]
            #     gripper_prediction_ = gripper_head_predictions[gripper_target_index]
            #     gripper_target_pose = gripper_poses[gripper_target_index]
            #     gripper_loss+=gripper_collision_loss(gripper_target_pose, gripper_target_point, pc,objects_mask, gripper_prediction_,self.objects_collision_statistics ,self.bin_collision_statistics)/m

            for k in range(batch_size):
                '''gripper-object collision'''
                gripper_target_index = balanced_sampling(gripper_head_predictions[:, 0], mask=objects_mask_numpy,
                                                         exponent=10.0,
                                                         balance_indicator=self.objects_collision_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = gripper_head_predictions[gripper_target_index]
                gripper_target_pose = gripper_poses[gripper_target_index]
                gripper_loss += gripper_object_collision_loss(gripper_target_pose, gripper_target_point, pc,
                                                              objects_mask_numpy, gripper_prediction_,
                                                              self.objects_collision_statistics) / batch_size

            for k in range(batch_size):

                '''gripper-bin collision'''
                gripper_target_index = balanced_sampling(gripper_head_predictions[:, 1], mask=objects_mask_numpy,
                                                         exponent=10.0,
                                                         balance_indicator=self.bin_collision_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = gripper_head_predictions[gripper_target_index]
                gripper_target_pose = gripper_poses[gripper_target_index]
                gripper_loss += gripper_bin_collision_loss(gripper_target_pose, gripper_target_point, pc,
                                                           objects_mask_numpy, gripper_prediction_,
                                                           self.bin_collision_statistics) / batch_size

            for k in range(batch_size):
                '''suction seal head'''
                suction_target_index = balanced_sampling(suction_head_predictions, mask=objects_mask_numpy,
                                                         exponent=10.0,
                                                         balance_indicator=self.suction_head_statistics.label_balance_indicator)
                suction_prediction_ = suction_head_predictions[suction_target_index]
                suction_loss += suction_seal_loss(pc, normals, suction_target_index, suction_prediction_,
                                                  self.suction_head_statistics, objects_mask_numpy) / batch_size

            for k in range(batch_size):
                '''shift affordance head'''
                shift_target_index = balanced_sampling(shift_head_predictions, mask=None, exponent=10.0,
                                                       balance_indicator=self.shift_head_statistics.label_balance_indicator)
                shift_target_point = pc[shift_target_index]
                shift_prediction_ = shift_head_predictions[shift_target_index]
                shift_loss += shift_affordance_loss(pc, shift_target_point, objects_mask_numpy,
                                                    self.shift_head_statistics, shift_prediction_) / batch_size

            if non_zero_background_loss_counter: background_loss / non_zero_background_loss_counter

            loss = (suction_loss * 1 + gripper_loss * 1 + shift_loss * 1 + gripper_sampling_loss * 2.0 + suction_sampling_loss + background_loss * 30.0)
            loss.backward()
            self.gan.generator_optimizer.step()

            # if not is_noise_label:
            self.ref_generator.optimizer.step()
            self.ref_generator.model.zero_grad()
            self.gan.generator_optimizer.zero_grad()

            with torch.no_grad():
                self.gripper_sampler_statistics.loss = gripper_sampling_loss.item()
                self.suction_sampler_statistics.loss = suction_sampling_loss.item()
                self.suction_head_statistics.loss = suction_loss.item()
                self.shift_head_statistics.loss = shift_loss.item()
                self.background_detector_statistics.loss = background_loss.item()

        pi.end()

        # self.view_result(gripper_poses[unvalid_mask])

        self.export_check_points()
        self.clear()

    def view_result(self, values):
        with torch.no_grad():
            self.suction_sampler_statistics.print()
            self.suction_head_statistics.print()
            self.bin_collision_statistics.print()
            self.objects_collision_statistics.print()

            self.shift_head_statistics.print()
            self.background_detector_statistics.print()
            self.gripper_sampler_statistics.print()
            self.critic_statistics.print()

            # values = gripper_pose.permute(1, 0, 2, 3).flatten(1).detach()
            try:
                print(f'gripper_pose sample = {values[np.random.randint(0, values.shape[0])].cpu()}')
            except Exception as e:
                pass
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
            self.superior_A_model_moving_rate.view()

    def save_statistics(self):
        self.moving_collision_rate.save()
        self.moving_firmness.save()
        self.moving_out_of_scope.save()
        self.relative_sampling_timing.save()
        self.moving_anneling_factor.save()
        self.superior_A_model_moving_rate.save()

        self.suction_head_statistics.save()
        self.bin_collision_statistics.save()
        self.objects_collision_statistics.save()
        self.shift_head_statistics.save()
        self.critic_statistics.save()
        self.background_detector_statistics.save()
        self.gripper_sampler_statistics.save()
        self.suction_sampler_statistics.save()

        self.data_tracker.save()

    def export_check_points(self):
        self.gan.export_models()
        self.gan.export_optimizers()

    def clear(self):
        self.suction_head_statistics.clear()
        self.bin_collision_statistics.clear()
        self.objects_collision_statistics.clear()
        self.shift_head_statistics.clear()
        self.gripper_sampler_statistics.clear()
        self.suction_sampler_statistics.clear()
        self.critic_statistics.clear()
        self.background_detector_statistics.clear()

if __name__ == "__main__":
    lr = 1e-5
    train_action_net = TrainActionNet(n_samples=None, learning_rate=lr)
    torch.cuda.empty_cache()

    with torch.no_grad() if view_mode else torch.enable_grad():

        for i in range(10000):
            # try:
                cuda_memory_report()
                train_action_net.initialize(n_samples=None)
                train_action_net.begin()
            # except Exception as e:
            #     pass
