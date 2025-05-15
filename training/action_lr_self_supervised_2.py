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
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds,point_clouds_to_depth
from lib.image_utils import view_image
from lib.loss.balanced_bce_loss import BalancedBCELoss
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from models.action_net import ActionNet, Critic, action_module_key2
from models.scope_net import scope_net_vanilla, gripper_scope_module_key
from records.training_satatistics import TrainingTracker, MovingRate, truncate
from registration import camera
from training.learning_objectives.gripper_collision import gripper_collision_loss, evaluate_grasps3, \
    gripper_object_collision_loss,gripper_bin_collision_loss
from training.learning_objectives.shift_affordnace import shift_affordance_loss
from training.learning_objectives.suction_seal import suction_seal_loss
from visualiztion import view_o3d, view_npy_open3d

detach_backbone=False
generator_exponent = 2.0
discriminator_exponent = 2.0
firmness_exponent=2.0
lock = FileLock("file.lock")

initlize_training=True

max_n = 500
batch_size=16

training_buffer = online_data2()
training_buffer.main_modality=training_buffer.depth

bce_loss=nn.BCELoss()

balanced_bce_loss=BalancedBCELoss()
print=custom_print

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)
cache_name='normals'
discrepancy_distance=1.0

def suction_sampler_loss(pc,target_normal,file_index):
    file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
    if os.path.exists(file_path):
        labels = load_pickle(file_path)
    else:
        labels = estimate_suction_direction(pc,view=False)  # inference time on local computer = 1.3 s        if file_index is not None:
        file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
        save_pickle(file_path, labels)

    labels = torch.from_numpy(labels).to('cuda')

    return ((1 - cos(target_normal, labels.squeeze())) ** 2).mean()


def balanced_sampling(values,mask=None,exponent=2.0,balance_indicator=1.0):

    max_=values.max().item()
    min_=values.min().item()
    range_=max_-min_

    pivot_point = np.sqrt(np.abs(balance_indicator)) * np.sign(balance_indicator)
    xa = ((max_ - values) / range_) * pivot_point
    selection_probability = ((1 - pivot_point) / 2 + xa + 0.5 * (1 - abs(pivot_point)))
    selection_probability = selection_probability ** exponent

    if mask is None:
        dist=Categorical(probs=selection_probability)
    else:
        dist = MaskedCategorical(probs=selection_probability, mask=torch.from_numpy(mask).cuda())

    target_index = dist.sample()

    return target_index

def critic_loss(c_,s_,f_,prediction_,label_,loss_terms_counters):
    if  s_[0] > 0:
        if c_[1] + s_[1] == 0:
            margin=abs(c_[1] -c_[0] )
            loss_terms_counters[0]+=1
            return (torch.clamp(prediction_ - label_ + discrepancy_distance+margin, 0.)**discriminator_exponent), True,loss_terms_counters
        # elif s_[1] == 0 :
        #     return (torch.clamp(prediction_ - label_ +discrepancy_distance, 0.)** discriminator_exponent) , True
        else:
            # if s_[0]>s_[1]:
            #     return (torch.clamp(prediction_ - label_ , 0.) ** discriminator_exponent), True
            # elif s_[1]>s_[0]:
            #     return (torch.clamp(label_-prediction_  , 0.) ** discriminator_exponent), True
            # else:
            return 0.0, False,loss_terms_counters
    elif sum(c_) + sum(s_) > 0:
        if c_[1] + s_[1] == 0:
            # print('d')
            # cur = (torch.clamp(label_ - prediction_ - discrepancy_distance, 0.) ** discriminator_exponent)
            # if cur.item() > 0.:
            #     print(f'curriculum measure C label={label_.item()}, prediction={prediction_.item()}')
            #     return cur, True
            print('collision contrast')
            margin = abs(c_[1] - c_[0])
            loss_terms_counters[1] += 1
            return (torch.clamp(prediction_ - label_ +discrepancy_distance+margin, 0.)**discriminator_exponent), True,loss_terms_counters
        elif c_[0] + s_[0] == 0 and (loss_terms_counters[2]<math.ceil(batch_size/4.)) :
            margin = abs(c_[1] - c_[0])
            loss_terms_counters[2] += 1

            print('network found feasible grasp')
            return (torch.clamp( label_-prediction_ + discrepancy_distance+margin, 0.)**discriminator_exponent), True,loss_terms_counters
        else:
            # print('f')

            return 0.0, False,loss_terms_counters

    else:
        # print('g')

        '''improve firmness'''
        # print(f'f____{sum(c_)} , {sum(s_) }')
        if f_[1] > f_[0]:
            margin = abs(f_[1] - f_[0])
            loss_terms_counters[3] += 1
            return (torch.clamp(prediction_ - label_ +margin, 0.)**firmness_exponent) , True,loss_terms_counters
        elif (f_[0] >= f_[1]) and (loss_terms_counters[4]<math.ceil(batch_size/4.)):
            margin = abs(f_[1] - f_[0])

            print('network found firmed sample')
            loss_terms_counters[4] += 1

            return (torch.clamp(label_ - prediction_+margin, 0.)**firmness_exponent), True,loss_terms_counters
        else:
            return 0.0, False,loss_terms_counters

class TrainActionNet:
    def __init__(self,n_samples=None,epochs=1,learning_rate=5e-5):
        self.n_samples=n_samples
        self.size=n_samples
        self.epochs=epochs
        self.learning_rate=learning_rate

        '''model wrapper'''
        self.gan=self.prepare_model_wrapper()
        self.ref_generator=self.initialize_ref_generator()
        self.data_loader=None

        '''Moving rates'''
        self.moving_collision_rate=None
        self.moving_firmness=None
        self.moving_out_of_scope=None
        self.relative_sampling_timing=None

        '''initialize statistics records'''
        self.bin_collision_statistics = None
        self.objects_collision_statistics=None
        self.suction_head_statistics = None
        self.shift_head_statistics = None
        self.gripper_sampler_statistics = None
        self.suction_sampler_statistics = None
        self.critic_statistics = None
        self.background_detector_statistics = None
        self.data_tracker = None

        self.sampling_centroid=None
        self.diversity_momentum=1.0

    def initialize_ref_generator(self):
        model_wrapper=ModelWrapper(model=copy.deepcopy(self.gan.generator), module_key='ref_generator')
        # model_wrapper.optimizer = torch.optim.Adam(model_wrapper.model.parameters(), lr=self.learning_rate, betas=(0.1, 0.999), eps=1e-8)
        model_wrapper.optimizer = torch.optim.SGD(model_wrapper.model.parameters(), lr=self.learning_rate*10)

        model_wrapper.model.train(True)
        return model_wrapper

    def step_ref_generator_training(self,depth,mask,gripper_pose,shuffle_mask,loss_exponent=1.0):
        approach_direction = gripper_pose[:, 0:3, ...].detach().clone()

        self.ref_generator.model.zero_grad()

        gripper_pose_ref,noisy_dist,noisy_width= self.ref_generator.model.ref_generator_forward(depth.clone(),approach_direction,randomization_factor=0.)

        models_separation=(gripper_pose_ref[:,3:,...].detach()-gripper_pose[:,3:,...].detach()).pow(2).mean()
        # print(gripper_pose_ref[:,3:,...])
        # print(gripper_pose[:,3:,...])
        # print(f'models_separation={models_separation}')
        if models_separation>0.2:
            self.ref_generator = self.initialize_ref_generator()
            return True,gripper_pose_ref.detach()

        gripper_pose_ref2 = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :][mask]
        noisy_dist = noisy_dist.permute(0, 2, 3, 1)[0, :, :, :][mask]
        noisy_width = noisy_width.permute(0, 2, 3, 1)[0, :, :, :][mask]

        dist_loss=torch.abs(gripper_pose_ref2[:,5:6]-noisy_dist)[shuffle_mask].mean()**loss_exponent
        width_loss=torch.abs(gripper_pose_ref2[:,6:7]-noisy_width)[shuffle_mask].mean()**loss_exponent

        width_mean_loss=torch.abs((gripper_pose_ref2[:,-1])[shuffle_mask].mean()-0.7)**loss_exponent
        dist_mean_loss=torch.abs((gripper_pose_ref2[:,-2])[shuffle_mask].mean()-0.3)**loss_exponent

        beta_x_entropy_loss=self.soft_entropy_loss(gripper_pose_ref2[:,3:4][shuffle_mask],bins=60,sigma=0.1)**loss_exponent
        beta_y_entropy_loss=self.soft_entropy_loss(gripper_pose_ref2[:,4:5][shuffle_mask],bins=60,sigma=0.1)**loss_exponent

        loss=dist_mean_loss+width_mean_loss+beta_x_entropy_loss+beta_y_entropy_loss+dist_loss+width_loss

        loss.backward()
        self.ref_generator.optimizer.step()
        self.ref_generator.model.zero_grad()

        return models_separation>0.01,gripper_pose_ref.detach()

    def initialize(self,n_samples=None):
        self.n_samples=n_samples
        self.prepare_data_loader()

        '''Moving rates'''
        self.moving_collision_rate=MovingRate(action_module_key2+'_collision',decay_rate=0.0001,initial_val=1.)
        self.moving_firmness=MovingRate(action_module_key2+'_firmness',decay_rate=0.0001,initial_val=0.)
        self.moving_out_of_scope=MovingRate(action_module_key2+'_out_of_scope',decay_rate=0.0001,initial_val=1.)
        self.relative_sampling_timing=MovingRate(action_module_key2+'_relative_sampling_timing',decay_rate=0.0001,initial_val=1.)

        '''initialize statistics records'''
        self.suction_head_statistics = TrainingTracker(name=action_module_key2+'_suction_head', iterations_per_epoch=len(self.data_loader), track_label_balance=True)
        self.bin_collision_statistics = TrainingTracker(name=action_module_key2+'_bin_collision', iterations_per_epoch=len(self.data_loader), track_label_balance=True)
        self.objects_collision_statistics = TrainingTracker(name=action_module_key2+'_objects_collision', iterations_per_epoch=len(self.data_loader), track_label_balance=True)
        self.shift_head_statistics = TrainingTracker(name=action_module_key2+'_shift_head', iterations_per_epoch=len(self.data_loader), track_label_balance=True)
        self.gripper_sampler_statistics = TrainingTracker(name=action_module_key2+'_gripper_sampler', iterations_per_epoch=len(self.data_loader), track_label_balance=False)
        self.suction_sampler_statistics = TrainingTracker(name=action_module_key2+'_suction_sampler', iterations_per_epoch=len(self.data_loader), track_label_balance=False)
        self.critic_statistics = TrainingTracker(name=action_module_key2+'_critic', iterations_per_epoch=len(self.data_loader), track_label_balance=False)
        self.background_detector_statistics = TrainingTracker(name=action_module_key2+'_background_detector', iterations_per_epoch=len(self.data_loader), track_label_balance=False)

        self.data_tracker = DataTracker(name=gripper_grasp_tracker)

        gripper_scope = ModelWrapper(model=scope_net_vanilla(in_size=6), module_key=gripper_scope_module_key)
        gripper_scope.ini_model(train=False)
        self.gripper_arm_reachability_net = gripper_scope.model

    def prepare_data_loader(self):
        file_ids=training_buffer.get_indexes()
        # file_ids = sample_positive_buffer(size=self.n_samples, dict_name=gripper_grasp_tracker,
        #                                   disregard_collision_samples=True,sample_with_probability=False)
        print(Fore.CYAN, f'Buffer size = {len(file_ids)}',Fore.RESET)
        dataset = ActionDataset2(data_pool=training_buffer, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers,
                                                       shuffle=True)
        self.size=len(dataset)
        self.data_loader= data_loader

    def prepare_model_wrapper(self):
        '''load  models'''
        gan = GANWrapper(action_module_key2, ActionNet, Critic)
        gan.ini_models(train=True)
        # gan.critic_sgd_optimizer(learning_rate=self.learning_rate*10)
        gan.critic_adam_optimizer(learning_rate=self.learning_rate*10,beta1=0.9)
        gan.generator_adam_optimizer(learning_rate=self.learning_rate,beta1=0.5)
        # gan.generator_sgd_optimizer(learning_rate=self.learning_rate)
        return gan

    def simulate_elevation_variations(self,original_depth,seed,max_elevation=0.2,exponent=2.0):
        with torch.no_grad():
            _, _, _, _, _, background_class_3, _ = self.gan.generator(
                original_depth.clone(),seed=seed)

            '''Elevation-based Augmentation'''
            objects_mask = background_class_3 <= 0.5
            shift_entities_mask = objects_mask & (original_depth > 0.0001)
            new_depth = original_depth.clone().detach()
            new_depth[shift_entities_mask] -= max_elevation * (np.random.rand()**exponent) * camera.scale

            return new_depth

    def analytical_bin_mask(self,pc,file_ids):
        try:
            bin_mask = bin_planes_detection(pc, sides_threshold=0.005, floor_threshold=0.0015, view=False,
                                            file_index=file_ids[0], cache_name='bin_planes2')
        except Exception as error_message:
            print(file_ids[0])
            print(error_message)
            bin_mask = None
        return bin_mask

    def step_discriminator_learning(self,depth,pc,mask,bin_mask,gripper_pose,gripper_pose_ref,griper_collision_classifier_2):
        '''self supervised critic learning'''
        loss = torch.tensor([0.], device=gripper_pose.device)
        with torch.no_grad():
            generated_grasps_cat = torch.cat([gripper_pose, gripper_pose_ref], dim=0)
            depth_cat = depth.repeat(2, 1, 1, 1)
        critic_score = self.gan.critic(depth_cat, generated_grasps_cat)
        counter = 0
        tracked_indexes = []
        collide_with_objects_p = griper_collision_classifier_2[0, 0][mask].detach()
        collide_with_bins_p = griper_collision_classifier_2[0, 1][mask].detach()

        gripper_pose2 = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask]
        ref_gripper_pose2 = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :][mask]
        gen_scores_ = critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        ref_scores_ = critic_score.permute(0, 2, 3, 1)[1, :, :, 0][mask]

        # if np.random.rand()>0.1:
        #     # pivot=0.5
        #     selection_p=1.0-torch.sqrt(collide_with_objects_p.detach().clone()*collide_with_bins_p.detach().clone())
        #     # if not initlize_training:
        #     #     selection_p=1-torch.abs(selection_p-pivot)
        # else:
        if self.sampling_centroid is None:
            selection_p = torch.rand_like(collide_with_objects_p)
        else:
            exp = 1 + max(0., 10 * (1 - self.diversity_momentum))
            dist = 1.001 - F.cosine_similarity(ref_gripper_pose2[:, 3:5].detach().clone(),
                                               self.sampling_centroid[None, :], dim=-1)
            selection_p = (dist / 2.) ** exp

        assert torch.isnan(selection_p).any() == False
        assert torch.isinf(selection_p).any() == False

        selection_mask = torch.from_numpy(bin_mask <= 0.5).cuda()
        n = int(min(max_n, selection_mask.sum()))

        # speculated_generator_loss = 0.
        loss_terms_counters=[0,0,0,0,0]
        for t in range(n):
            dist = MaskedCategorical(probs=selection_p, mask=selection_mask)

            target_index = dist.sample()
            selection_mask[target_index] = False
            target_point = pc[target_index]

            target_generated_pose = gripper_pose2[target_index]
            target_ref_pose = ref_gripper_pose2[target_index]
            label_ = ref_scores_[target_index]
            prediction_ = gen_scores_[target_index]
            c_, s_, f_ = evaluate_grasps3(target_point, target_generated_pose, target_ref_pose, pc, visualize=False)
            # print(c_,'  ',s_)

            if c_[0] == 0 or c_[1] == 0 or (
                    collide_with_bins_p[target_index] <= 0.5 and collide_with_objects_p[target_index] <= 0.5):
                self.moving_collision_rate.update(int(c_[0] > 0))
            if sum(c_) == 0 and sum(s_) == 0:
                self.moving_firmness.update(int(f_[0] > f_[1]))
            self.moving_out_of_scope.update(int(s_[0] > 0))

            l, counted,loss_terms_counters = critic_loss(c_, s_, f_, prediction_,
                                     label_,loss_terms_counters)  # ,early_mode=self.moving_out_of_scope.val>0.2 or self.moving_collision_rate.val>0.65)
            if counted:
                # print(target_ref_pose[3:].detach(), '--', target_generated_pose[3:].detach(),'c--',c_,'s--',s_,'f--',f_ )
                counter += 1
                loss += l / batch_size
                avoid_collision = (c_[0] > 0. or s_[0] > 0.)
                # speculated_generator_loss += avoid_collision * torch.clamp(label_.detach() - prediction_.detach(), 0.)
                tracked_indexes.append((target_index, avoid_collision, f_[1] > f_[0]))
                if self.sampling_centroid is None:
                    self.sampling_centroid = target_ref_pose[3:5].detach().clone()
                else:
                    # change=(self.sampling_centroid-target_ref_pose[3:].detach().clone())**2
                    diffrence = ((1 - F.cosine_similarity(target_ref_pose[3:5].detach().clone()[None, :],
                                                          self.sampling_centroid[None, :], dim=-1)) / 2) ** 2.0
                    self.diversity_momentum = self.diversity_momentum * 0.9 + diffrence.item() * 0.1
                    self.sampling_centroid = self.sampling_centroid * 0.9 + target_ref_pose[3:5].detach().clone() * 0.1

            if counter == batch_size:
                print(loss_terms_counters)
                self.relative_sampling_timing.update((t + 1) / n)
                break
        l_c = loss.item()
        self.critic_statistics.loss = l_c
        if counter == batch_size:
            loss.backward()
            self.gan.critic_optimizer.step()
            self.gan.critic_optimizer.zero_grad()
            return True, tracked_indexes
        else:
            print('pass, counter/Batch_size=', counter, '/', batch_size)
            return False, tracked_indexes

    def soft_entropy_loss(self,values,bins=40,sigma=0.1,min_val=None,max_val=None):
        """
        Differential approximation of entropy oiver scaler output
        """
        N=values.size(0)

        '''create bin centers'''
        if min_val is None and max_val is None:
            min_val,max_val=values.min().item(),values.max().item()
        bin_centers=torch.linspace(min_val,max_val,steps=bins,device=values.device)
        bin_centers=bin_centers.view(1,-1)

        '''compute soft assignment via Gaussion kernel'''
        dists=(values-bin_centers).pow(2)  # [B, num_bins]
        soft_assignments=torch.exp(-dists/(2*sigma**2)) # [B, num_bins]

        '''normalize per sample and then sum over batch'''
        probs=soft_assignments/(soft_assignments.sum(dim=1,keepdim=True)+1e-8)
        avg_probs=probs.mean(dim=0) # [num_bins], average histogram over batch
        entropy=-torch.sum(avg_probs*torch.log(avg_probs+1e-8))
        # return -entropy
        max_entropy=torch.log(torch.tensor([bins],device=entropy.device).float())
        return (max_entropy-entropy)/max_entropy # maximize entropy

    def get_generator_loss(self,depth,mask,gripper_pose,shuffle_mask,gripper_pose_ref,tracked_indexes):
        gripper_sampling_loss = gripper_pose.mean() * 0.0

        '''gripper sampler loss'''
        with torch.no_grad():
            ref_critic_score = self.gan.critic(depth.clone(), gripper_pose_ref)
            assert not torch.isnan(ref_critic_score).any(), f'{ref_critic_score}'
            ref_scores_ = ref_critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]

        generated_critic_score = self.gan.critic(depth.clone(), gripper_pose, detach_backbone=True)
        pred_scores_ = generated_critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]

        for j in range(batch_size):
            target_index = tracked_indexes[j][0]
            avoid_collision = tracked_indexes[j][1]
            label = ref_scores_[target_index]
            pred_ = pred_scores_[target_index]
            gripper_sampling_loss += avoid_collision * (
                        torch.clamp(label - pred_, 0.) ** generator_exponent) / batch_size

        if initlize_training:
            gripper_pose2 = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask]

            width_mean_loss = torch.abs((gripper_pose2[:, -1][shuffle_mask]).mean() - 0.7) ** 2.
            dist_mean_loss = torch.abs((gripper_pose2[:, -2][shuffle_mask]).mean() - 0.3) ** 2.

            max_std=0.5
            width_std_loss =torch.clamp(max_std-gripper_pose2[:, -1][shuffle_mask].std(),0.)** 2.
            dist_std_loss =torch.clamp(max_std-gripper_pose2[:, -2][shuffle_mask].std(),0.)** 2.

            width_scope_loss=torch.clamp(gripper_pose2[:, 6:7]-1,min=0.).mean()**2
            dist_scope_loss=torch.clamp(-gripper_pose2[:, 5:6],min=0.).mean()**2

            beta_x_entropy_loss = self.soft_entropy_loss(gripper_pose2[:, 3:4][shuffle_mask], bins=60, sigma=0.1) ** 2
            beta_y_entropy_loss = self.soft_entropy_loss(gripper_pose2[:, 4:5][shuffle_mask], bins=60, sigma=0.1) ** 2

            return gripper_sampling_loss+(width_mean_loss+dist_mean_loss+beta_x_entropy_loss+beta_y_entropy_loss+width_std_loss+dist_std_loss+width_scope_loss+dist_scope_loss)*0.01
        else:
            return gripper_sampling_loss

    def begin(self):

        pi = progress_indicator('Begin new training round: ', max_limit=len(self.data_loader))
        gripper_pose=None
        for i, batch in enumerate(self.data_loader, 0):
            depth,file_ids= batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pi.step(i)

            '''Elevation-based augmentation'''
            seed=np.random.randint(0,5000)
            # if np.random.rand()>0.5: depth=self.simulate_elevation_variations(depth,seed)

            with torch.no_grad():
                gripper_pose, suction_direction, griper_collision_classifier_2, _, _, background_class_2, _ = self.gan.generator(
                    depth.clone(), seed=seed)

            '''get parameters'''
            pc, mask = depth_to_point_clouds(depth[0, 0].cpu().numpy(), camera)
            pc = transform_to_camera_frame(pc, reverse=True)

            '''shuffle mask'''
            objects_collision_class = griper_collision_classifier_2.permute(0, 2, 3, 1)[0, :, :, 0][mask].detach()
            bin_collision_class = griper_collision_classifier_2.permute(0, 2, 3, 1)[0, :, :, 1][mask].detach()
            objects_mask = background_class_2.permute(0, 2, 3, 1)[0, :, :, 0][mask].detach() <= .5
            shuffle_mask = (objects_collision_class > 0.5) | (bin_collision_class > 0.5)
            shuffle_mask = shuffle_mask & objects_mask

            ''''train ref generator'''
            seperation_criteria,gripper_pose_ref=self.step_ref_generator_training(depth,mask,gripper_pose,shuffle_mask)
            if not seperation_criteria:continue

            if i % 25 == 0 and i != 0:
                gripper_poses = gripper_pose[0].permute(1, 2, 0)[mask].detach()  # .cpu().numpy()
                self.view_result(gripper_poses[shuffle_mask])
                self.export_check_points()
                self.save_statistics()
                gripper_pose2 = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask][shuffle_mask]
                for p in range(3, 7):
                    subindexes=torch.randperm(gripper_pose2.size(0))
                    if subindexes.shape[0]>1000:subindexes=subindexes[:1000]
                    values = gripper_pose2[subindexes, p:p + 1].detach().clone().cpu()
                    print(f'Mean separation dist ({p}): {torch.cdist(values, values, p=1).mean()}')

            '''background detection head'''
            bin_mask=self.analytical_bin_mask(pc,file_ids)
            if bin_mask is None: continue

            '''zero grad'''
            self.gan.critic.zero_grad()
            self.gan.generator.zero_grad()

            train_generator,tracked_indexes=self.step_discriminator_learning( depth, pc, mask, bin_mask, gripper_pose, gripper_pose_ref,
                                        griper_collision_classifier_2)
            if not train_generator:continue

            # if speculated_generator_loss<1e-4:
            #     print('pass generator training',f' critic loss: {loss.item()}')
            #     continue # train the discriminator

            '''zero grad'''
            self.gan.critic.zero_grad()
            self.gan.generator.zero_grad()
            
            '''generated grasps'''
            gripper_pose, suction_direction, griper_collision_classifier, suction_quality_classifier, shift_appealing_classifier,background_class,depth_features = self.gan.generator(
                depth.clone(),seed=seed,detach_backbone=detach_backbone)

            # assert (gripper_pose.detach()-early_sampled_poses).sum()==0, f'{(gripper_pose.detach()-early_sampled_poses).sum()==0}'

            assert not torch.isnan(gripper_pose).any(), f'{gripper_pose}'

            # if detach_beta: gripper_pose[:, 3:5] = gripper_pose[:, 3:5].detach()
            # if detach_dist: gripper_pose[:, -2] = gripper_pose[:, -2].detach()
            # if detach_width: gripper_pose[:, -1] = gripper_pose[:, -1].detach()

            '''loss computation'''
            suction_loss=suction_quality_classifier.mean()*0.0
            gripper_loss=griper_collision_classifier.mean()*0.0
            shift_loss=shift_appealing_classifier.mean()*0.0
            background_loss=background_class.mean()*0.0
            suction_sampling_loss = suction_direction.mean()*0.0

            non_zero_background_loss_counter=0

            gripper_sampling_loss=self.get_generator_loss( depth, mask, gripper_pose,shuffle_mask, gripper_pose_ref, tracked_indexes)
            print(f'generator loss= {gripper_sampling_loss.item()}')

                # gripper_sampling_loss-=weight*pred_/m
            self.gripper_sampler_statistics.loss=gripper_sampling_loss.item()


            suction_sampling_loss += suction_sampler_loss(pc, suction_direction.permute(0, 2, 3, 1)[0][mask],file_index=file_ids[0])

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

            gripper_poses=gripper_pose[0].permute(1,2,0)[mask].detach()#.cpu().numpy()
            suction_head_predictions=suction_quality_classifier[0, 0][mask]
            gripper_head_predictions=griper_collision_classifier[0, :].permute(1,2,0)[mask]
            shift_head_predictions = shift_appealing_classifier[0, 0][mask]
            background_class_predictions = background_class.permute(0,2, 3, 1)[0, :, :, 0][mask]

            '''limits'''
            with torch.no_grad():
                normals = suction_direction[0].permute(1, 2, 0)[mask].detach().cpu().numpy()
                objects_mask = bin_mask <= 0.5
                # objects_collision_max_score = torch.max(griper_collision_classifier[:,0]).item()
                # objects_collision_score_range = (objects_collision_max_score - torch.min(griper_collision_classifier[:,0])).item()
                # bin_collision_max_score = torch.max(griper_collision_classifier[:,1]).item()
                # bin_collision_score_range = (bin_collision_max_score - torch.min(griper_collision_classifier[:,1])).item()
                # suction_head_max_score = torch.max(suction_quality_classifier).item()
                # suction_head_score_range = (suction_head_max_score - torch.min(suction_quality_classifier)).item()
                # shift_head_max_score = torch.max(shift_appealing_classifier).item()
                # shift_head_score_range = (shift_head_max_score - torch.min(shift_appealing_classifier)).item()


            if bin_mask is None:
                print(Fore.RED,f'Failed to generate label for background segmentation, file id ={file_ids[0]}',Fore.RESET)
            else:
                label = torch.from_numpy(bin_mask).to(background_class_predictions.device).float()
                background_loss+=bce_loss(background_class_predictions,label)
                self.background_detector_statistics.update_confession_matrix(label,background_class_predictions.detach())
                non_zero_background_loss_counter+=1

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
                gripper_target_index=balanced_sampling(gripper_head_predictions[:,0], mask=objects_mask, exponent=10.0, balance_indicator=self.objects_collision_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = gripper_head_predictions[gripper_target_index]
                gripper_target_pose = gripper_poses[gripper_target_index]
                gripper_loss+=gripper_object_collision_loss(gripper_target_pose, gripper_target_point, pc,objects_mask, gripper_prediction_,self.objects_collision_statistics )/batch_size

            for k in range(batch_size):
                '''gripper-bin collision'''
                gripper_target_index=balanced_sampling(gripper_head_predictions[:,1], mask=objects_mask, exponent=10.0, balance_indicator=self.bin_collision_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = gripper_head_predictions[gripper_target_index]
                gripper_target_pose = gripper_poses[gripper_target_index]
                gripper_loss+=gripper_bin_collision_loss(gripper_target_pose, gripper_target_point, pc,objects_mask, gripper_prediction_,self.bin_collision_statistics )/batch_size

            for k in range(batch_size):
                '''suction seal head'''
                suction_target_index=balanced_sampling(suction_head_predictions, mask=objects_mask, exponent=10.0, balance_indicator=self.suction_head_statistics.label_balance_indicator)
                suction_prediction_ = suction_head_predictions[suction_target_index]
                suction_loss+=suction_seal_loss(pc,normals,suction_target_index,suction_prediction_,self.suction_head_statistics,objects_mask)/batch_size

            for k in range(batch_size):
                '''shift affordance head'''
                shift_target_index=balanced_sampling(shift_head_predictions, mask=None, exponent=10.0, balance_indicator=self.shift_head_statistics.label_balance_indicator)
                shift_target_point = pc[shift_target_index]
                shift_prediction_=shift_head_predictions[shift_target_index]
                shift_loss+=shift_affordance_loss(pc,shift_target_point,objects_mask,self.shift_head_statistics,shift_prediction_)/batch_size

            if non_zero_background_loss_counter: background_loss/non_zero_background_loss_counter


            loss=(suction_loss*1+gripper_loss*1+shift_loss*1+gripper_sampling_loss*10.0+suction_sampling_loss+background_loss*10.0)
            loss.backward()
            self.gan.generator_optimizer.step()
            self.gan.generator_optimizer.zero_grad()

            with torch.no_grad():
                self.gripper_sampler_statistics.loss = gripper_sampling_loss.item()
                self.suction_sampler_statistics.loss = suction_sampling_loss.item()
                self.suction_head_statistics.loss = suction_loss.item()
                self.shift_head_statistics.loss = shift_loss.item()
                self.background_detector_statistics.loss=background_loss.item()



        pi.end()

        self.view_result(gripper_poses[shuffle_mask])

        self.export_check_points()
        self.clear()

    def view_result(self,values):
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

            print(f'gripper_pose sample = {values[np.random.randint(0,values.shape[0])].cpu()}')
            print(f'gripper_pose std = {torch.std(values, dim=0).cpu()}')
            print(f'gripper_pose mean = {torch.mean(values, dim=0).cpu()}')
            print(f'gripper_pose max = {torch.max(values, dim=0)[0].cpu()}')
            print(f'gripper_pose min = {torch.min(values, dim=0)[0].cpu()}')

            self.moving_collision_rate.view()
            self.moving_firmness.view()
            self.moving_out_of_scope.view()
            self.relative_sampling_timing.view()

    def save_statistics(self):
        self.moving_collision_rate.save()
        self.moving_firmness.save()
        self.moving_out_of_scope.save()
        self.relative_sampling_timing.save()

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
    lr = 5e-5
    train_action_net = TrainActionNet( n_samples=None, learning_rate=lr)
    train_action_net.initialize(n_samples=100)
    torch.cuda.empty_cache()

    train_action_net.begin()
    for i in range(1000):
        try:
            cuda_memory_report()
            train_action_net.initialize(n_samples=None)
            train_action_net.begin()
        except Exception as error_message:
            torch.cuda.empty_cache()
            print(Exception,error_message)
