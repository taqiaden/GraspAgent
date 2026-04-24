import copy
import os
import random
import time
import traceback
from abc import abstractmethod
from collections import deque
import torch
from colorama import Fore
from matplotlib import pyplot as plt
from GraspAgent_2.utils.Voxel_operations import crop_cube, view_3d_occupancy_grid
from lib.cuda_utils import cuda_memory_report
from lib.loss.D_loss import binary_l1
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from GraspAgent_2.utils.Online_clustering import OnlingClustering
from GraspAgent_2.utils.dynamic_dataset import DynamicDataManagement, SynthesisedData
from Online_data_audit.data_tracker import DataTracker
from records.training_satatistics import MovingRate, TrainingTracker
from torch.distributions import Categorical
import spconv.pytorch as spconv
from torch_scatter import scatter_mean
import numpy as np
from visualiztion import view_npy_open3d

def hinge_loss(positive, negative, margin, k=1.):
    loss = torch.clamp((negative.squeeze() - positive.squeeze()) + margin * k, 0.)
    return loss

def c_loss(pred, label):
    loss = (1 * binary_l1(pred + 0.5, label))  # **2
    # loss=bce_with_logits(pred,label)
    # loss=sigmoid_focal_loss(pred, label, gamma=2.0, alpha=0.5)
    return loss

def logits_to_probs(logits):
    return torch.clamp(logits + 0.5, 0, 1)
    # return F.sigmoid(logits)
    # return torch.clamp(logits,0,1)

def weighted_scatter_loss(x, weights,eps=1e-6):

    N, M = x.shape

    if N > 1000:
        idx = torch.randperm(N, device=x.device)[:1000]
        x = x[idx]
        weights=weights[idx]
        N = 1000

    weights=weights/(weights.sum()+eps)

    weights=weights[:,None]*weights[None,:]

    diff=x[:,None,:]-x[None,:,:]

    dist2=diff**2

    weighted=weights[:,:,None]*dist2

    loss=-weighted.sum()/(weights.sum()*x.shape[1]+eps)

    return loss


def random_sampling(prob, mask=None):
    with torch.no_grad():

        if mask is None:
            dist = Categorical(probs=prob)
        else:
            dist = MaskedCategorical(probs=prob, mask=mask)

        target_index = dist.sample()

        return target_index

def visualize_depth_with_flat_index(depth, i):
    """
    depth: (H, W) depth map, e.g. (600, 600)
    i: index into depth.reshape(-1)
    """
    H, W = depth.shape

    # Convert flat index back to 2D index
    row, col = np.unravel_index(i, (H, W))

    plt.figure(figsize=(6, 6))
    plt.imshow(depth, cmap='viridis')
    plt.colorbar(label='Depth')

    # Highlight the selected point
    plt.scatter(col, row, c='red', s=80, marker='x')

    plt.title(f"Flat index {i} → (row={row}, col={col})")
    plt.axis('off')
    plt.show()

class AbstractGraspAgentTraining:
    def __init__(self, args, n_samples=None, epochs=1 ,model_key='test',
                 test_mode=False,pose_interpolation=None,process_pose=None,
                 n_param=1,track_statistics_history=False):

        self.args = args
        self.model_key=model_key
        self.test_mode=test_mode
        self.max_n=5 if test_mode else 30
        self.train_policy_only=False
        self.view=False
        self.synthesizie_only=False
        
        '''hand specific fucntions'''
        self.pose_interpolation=pose_interpolation
        self.process_pose=process_pose
        self.gan =None
        self.sim_env=None

        ''''''
        self.track_statistics_history=track_statistics_history

        
        self.batch_size = 2
        self.max_G_norm = 500
        self.max_D_norm = 100
        self.iter_per_scene = 1

        self.n_samples = n_samples
        self.size = n_samples
        self.epochs = epochs

        '''Moving rates'''
        self.moving_collision_rate = None
        self.skip_rate = None
        self.superior_A_model_moving_rate = None
        self.G_grad_norm_MR = None
        self.D_grad_norm_MR = None
        self.bin_collision_statistics = None
        self.objects_collision_statistics = None
        self.collision_statistics = None
        self.grasp_quality_statistics = None

        '''initialize statistics records'''
        self.gripper_sampler_statistics = None
        self.critic_statistics = None
        self.critic_statistics = None
        self.data_tracker = None

        self.tmp_pose_record = []

        self.n_param = n_param

        self.max_scenes = 1000


        root_dir = os.getcwd()  # current working directory

        self.tou = 1


        self.DDM = DynamicDataManagement(key=self.model_key + '_synthesized_dynamic_data')

        self.loaded_synthesised_data = None

        self.skipped_last=True

        # self.vectors_clusters=OnlingClustering(key_name=self.model_key+'_vectors_clusters',number_of_centers=16,vector_size=5,decay_rate=0.01,use_euclidean_dist=False)
        approach_centers = torch.tensor([[0., 1., 0],[0., -1., 0],[1., 0, 0],[-1., 0, 0],[0., 0, -1]], device='cuda')
        beta_centers=torch.tensor([[0., 1],[0., -1],[1., 0],[-1., 0]], device='cuda')
        # Repeat approach_centers n_beta times
        alpha_repeated = approach_centers.repeat_interleave(beta_centers.shape[0] , dim=0)  # (20, 3)

        # Tile beta_centers n_alpha times
        beta_tiled = beta_centers.repeat(approach_centers.shape[0] , 1)  # (20, 2)

        # Concatenate along dimension 1
        alpha_beta = torch.cat([alpha_repeated, beta_tiled], dim=1)  # (20, 5)

        # print(alpha_beta)
        # exit()

        self.approach_beta_clusters=OnlingClustering(key_name=self.model_key+'_approach_beta_clusters',number_of_centers=8,vector_size=5,decay_rate=0.01,use_euclidean_dist=False,static_centers=alpha_beta)


    def initialize(self, n_samples=None):
        self.n_samples = n_samples

        '''Moving rates'''
        self.moving_collision_rate = MovingRate(self.model_key + '_collision', decay_rate=0.01,
                                                initial_val=1.,track_history=self.track_statistics_history)
        self.skip_rate = MovingRate(self.model_key + '_skip_rate',
                                    decay_rate=0.1,
                                    initial_val=1.,track_history=self.track_statistics_history)
        self.superior_A_model_moving_rate = MovingRate(self.model_key + '_superior_A_model',
                                                       decay_rate=0.01,
                                                       initial_val=0.,track_history=self.track_statistics_history)

        self.G_grad_norm_MR = MovingRate(self.model_key + '_G_grad_norm',
                                         decay_rate=0.01,
                                         initial_val=0.,track_history=self.track_statistics_history)

        self.D_grad_norm_MR = MovingRate(self.model_key + '_D_grad_norm',
                                         decay_rate=0.01,
                                         initial_val=0.,track_history=self.track_statistics_history)

        self.Ave_samples_per_scene = MovingRate(self.model_key + '_Ave_samples_per_scene',
                                                decay_rate=0.01,
                                                initial_val=0.,track_history=self.track_statistics_history)
        self.Ave_importance = MovingRate(self.model_key + '_Ave_importance',
                                         decay_rate=0.01,
                                         initial_val=0.,track_history=self.track_statistics_history)

        self.Ave_uniquness = MovingRate(self.model_key + 'Ave_uniquness',
                                                       decay_rate=0.01,
                                                       initial_val=0.,load_last=True,track_history=self.track_statistics_history)

        # self.superior_A_model_moving_rate.moving_rate=0
        # self.superior_A_model_moving_rate.save()
        # exit()

        '''initialize statistics records'''
        self.bin_collision_statistics = TrainingTracker(name=self.model_key + '_bin_collision',
                                                        track_label_balance=True,track_history=self.track_statistics_history)
        self.objects_collision_statistics = TrainingTracker(name=self.model_key + '_objects_collision',
                                                            track_label_balance=True,track_history=self.track_statistics_history)
        self.collision_statistics = TrainingTracker(name=self.model_key + '_collision',
                                                    track_label_balance=True, decay_rate=0.01,track_history=self.track_statistics_history)

        self.gripper_sampler_statistics = TrainingTracker(name=self.model_key + '_gripper_sampler',
                                                          track_label_balance=False,track_history=self.track_statistics_history)

        self.grasp_quality_statistics = TrainingTracker(name=self.model_key + '_grasp_quality',
                                                        track_label_balance=True, decay_rate=0.01,track_history=self.track_statistics_history)

        self.critic_statistics = TrainingTracker(name=self.model_key + '_critic',
                                                 track_label_balance=False,track_history=self.track_statistics_history)

        self.data_tracker = DataTracker(name=self.model_key)

    @abstractmethod
    def prepare_model_wrapper(self):
        pass

    def step_discriminator(self, cropped_spheres, depth, clean_depth, gripper_pose, gripper_pose_ref, pairs, floor_mask,
                           grasp_quality, latent_vector):

        '''zero grad'''
        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.critic.zero_grad()
        self.gan.critic_optimizer.zero_grad()

        '''self supervised critic learning'''
        with torch.no_grad():

            generated_grasps_stack = []
            for pair in pairs:
                index = pair[0]

                pred_pose = gripper_pose[index]
                label_pose = gripper_pose_ref[index]
                pair_pose = torch.stack([pred_pose, label_pose])
                generated_grasps_stack.append(pair_pose)
            generated_grasps_stack = torch.stack(generated_grasps_stack)

        anchor, positive_negative, scores = self.gan.critic(clean_depth[None, None, ...], generated_grasps_stack, pairs,
                                                            ~floor_mask.view(1, 1, 600, 600), cropped_spheres,
                                                            latent_vector=latent_vector  )


        generated_scores = scores[:, 0]
        ref_scores = scores[:, 1]

        loss = torch.tensor(0., device=depth.device)

        for j in range(len(pairs)):
            target_index = pairs[j][0]
            k = pairs[j][1]
            margin = pairs[j][2]

            assert margin >= 0 , f'margin=,{margin}'

            if k > 0:
                if ref_scores[j]-generated_scores[j]<=1:
                    loss += (hinge_loss(positive=ref_scores[j], negative=generated_scores[j],
                                        margin=margin)) / self.batch_size
                else:
                    loss += (hinge_loss(positive=generated_scores[j], negative=ref_scores[j],
                                        margin=-1)) / self.batch_size
            else:
                loss += (hinge_loss(positive=generated_scores[j], negative=ref_scores[j],
                                    margin=margin) ) / self.batch_size

        loss.backward()

        self.critic_statistics.loss = loss.item()

        '''GRADIENT CLIPPING'''
        params = list(self.gan.critic.back_bone.parameters())
        backbone_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf'))

        params = list(self.gan.critic.att_block.parameters())

        decoder_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf'))

        norm = torch.nn.utils.clip_grad_norm_(self.gan.critic.parameters(), max_norm=float('inf'))

        self.D_grad_norm_MR.update(norm.item())

        print(Fore.LIGHTGREEN_EX, f' D  norm : {norm}, backbone_norm:{backbone_norm}, decoder_norm={decoder_norm}',
              Fore.RESET)

        self.gan.critic_optimizer.step()

        self.gan.critic.zero_grad()
        self.gan.critic_optimizer.zero_grad()

        print(Fore.LIGHTYELLOW_EX, f'd_loss={loss.item()}',
              Fore.RESET)

    def print_pairs_info(self, pairs, gripper_pose, gripper_pose_ref):
        for j in range(len(pairs)):
            target_index = pairs[j][0]
            k = pairs[j][1]
            margin = pairs[j][2]

            target_generated_pose = gripper_pose[target_index].detach()
            target_ref_pose = gripper_pose_ref[target_index].detach()

            if k < 0:
                print(Fore.LIGHTGREEN_EX,
                      f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()} , m={margin} ',
                      Fore.RESET)
            elif k > 0:
                print(Fore.LIGHTCYAN_EX,
                      f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()} , m={margin} ',
                      Fore.RESET)
            # else:
            #     print(Fore.LIGHTGREEN_EX,f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()} ',Fore.RESET)

    def get_generator_loss(self, cropped_spheres, depth, clean_depth, gripper_pose, gripper_pose_ref, pairs, floor_mask,
                           latent_vector):

        gripper_pose = gripper_pose[0].permute(1, 2, 0).reshape(360000, self.n_param)
        gripper_pose_ref = gripper_pose_ref[0].permute(1, 2, 0).reshape(360000, self.n_param)

        generated_grasps_stack = []
        for pair in pairs:
            index = pair[0]
            pred_pose = gripper_pose[index]

            label_pose = gripper_pose_ref[index]
            pair_pose = torch.stack([pred_pose, label_pose])
            generated_grasps_stack.append(pair_pose)

        generated_grasps_stack = torch.stack(generated_grasps_stack)

        anchor, positive_negative, scores = self.gan.critic(clean_depth[None, None, ...], generated_grasps_stack, pairs,
                                                            ~floor_mask.view(1, 1, 600, 600), cropped_spheres,
                                                            latent_vector=latent_vector, detach_backbone=True)

        gen_scores = scores[:, 0]
        ref_scores = scores[:, 1]

        loss = torch.tensor(0., device=depth.device)

        for j in range(len(pairs)):
            loss += (hinge_loss(positive=gen_scores[j], negative=ref_scores[j], margin=0.) ) / self.batch_size

        return loss

    def step_generator(self, cropped_spheres, depth, clean_depth, floor_mask, pc, gripper_pose_ref, pairs,
                       latent_vector):
        '''zero grad'''
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.generator_optimizer.zero_grad(set_to_none=True)
        self.gan.sampler_optimizer.zero_grad(set_to_none=True)

        '''generated grasps'''
        gripper_pose, grasp_quality_logits, grasp_collision_logits, features, model_B_quality_logits = self.gan.generator(
            depth[None, None, ...], ~floor_mask.view(1, 1, 600, 600), latent_vector,
            model_B_poses=gripper_pose_ref)

        gripper_pose_PW = gripper_pose[0].permute(1, 2, 0).reshape(360000, self.n_param)
        collision_logits = grasp_collision_logits[0, 2].reshape(-1)
        grasp_quality_logits = grasp_quality_logits[0, 0].reshape(-1)

        '''loss computation'''
        gripper_collision_loss = torch.tensor(0., device=depth.device)
        gripper_quality_loss_ = torch.tensor(0., device=depth.device)
        n = 2

        start = time.time()
        positive_counter = 0
        negative_counter = 0
        s = int(n / 2)
        for k in range(n):
            '''gripper collision'''
            while True:

                gripper_target_index = random_sampling(torch.rand_like(collision_logits),
                                                       mask=~floor_mask.detach())
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = collision_logits[gripper_target_index].squeeze()
                gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

                contact_with_obj, contact_with_floor = self.check_collision(gripper_target_point, gripper_target_pose,
                                                                            view=False)
                collision = contact_with_obj or contact_with_floor
                label = torch.ones_like(gripper_prediction_) if collision else torch.zeros_like(gripper_prediction_)

                object_collision_loss = c_loss(gripper_prediction_, label)

                if time.time() - start > 5 or self.skip_rate.val > 0.9:
                    print(Fore.RED, f'collision classifier exploration timeout', Fore.RESET)
                    break
                if collision and positive_counter >= s: continue
                if (not collision) and negative_counter >= s: continue
                break

            with torch.no_grad():
                self.collision_statistics.loss = object_collision_loss.item()
                self.collision_statistics.update_confession_matrix(label.detach(),
                                                                   logits_to_probs(gripper_prediction_.detach()))
            if collision:
                positive_counter += 1
            else:
                negative_counter += 1
            gripper_collision_loss += object_collision_loss / (n)

       

        floor_mean_score = grasp_quality_logits[floor_mask].mean()#.item()
        obj_mean_score = grasp_quality_logits[~floor_mask].mean().item()

        MDL = ((grasp_quality_logits[floor_mask] - floor_mean_score) ** 2).mean() * 100
        floor_loss=(torch.clamp(grasp_quality_logits[floor_mask]+0.5,min=0.)**2).mean()


        print(
            f'--------------------------------------------------- floor_mean_score : {floor_mean_score}, obj_mean_score: {obj_mean_score}')

        start = time.time()
        positive_counter = 0
        negative_counter = 0
        n = 2
        s = int(n / 2)
        for k in range(n):
            '''grasp quality'''
            while True:
                # gripper_target_index = balanced_sampling(logits_to_probs(grasp_quality_logits.detach()),
                #                                          mask=~floor_mask.detach(),
                #                                          exponent=2.0,
                #                                          balance_indicator=self.grasp_quality_statistics.label_balance_indicator)
                gripper_target_index = random_sampling(torch.rand_like(collision_logits),
                                                       mask=~floor_mask.detach())

                # dist = MaskedCategorical(probs=logits_to_probs(grasp_quality_logits.detach().clamp(min=0.1)), mask=(~floor_mask.detach()))
                # gripper_target_index = dist.sample()

                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = grasp_quality_logits[gripper_target_index].squeeze()
                gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

                grasp_success, initial_collision, n_grasp_contact, self_collide, stable_grasp, warning_flag, plan_found, grasped_obj = self.evaluate_grasp(
                    gripper_target_point, gripper_target_pose, view=False,
                    shake=False, check_kinematics=False,
                    update_obj_prob=gripper_prediction_.item() )
                # grasp_success=grasp_success and stable_grasp
                # if not grasp_success and self.grasp_quality_statistics.label_balance_indicator<(np.random.random()**2)*-1: continue
                # else: break
                if warning_flag: continue
                if time.time() - start > 5 * s or self.skip_rate.val > 0.9:
                    print(Fore.RED, f'quality policy exploration timeout', Fore.RESET)
                    break
                # if initial_collision: continue

                label = torch.ones_like(gripper_prediction_) if grasp_success else torch.zeros_like(gripper_prediction_)
                self.grasp_quality_statistics.update_confession_matrix(label.detach(),
                                                                       logits_to_probs(gripper_prediction_.detach()))

                # elif grasp_success and not stable_grasp: continue
                # elif not grasp_success and self.grasp_quality_statistics.label_balance_indicator<(np.random.random()**2)*-1: continue
                if grasp_success and positive_counter >= s: continue
                if (not grasp_success) and negative_counter >= s: continue

                break
            if grasp_success:
                positive_counter += 1
            else:
                negative_counter += 1
            label = torch.ones_like(gripper_prediction_) if grasp_success else torch.zeros_like(gripper_prediction_)
            grasp_quality_loss = c_loss(gripper_prediction_, label)

            with torch.no_grad():
                self.grasp_quality_statistics.loss = grasp_quality_loss.item()
                self.objects_collision_statistics.update_confession_matrix(label.detach(),
                                                                           logits_to_probs(
                                                                               gripper_prediction_.detach()))

            gripper_quality_loss_ += grasp_quality_loss / (n)

            # print(f'positive_counter: {positive_counter}, negative_counter: {negative_counter}')

        def get_repulsive_loss():
            cloned_quality_decoder = copy.deepcopy(self.gan.generator.grasp_quality_)
            max_ = 1.3
            min_ = 1.15
            standarized_depth_ = (depth[None, None, ...].clone() - min_) / (max_ - min_)

            standarized_depth_ = (standarized_depth_ - 0.5) / 0.5
            gripper_pose_x = torch.cat([gripper_pose.clone(), standarized_depth_], dim=1)

            grasp_quality_x = cloned_quality_decoder(features, gripper_pose_x)

            # grasp_quality_x = (grasp_quality_x - grasp_quality_x[
            #     floor_mask.view(1, 1, 600, 600)].mean() + self.gan.generator.bias.item())*self.gan.generator.scale.item()
            grasp_quality_obj_x = grasp_quality_x[~floor_mask.view(1, 1, 600, 600)]

            # mid_range = grasp_quality_obj_x.min().clamp(min=-0.5,max=0.5).item() + (
            #             grasp_quality_obj_x.max().clamp(min=-0.5,max=0.5) - grasp_quality_obj_x.min().clamp(min=-0.5,max=0.5)).item() / 2

            # P_loss = torch.relu(0.5 - grasp_quality_obj_x[grasp_quality_obj_x >= mid_range]).mean() if (grasp_quality_obj_x >= mid_range).any() else 0.
            # N_loss = torch.relu(0.5+grasp_quality_obj_x[grasp_quality_obj_x < mid_range]).mean() if (grasp_quality_obj_x < mid_range).any() else 0.

            loss=torch.clamp(0.5-torch.abs(grasp_quality_obj_x),min=0.).mean()
            return loss

        diversity_loss=0.
        if len(pairs) == self.batch_size:

            gripper_sampling_loss = self.get_generator_loss(cropped_spheres,
                                                            depth, clean_depth, gripper_pose, gripper_pose_ref,
                                                            pairs, floor_mask, latent_vector)

            assert not torch.isnan(gripper_sampling_loss).any(), f'{gripper_sampling_loss}'

            repulsive_loss = get_repulsive_loss()  # .clamp(max=gripper_sampling_loss.item())

            orientation = gripper_pose[0, 0:5].reshape(5, -1).permute(1, 0)[~floor_mask]

            weight=(1-torch.abs(0.5-logits_to_probs(grasp_quality_logits[~floor_mask]).detach()))**2#(1-torch.abs(0.5-logits_to_probs(grasp_quality_logits[~floor_mask]).detach())*2)

            diversity_loss = weighted_scatter_loss(gripper_pose.reshape(self.n_param, -1).permute(1, 0)[~floor_mask],weights=weight) if len(
                pairs) == self.batch_size else torch.tensor(
                [0.], device=orientation.device)

            print(Fore.LIGHTYELLOW_EX,
                  f'gripper_sampling_loss={gripper_sampling_loss.item()}, gripper_quality_loss_={gripper_quality_loss_.item()}, gripper_collision_loss={gripper_collision_loss.item()}, MDL={MDL.item()}, floor_loss={floor_loss.item()} repulsive_loss={repulsive_loss.item()}, diversity_loss={diversity_loss.item()}',
                  Fore.RESET)
            with torch.no_grad():
                self.gripper_sampler_statistics.loss = gripper_sampling_loss.item()
        else:
            # repulsive_loss=get_repulsive_loss()
            repulsive_loss = 0.
            print(Fore.LIGHTYELLOW_EX,
                  f' gripper_quality_loss_={gripper_quality_loss_.item()}, gripper_collision_loss={gripper_collision_loss.item()}, MDL={MDL.item()}, floor_loss={floor_loss.item()}, repulsive_loss={repulsive_loss}',
                  Fore.RESET)
            gripper_sampling_loss = 0.

        loss = (gripper_sampling_loss + gripper_collision_loss + gripper_quality_loss_) +diversity_loss#+floor_loss#+repulsive_loss

        # if abs(loss.item())>0.0:
        # try:
        loss.backward()

        '''GRADIENT CLIPPING'''
        # params=list(self.gan.generator.back_bone.parameters()) + \
        #  list(self.gan.generator.CH_PoseSampler.parameters())
        norm = torch.nn.utils.clip_grad_norm_(self.gan.generator.parameters(), max_norm=float('inf'))
        self.G_grad_norm_MR.update(norm.item())

        norm1 = torch.nn.utils.clip_grad_norm_(self.gan.generator.back_bone.parameters(), max_norm=float('inf'))
        norm2 = torch.nn.utils.clip_grad_norm_(self.gan.generator.back_bone2_.parameters(), max_norm=float('inf'))
        # norm3=torch.nn.utils.clip_grad_norm_(self.gan.generator.back_bone3_.parameters(), self.max_norm=float('inf'))

        print(Fore.LIGHTGREEN_EX, f' G norm : {norm}, backbone1:{norm1}, backbone2: {norm2}, ', Fore.RESET)

        # params2 = list(self.gan.generator.back_bone.parameters()) + \
        #          list(self.gan.generator.grasp_quality.parameters())+ \
        #          list(self.gan.generator.grasp_collision.parameters())+ \
        #          list(self.gan.generator.grasp_collision2_.parameters())+ \
        #          list(self.gan.generator.grasp_collision3_.parameters())
        # norm = torch.nn.utils.clip_grad_norm_(params2, self.max_norm=float('inf'))
        # print(Fore.LIGHTGREEN_EX, f' G norm 2 : {norm}', Fore.RESET)

        self.gan.generator_optimizer.step()
        self.gan.sampler_optimizer.step()

        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator_optimizer.zero_grad(set_to_none=True)
        self.gan.critic_optimizer.zero_grad(set_to_none=True)
        self.gan.sampler_optimizer.zero_grad(set_to_none=True)

    def check_collision(self, target_point, target_pose, view=False):
        with torch.no_grad():
            quat, fingers, shifted_point = self.process_pose(target_point, target_pose, view=view)

        return self.sim_env.check_collision(hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers, view=view)

    def evaluate_grasp(self, target_point, target_pose, view=False, hard_level=0, shake=False, check_kinematics=True,
                       update_obj_prob=None):
        grasped_obj = None
        with torch.no_grad():
            quat, fingers, shifted_point = self.process_pose(target_point, target_pose, view=self.test_mode)

            if view:
                in_scope, grasp_success, contact_with_obj, contact_with_floor, n_grasp_contact, self_collide, stable_grasp = self.sim_env.view_grasp(
                    hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers,
                    view=view, hard_level=hard_level)
                warning_flag = False
            else:
                in_scope, grasp_success, contact_with_obj, contact_with_floor, n_grasp_contact, self_collide, stable_grasp, warning_flag, grasped_obj = self.sim_env.check_graspness(
                    hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers,
                    view=view, hard_level=hard_level, shake=shake, update_obj_prob=update_obj_prob)

            initial_collision = contact_with_obj or contact_with_floor

            if warning_flag: print(Fore.RED, f' ----------------------------- warning_flag', Fore.RESET)

            # print('in_scope, grasp_success, contact_with_obj, contact_with_floor :',in_scope, grasp_success, contact_with_obj, contact_with_floor )

            if grasp_success is not None:
                if grasp_success and not contact_with_obj and not contact_with_floor:
                    plan_found = self.kinematics.kinematic_plan_exist(quat, shifted_point) if check_kinematics else True
                    return grasp_success, initial_collision, n_grasp_contact, self_collide, stable_grasp, warning_flag, plan_found, grasped_obj

        return False, initial_collision, n_grasp_contact, self_collide, stable_grasp, warning_flag, None, grasped_obj

    def sample_contrastive_pairs(self, pc, floor_mask, gripper_pose, gripper_pose_ref,
                                 annealing_factor, grasp_quality, grasp_collision,
                                 superior_A_model_moving_rate, latent_vector, model_b_quality):
        start = time.time()

        d_pairs = []
        g_pairs = []

        all_pairs = []

        selection_mask = (~floor_mask)  # & (latent_vector.reshape(-1)!=0)
        grasp_quality = grasp_quality[0, 0].reshape(-1)
        model_b_quality = model_b_quality[0, 0].reshape(-1)
        grasp_collision = grasp_collision[0, 2].reshape(-1)
        gripper_pose_PW = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000, self.n_param)
        clipped_gripper_pose_PW = gripper_pose_PW.clone()
        clipped_gripper_pose_PW[:, 5:5 + 3] = torch.clip(clipped_gripper_pose_PW[:, 5:5 + 3], 0, 1)
        gripper_pose_ref_PW = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000, self.n_param)


        selection_p = torch.rand_like(
            grasp_quality)   if  len(self.DDM)<self.max_scenes else grasp_quality.clamp(min=0.001)
        if self.test_mode: selection_p = 0.001  + grasp_quality ** 2

        avaliable_iterations = selection_mask.sum()
        if avaliable_iterations < 3: return [], [], None

        initial_mask = (grasp_quality > 0.5) | (model_b_quality > 0.5)
        initial_mask = initial_mask & selection_mask
        high_quality_samples = initial_mask.sum()

        n = int(min(self.max_n, avaliable_iterations))

        print(Fore.LIGHTBLACK_EX, '# Available candidates =', avaliable_iterations.item(), ' high_quality_samples =',
              high_quality_samples.item(), Fore.RESET)

        counter = 0
        counter2 = 0

        sampler_samples = 0

        sampled_obj_ids = []

        t = 0
        while t < n:
            # print(t)
            time_out = time.time() - start
            if time_out > 5 and not self.test_mode: break
            t += 1
            # if torch.any(initial_mask):
            #     dist = MaskedCategorical(probs=selection_p, mask=initial_mask)
            # else:
            # ideal_ref=False
            importance = None
            if self.loaded_synthesised_data is not None and len(self.loaded_synthesised_data) > 0 :
                # target_index=preferred_indexes.pop()
                target_index, _, _, importance, _ = self.loaded_synthesised_data.sample_pop()
                # print(f'test ----- target_index={target_index}, ideal pose = {gripper_pose_ref_PW[target_index]}')
                # ideal_ref=True
            else:
                # if len(all_pairs)>=self.batch_size and len(self.DDM)>=self.max_scenes:break
                dist = MaskedCategorical(probs=selection_p, mask=selection_mask)
                target_index = torch.argmax(dist.probs).item() #if grasp_quality.max()<1.0  else dist.sample().item()
                # target_index = dist.sample().item()

            selection_mask[target_index] *= False
            initial_mask[target_index] *= False

            avaliable_iterations -= 1
            target_point = pc[target_index]

            target_generated_pose = gripper_pose_PW[target_index]
            target_ref_pose = gripper_pose_ref_PW[target_index]

            margin = 1.

            # assert target_ref_pose[0]>0,f'{target_ref_pose}'

            if self.test_mode:
                contact_with_obj, contact_with_floor = self.check_collision(target_point, target_ref_pose,
                                                                            view=True)
                if contact_with_obj or contact_with_floor: continue
                # contact_with_obj , contact_with_floor=self.check_collision(target_point, target_generated_pose,view=True)

                view_r = self.evaluate_grasp(
                    target_point, target_ref_pose, view=True, shake=True)
                #
                print(Fore.LIGHTCYAN_EX,
                      f'return f1: {view_r}, quality_score: {grasp_quality[target_index].item()}, max score={grasp_quality.max().item()}')
                # view_r2 = self.evaluate_grasp(
                #     target_point, target_generated_pose, view=False, shake=True)
                # print(Fore.LIGHTCYAN_EX,f'return f2: {view_r2}')

                g_pairs.append((target_index, 1, 1))
                d_pairs.append((target_index, 1, 1))
                return d_pairs, g_pairs, 1

            ref_quality = model_b_quality[target_index]
            gen_quality = grasp_quality[target_index]

            ref_success, ref_initial_collision, ref_n_grasp_contact, ref_self_collide, stable_ref_grasp, warning_flag, ref_plan_found, ref_grasped_obj = self.evaluate_grasp(
                target_point, target_ref_pose, view=False, shake=False, update_obj_prob=None, check_kinematics=False)
            # if ideal_ref : assert ref_success
            # if ref_success and (not ref_plan_found): continue
            # if ref_success and ( not stable_ref_grasp): continue

            # ref_success=ref_success   and ref_plan_found

            if warning_flag:
                break
                # continue
            gen_success, gen_initial_collision, gen_n_grasp_contact, gen_self_collide, stable_gen_grasp, warning_flag, gen_plan_found, gen_grasped_obj = self.evaluate_grasp(
                target_point, target_generated_pose, view=False, shake=False, check_kinematics=False,
                update_obj_prob=gen_quality.item())
            # if gen_success and (not gen_plan_found): continue
            # gen_success=gen_success and stable_gen_grasp

            # print(f'ref_success:{ref_success}, gen_success:{gen_success}')

            if warning_flag:
                break

            if t == 1 and self.skip_rate() > 0.9:
                print(
                    f' ref ---- {target_ref_pose}, {ref_success, ref_initial_collision, ref_n_grasp_contact, ref_self_collide}')
                print()
                print(
                    f' gen ---- {target_generated_pose}, {gen_success, gen_initial_collision, gen_n_grasp_contact, gen_self_collide}')


            if gen_success:

                if importance is None or not ref_success or grasp_quality[target_index].item()>importance:

                    importance = max(0.01,
                                     grasp_quality[target_index].item())# if importance is None else
                                                                                                   # grasp_quality[
                                                                                                   #     target_index].item())


                    all_pairs.append(
                        (target_index, target_point, gripper_pose_PW[target_index], importance, gen_grasped_obj))
                else:
                    importance *= 0.5
                    all_pairs.append(
                        (target_index, target_point, gripper_pose_ref_PW[target_index], importance, ref_grasped_obj))

            elif ref_success:
                # k_dist = (1 - F.cosine_similarity(target_generated_pose[0:5], target_ref_pose[0:5], dim=-1)) / 2
                # q = (1 - k_dist.item()) * grasp_quality[target_index].item()
                #     target_ref_pose[0:5].detach()).item()
                # if importance is None else max(0.01, q)

                if (importance is not None and importance>0.1) or len(self.DDM)<self.max_scenes:
                    importance = 0.5*importance if importance is not None else grasp_quality[target_index].item()
                    all_pairs.append(
                        (target_index, target_point, gripper_pose_ref_PW[target_index], importance, ref_grasped_obj))

                if ref_grasped_obj in sampled_obj_ids:
                    if len(self.loaded_synthesised_data) > 0: continue
                else:
                    sampled_obj_ids.append(sampled_obj_ids)

            # if ref_success == gen_success:
            if not ref_success and not gen_success:
                self.sim_env.update_obj_info(1e-2, decay=0.9)
                continue
            elif ref_success and not gen_success:
                k=1
            elif gen_success and not ref_success:
                k=-1
            else:
                continue



            if k == 1:
                sampler_samples += 1

            counter += 1
            t = 0
            hh = (counter / self.batch_size) ** 2
            n = int(min(hh * self.max_n + n, avaliable_iterations))



            if ref_success and not gen_success:  # and counter2 < self.batch_size:
                superior_A_model_moving_rate.update(0.)
                counter2 += 1
            elif gen_success and not ref_success:  # and counter2 < self.batch_size:
                superior_A_model_moving_rate.update(1.)
                counter2 += 1
            else: continue

            if len(d_pairs) < self.batch_size and not (ref_success and gen_success):

                margin=0. if gen_initial_collision or ref_initial_collision else (1-grasp_quality[target_index])

                d_pairs.append((target_index, k, margin,  target_point))


            if len(g_pairs) < self.batch_size and ref_success and not gen_success:

                g_pairs.append((target_index, k, margin, target_point))

            self.tmp_pose_record.append(target_generated_pose.detach().clone())

            superior_pose = target_ref_pose if k > 0 else target_generated_pose

            self.approach_beta_clusters.update(superior_pose[0:5].detach().clone())




            if len(d_pairs) == self.batch_size and len(g_pairs) == self.batch_size: break

        if len(all_pairs) > 0:
            '''Update dynamic data'''
            self.sim_env.restore_simulation_state()
            synthesised_data_obj = SynthesisedData()
            synthesised_data_obj.obj_ids = self.sim_env.objects
            synthesised_data_obj.obj_poses = self.sim_env.objects_poses

            assert 7 * len(self.sim_env.objects) == len(self.sim_env.objects_poses)

            for pair in all_pairs:
                target_index, target_point, pose, importance, grasped_object = pair


                target_point = pc[target_index]

                U_alpha_beta_score = self.approach_beta_clusters.get_uniqueness_score(
                    pose[0:5].detach()).item()


                synthesised_data_obj.target_indexes.append(target_index)
                synthesised_data_obj.grasp_target_points.append(target_point.cpu().numpy())
                synthesised_data_obj.grasp_parameters.append(pose.cpu().numpy())
                synthesised_data_obj.importance.append(importance)
                synthesised_data_obj.grasped_objects.append(grasped_object)

                synthesised_data_obj.uniqueness.append(U_alpha_beta_score)

            if self.loaded_synthesised_data is not None:
                synthesised_data_obj.id = self.loaded_synthesised_data.id
                for n in range(len(self.loaded_synthesised_data.target_indexes)):
                    target_index = self.loaded_synthesised_data.target_indexes[n]
                    # if target_index in synthesised_data_obj.target_indexes: continue

                    if self.loaded_synthesised_data.grasped_objects[n] is None: continue
                    U_alpha_beta_score = self.approach_beta_clusters.get_uniqueness_score(
                        torch.tensor(self.loaded_synthesised_data.grasp_parameters[n][0:5]).cuda()).item()


                    synthesised_data_obj.target_indexes.append(target_index)
                    synthesised_data_obj.grasp_target_points.append(self.loaded_synthesised_data.grasp_target_points[n])
                    synthesised_data_obj.grasp_parameters.append(self.loaded_synthesised_data.grasp_parameters[n])
                    synthesised_data_obj.importance.append(self.loaded_synthesised_data.importance[n])
                    synthesised_data_obj.grasped_objects.append(self.loaded_synthesised_data.grasped_objects[n])

                    synthesised_data_obj.uniqueness.append(U_alpha_beta_score)

            # synthesised_data_obj.filter_best_grasps()
            if self.loaded_synthesised_data is None:
                # if max(synthesised_data_obj.importance) >= 0.5:
                if len(self.DDM)>=self.max_scenes:
                    importance, uniqueness = synthesised_data_obj.unique_obj_max_scores()
                    ave_uniqueness = sum(uniqueness) / len(uniqueness)
                    ave_impo = sum(importance) / len(importance)
                    self.Ave_importance.update(ave_impo)
                    ave_samples = len(importance)
                    self.Ave_samples_per_scene.update(ave_samples)
                    self.Ave_uniquness.update(ave_uniqueness)

                    if  not self.Ave_uniquness.lower_rejection_criteria(ave_uniqueness, k=2.):
                        self.DDM.save_data_point(synthesised_data_obj)
                        if len(self.DDM)-len(self.DDM.low_quality_samples_tracker)<self.max_scenes:
                            print(Fore.GREEN,
                                  f'Add new sample, criteria ( ave_impo, ave_uniqueness): {ave_impo, ave_uniqueness} ',
                                  Fore.RESET)

                    else:
                        d_pairs=[]
                        print(Fore.RED,
                              f'Ignore new sample, criteria ( ave_impo, ave_uniqueness): {ave_impo, ave_uniqueness} ',
                              Fore.RESET)
                else:
                    self.DDM.save_data_point(synthesised_data_obj)
            else:
                importance, uniqueness = synthesised_data_obj.unique_obj_max_scores()
                ave_uniqueness = sum(uniqueness) / len(uniqueness)
                ave_impo = sum(importance)   / len(importance)
                self.Ave_importance.update(ave_impo)
                ave_samples = len(importance)
                self.Ave_samples_per_scene.update(ave_samples)
                self.Ave_uniquness.update(ave_uniqueness)

                self.DDM.update_old_record(synthesised_data_obj)
                c_Importance = self.Ave_importance.lower_rejection_criteria(ave_impo, k=2.)
                c_Uniquness = self.Ave_uniquness.lower_rejection_criteria(ave_uniqueness, k=2.)
                c_Importance_too_confident = self.Ave_importance.upper_rejection_criteria(ave_impo, k=2.)

                if len(self.DDM) >= self.max_scenes and c_Uniquness:# ( (c_Importance and c_Uniquness) or (c_Importance_too_confident and c_Uniquness)) :
                    print(Fore.RED,
                          f'poor sample detected, criteria ( c_Importance, c_Uniquness,c_Importance_too_confident): {c_Importance, c_Uniquness,c_Importance_too_confident} ',
                          Fore.RESET)
                    self.DDM.low_quality_samples_tracker.append(self.loaded_synthesised_data.id)

            self.skip_rate.update(0.)
        else:
            self.skip_rate.update(1.)

            if self.loaded_synthesised_data is not None:
                print(Fore.LIGHTMAGENTA_EX, 'Low quality sample detected, poor samples ', Fore.RESET)

                self.DDM.low_quality_samples_tracker.append(self.loaded_synthesised_data.id)

        return d_pairs, g_pairs, sampler_samples

    def prepare_voxels(self, pairs, depth, pc, full_pointcloud, view=False):
        '''prepare cropped point clouds''''''prepare cropped point clouds'''
        # cropped_voxels = []
        cropped_spheres = []
        radius = 0.13
        batch_features_list = []
        batch_indices_list = []
        space_range = 2.0
        voxel_size = 0.02
        grid_size = int(space_range / voxel_size)
        b = 0
        for pair in pairs:
            index = pair[0]

            center = pc[index]

            # visualize_pointcloud_with_index_open3d(pc.cpu().numpy(),index)

            sub_pc = crop_cube(full_pointcloud, center, cube_size=2 * radius)
            sub_pc -= center
            sub_pc /= radius

            if view:
                visualize_depth_with_flat_index(depth.cpu().numpy(), index)
                view_npy_open3d(sub_pc.cpu().numpy(), view_coordinate=True)

            # sub_pc2=crop_cube(pc, center, cube_size=2*radius)
            # sub_pc2 -= center
            # sub_pc2/=radius
            # view_npy_open3d(sub_pc2.cpu().numpy(),view_coordinate=True)

            # Map [-1, 1] → [0, grid_size)
            coords = ((sub_pc + 1.0) / space_range * grid_size).floor().int()

            # Safety clamp
            coords = torch.clamp(coords, 0, grid_size - 1)

            # Unique voxels
            voxel_coords, inverse = torch.unique(
                coords, dim=0, return_inverse=True
            )

            # Voxel feature = mean xyz of points in that voxel
            voxel_features = scatter_mean(
                sub_pc, inverse, dim=0
            )

            batch_size = 1
            batch_indices = torch.zeros(
                (voxel_coords.shape[0], 1),
                dtype=torch.int32,
                device=sub_pc.device
            ) + b

            indices = torch.cat([
                batch_indices,
                voxel_coords[:, [2, 1, 0]]  # z, y, x
            ], dim=1)

            batch_indices_list.append(indices)
            batch_features_list.append(voxel_features)

            b += 1

        batch_features = torch.cat(batch_features_list, dim=0)
        batch_indices = torch.cat(batch_indices_list, dim=0)

        cropped_spheres = spconv.SparseConvTensor(
            features=batch_features.float(),  # (M, C=3)
            indices=batch_indices,  # (M, 4)
            spatial_shape=[grid_size] * 3,
            batch_size=self.batch_size
        )

        if view:
            x = cropped_spheres.dense()
            x = (x != 0).any(dim=1, keepdim=False).float().cpu().numpy()[0]
            view_3d_occupancy_grid(x)

        return cropped_spheres

    def step(self, i, report=False):

        self.sim_env.max_obj_per_scene = 10


        if (len(self.DDM.low_quality_samples_tracker)==0 or self.skipped_last) and (
                ((np.random.rand() < self.skip_rate.val ** 2) or len(self.DDM) >= self.max_scenes) and len(
                self.DDM) > 100):

            self.loaded_synthesised_data = self.DDM.load_random_sample()
            self.sim_env.objects = deque(self.loaded_synthesised_data.obj_ids)
            self.sim_env.objects_poses = self.loaded_synthesised_data.obj_poses

            # assert 7*len(self.sim_env.objects)==len(self.sim_env.objects_poses), f'{len(self.sim_env.objects)}, {len(self.sim_env.objects_poses)}'

            # print(Fore.LIGHTMAGENTA_EX,'----------Load saved data point .....................................objects:',self.sim_env.objects,Fore.RESET)
            self.sim_env.reload()
            # print(self.sim_env.objects_poses)


        else:
            self.loaded_synthesised_data = None

            self.sim_env.remove_objects(n=self.sim_env.max_obj_per_scene)

            # self.sim_env.drop_new_obj(selected_index=None, stablize=True)
            # if len(self.DDM) > 100:
            nn = random.randint(5, self.sim_env.max_obj_per_scene )
            # for k in range(nn):
            self.sim_env.drop_new_obj(selected_index=None, stablize=True,n=nn)


        # self.sim_env.print_state()

        '''get scene perception'''
        depth, pc, floor_mask = self.sim_env.get_scene_preception(view=False)

        # obj_normals=estimate_suction_direction(pc[~floor_mask],view=False,radius=0.01, self.max_nn=10)
        # approach=np.zeros((600*600,3))
        # approach[:,2]-=1
        # approach[~floor_mask]=-obj_normals
        # approach=approach.transpose().reshape(3,600,600)
        # approach=torch.from_numpy(approach).cuda()

        # view_npy_open3d(pc)
        full_objects_pc = self.sim_env.get_obj_point_clouds(view=False)
        full_pointcloud = np.vstack([pc[floor_mask], full_objects_pc])

        floor_mask = torch.from_numpy(floor_mask).cuda()

        # full_pointcloud=None
        # view_npy_open3d(full_pointcloud)
        full_pointcloud = torch.from_numpy(full_pointcloud).cuda()

        clean_depth = torch.from_numpy(depth).cuda()  # [600.600]
        depth = torch.from_numpy(depth).cuda()  # [600.600]

        # pc = torch.from_numpy(pc).cuda()

        # view_npy_open3d(pc)
        # depth=clean_depth
        # depth=add_reflective_blob_noise(clean_depth,n_blobs=np.random.randint(5,10), blob_radius=np.random.uniform(1, 3), outlier_scale=0.02)
        # view_image(clean_depth.cpu().numpy())
        # view_image(depth.cpu().numpy())

        # depth=add_depth_noise(depth,keep_mask=floor_mask.reshape(600,600))
        # pc, _ = self.sim_env.depth_to_pointcloud(depth.cpu().numpy(), self.sim_env.intr, self.sim_env.extr)
        pc = torch.from_numpy(pc).cuda()
        # view_npy_open3d(pc.cpu().numpy())

        # view_npy_open3d(pc.cpu().numpy())
        # return
        # torch.save(depth, 'depth_ch_tmp')
        # floor_mask = torch.from_numpy(floor_mask).cuda()
        # torch.save(floor_mask, 'floor_mask_ch_tmp')
        # exit()

        latent_vector = torch.randn((1, 8, depth.shape[0], depth.shape[1]), device='cuda')

        for k in range(self.iter_per_scene):

            with torch.no_grad():
                self.gan.generator.eval()
                gripper_pose, grasp_quality_logits, grasp_collision_logits, features, _ = self.gan.generator(
                    depth[None, None, ...], ~floor_mask.view(1, 1, 600, 600), latent_vector, detach_backbone=True)
                self.gan.generator.train()

                grasp_quality = logits_to_probs(grasp_quality_logits)
                grasp_collision = logits_to_probs(grasp_collision_logits)

                n = max(self.tou, self.skip_rate.val)  # ** 2
                f = (1 - grasp_quality.detach()).clamp(min=self.skip_rate.val ** 2) #** 2

                annealing_factor = f  # torch.clamp(f,min=n) #if self.skip_rate.val < 0.7 else n + (1 - n) * f
                print(Fore.LIGHTYELLOW_EX,
                      f'mean_annealing_factor= {annealing_factor.mean()},max_annealing_factor= {annealing_factor.max()},min_annealing_factor= {annealing_factor.min()}, tou={self.tou}, skip rate={self.skip_rate.val}',
                      Fore.RESET)

                self.max_G_norm = max(self.G_grad_norm_MR.val * self.tou ** 2 + (1 - self.tou ** 2) * 5, 5)
                self.max_D_norm = max(self.D_grad_norm_MR.val * self.tou ** 2 + (1 - self.tou ** 2) * 1, 1)

                # if self.tou<0.6:
                #     self.batch_size=1
                #     self.learning_rate=1e-5
                # elif self.tou>0.7:
                #     self.batch_size = 1
                #     self.learning_rate=1e-4

                gripper_pose_ref =  self.pose_interpolation(gripper_pose,
                                                         annealing_factor=annealing_factor)  # [b,self.n_param,600,600]
                if self.loaded_synthesised_data is not None:
                    '''inject saved poses'''
                    grasp_quality_p = grasp_quality[0, 0].reshape(-1)

                    gripper_pose_ref = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000, self.n_param)

                    # gen_gripper_pose = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000, self.n_param)
                    for t in range(len(self.loaded_synthesised_data.target_indexes)):
                        index = self.loaded_synthesised_data.target_indexes[t]
                        pose = self.loaded_synthesised_data.grasp_parameters[t]
                        # gen_pose=gen_gripper_pose[index]
                        # saved_target_point=self.loaded_synthesised_data.grasp_target_points[t]
                        # target_point = pc[index]

                        # print(f'saved_target_point: {saved_target_point}, target_point: {target_point}')
                        # print(f'saved pose: {pose}')

                        pose = torch.tensor(pose).cuda()

                        # if grasp_quality_p[index].item()<0.6:
                        # if self.loaded_synthesised_data.importance[t]>grasp_quality_p[index].item():
                        gripper_pose_ref[index] = pose

                        # k_dist = (1 - F.cosine_similarity(pose[0:5], gen_pose[0:5], dim=-1)) / 2

                        # self.loaded_synthesised_data.importance[t]=grasp_quality_p[index].item()

                    # gripper_pose_ref=gripper_pose_ref.permute(1,2,0)
                    # print(f'saved_pose : {gripper_pose_ref[0,:,t]} , pose : {pose}')

                    gripper_pose_ref = gripper_pose_ref.reshape(600, 600, self.n_param).permute(2, 0, 1).unsqueeze(0)

                model_b_quality_logits = self.gan.generator.get_grasp_quality(
                    depth[None, None, ...], ~floor_mask.view(1, 1, 600, 600), gripper_pose_ref)
                model_b_quality = logits_to_probs(model_b_quality_logits)

                # gripper_pose_ref[0,0:3]=approach
                # if i % int(50) == 0 and i != 0 and k == 0:
                #     try:
                #         self.export_check_points()
                #         self.save_statistics()
                #     except Exception as e:
                #         print(Fore.RED, str(e), Fore.RESET)
                if report and k == 0:
                    self.view_result(gripper_pose, floor_mask)

                d_pairs, g_pairs = [], []
                if not self.train_policy_only:
                    self.tmp_pose_record = []

                    d_pairs, g_pairs, sampler_samples = self.sample_contrastive_pairs(pc, floor_mask, gripper_pose,
                                                                                      gripper_pose_ref,
                                                                                      self.tou, grasp_quality.detach(),
                                                                                      grasp_collision.detach(),
                                                                                      self.superior_A_model_moving_rate,
                                                                                      latent_vector, model_b_quality)
                    # print(f'len d {len(d_pairs)}, len g {len(g_pairs)}')

                    if self.synthesizie_only: break

            if self.test_mode:
                if len(d_pairs) > 0 and self.view:
                    self.prepare_voxels(d_pairs, depth, pc, full_pointcloud, view=self.view)
                return

            self.tou = 1 - self.superior_A_model_moving_rate.val

            gripper_pose = gripper_pose[0].permute(1, 2, 0).reshape(360000, self.n_param)
            gripper_pose_ref_pixel = None if self.train_policy_only else gripper_pose_ref
            gripper_pose_ref = None if self.train_policy_only else gripper_pose_ref[0].permute(1, 2, 0).reshape(360000,
                                                                                                           self.n_param)

            if not self.train_policy_only and len(d_pairs) == self.batch_size:
                print()
                print(
                    '------------------------------------------------step_Critic--------------------------------------------------------')
                d_cropped_spheres = self.prepare_voxels(d_pairs, depth, pc, full_pointcloud)
                # d_cropped_spheres=None
                self.step_discriminator(d_cropped_spheres, depth, clean_depth, gripper_pose, gripper_pose_ref, d_pairs,
                                        floor_mask, grasp_quality, latent_vector=latent_vector)
                self.print_pairs_info(d_pairs, gripper_pose, gripper_pose_ref)

                print()
                self.skipped_last=False
            else:
                self.skipped_last=True

            # if sampler_samples==batch_size:
            if not self.train_policy_only and len(g_pairs) == self.batch_size:
                print()
                print(
                    '------------------------------------------------step_Policy_and_action--------------------------------------------------------')
                g_cropped_spheres = self.prepare_voxels(g_pairs, depth, pc, full_pointcloud)
                # g_cropped_spheres=None
                self.step_generator(g_cropped_spheres, depth, clean_depth, floor_mask, pc, gripper_pose_ref_pixel,
                                    g_pairs, latent_vector)
                self.print_pairs_info(g_pairs, gripper_pose, gripper_pose_ref)
                print()
            # elif self.skip_rate.val>0.9:
            elif self.skip_rate.val < 0.5 or self.train_policy_only:
                print()
                print(
                    '------------------------------------------------step_Policy--------------------------------------------------------')
                self.step_generator(None, depth, clean_depth, floor_mask, pc, gripper_pose_ref_pixel, g_pairs,
                                    latent_vector)
                print()

            if not self.train_policy_only and not (
                    (len(d_pairs) == self.batch_size) or (len(g_pairs) == self.batch_size)) and not self.test_mode:
                # self.superior_A_model_moving_rate.update(0)
                self.tou = 1 - self.superior_A_model_moving_rate.val
                if k == 0:
                    self.sim_env.remove_objects(n=2)
                    break
            # else:
            # self.sim_env.update_obj_info(0.9)

            # else:
            #     self.step_generator_without_sampler(depth, floor_mask, pc,latent_vector)
            # continue


    def view_result(self, gripper_poses, floor_mask):
        with torch.no_grad():
            self.sim_env.save_obj_dict()


            cuda_memory_report()

            values = gripper_poses[0].permute(1, 2, 0).reshape(360000, self.n_param).detach()  # .cpu().numpy()
            values = values[~floor_mask]

            self.gripper_sampler_statistics.print()
            self.critic_statistics.print()

            # values = gripper_pose.permute(1, 0, 2, 3).flatten(1).detach()
            try:
                print(f'gripper_pose sample = {values[np.random.randint(0, values.shape[0])].cpu()}')
            except Exception as e:
                print('result view error', str(e))
            print(f'gripper_pose std = {torch.std(values, dim=0).cpu()}')
            print(f'gripper_pose mean = {torch.mean(values, dim=0).cpu()}')
            print(f'gripper_pose max = {torch.max(values, dim=0)[0].cpu()}')
            print(f'gripper_pose min = {torch.min(values, dim=0)[0].cpu()}')

            self.moving_collision_rate.view()
            self.skip_rate.view()
            self.superior_A_model_moving_rate.view()

            self.Ave_samples_per_scene.view()
            self.Ave_importance.view()
            self.Ave_uniquness.view()

            self.G_grad_norm_MR.view()
            self.D_grad_norm_MR.view()

            self.bin_collision_statistics.print()
            self.objects_collision_statistics.print()
            self.collision_statistics.print()

            self.grasp_quality_statistics.print()
            # self.vectors_clusters.view()
            self.approach_beta_clusters.view()


    def save_statistics(self):
        self.moving_collision_rate.save()
        self.skip_rate.save()
        self.superior_A_model_moving_rate.save()

        self.Ave_samples_per_scene.save()
        self.Ave_importance.save()

        self.G_grad_norm_MR.save()
        self.D_grad_norm_MR.save()

        self.critic_statistics.save()
        self.gripper_sampler_statistics.save()
        self.approach_beta_clusters.save()


        self.Ave_uniquness.save()

        self.data_tracker.save()

        self.objects_collision_statistics.save()

        self.bin_collision_statistics.save()
        self.collision_statistics.save()
        self.grasp_quality_statistics.save()

        self.sim_env.save_obj_dict()

    def export_check_points(self):
        self.gan.export_models()
        self.gan.export_optimizers()

    def clear(self):
        self.critic_statistics.clear()

        self.bin_collision_statistics.clear()
        self.collision_statistics.clear()
        self.gripper_sampler_statistics.clear()
        self.grasp_quality_statistics.clear()

    def begin(self, iterations=10):
        pi = progress_indicator('Begin new training round: ', max_limit=iterations)

        for i in range(iterations):
            if self.skip_rate.val > 0.8:
                self.batch_size = 1
                self.iter_per_scene = 1  # 5
                self.sim_env.max_obj_per_scene = 1
            elif self.skip_rate.val < 0.4:
                self.batch_size = 2
                self.iter_per_scene = 1
                self.sim_env.max_obj_per_scene = int(7 * np.random.rand())
            # cuda_memory_report()
            # self.batch_size = 1

            if self.args.catch_exceptions:
                try:
                    self.step(i, report=i == iterations - 1)
                    pi.step(i)
                except Exception as e:
                    print(Fore.RED, str(e), Fore.RESET)
                    traceback.print_exc()
                    torch.cuda.empty_cache()
                    # self.sim_env.update_obj_info(0.1)
                    self.sim_env.remove_objects(n=self.sim_env.max_obj_per_scene)
                    if self.loaded_synthesised_data is not None: self.DDM.low_quality_samples_tracker.append(self.loaded_synthesised_data.id)

            else:
                self.step(i, report=i == iterations - 1)
                pi.step(i)

        pi.end()
        self.export_check_points()
        self.save_statistics()

        self.clear()
