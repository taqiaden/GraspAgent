import copy
import os
import torch.nn.functional as F
import numpy as np
from colorama import Fore
from filelock import FileLock
from torch import nn
import torch.nn.functional as F
from Configurations.config import workers
from Online_data_audit.data_tracker import  gripper_grasp_tracker, DataTracker
from analytical_suction_sampler import estimate_suction_direction
from check_points.check_point_conventions import GANWrapper, ModelWrapper
from dataloaders.Grasp_GAN_dl import GraspGANDataset2
from lib.IO_utils import custom_print, load_pickle, save_pickle
from lib.Multible_planes_detection.plane_detecttion import bin_planes_detection, cache_dir
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data2
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from lib.loss.balanced_bce_loss import BalancedBCELoss
from lib.models_utils import reshape_for_layer_norm
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from models.Grasp_GAN import N_gripper_sampling_module_key, G, D
from models.action_net import ActionNet, action_module_key
# from models.scope_net import scope_net_vanilla, gripper_scope_module_key
from records.training_satatistics import TrainingTracker, MovingRate
from registration import camera
from training.N_Grasp_GAN_training import collision_loss, firmness_loss
from training.learning_objectives.gripper_collision import  evaluate_grasps3
from training.suction_scope_training import weight_decay
from visualiztion import view_features, plt_features, dense_grasps_visualization, view_npy_open3d

detach_backbone = False
lock = FileLock("file.lock")
max_samples_per_image=30

max_n = 50
min_batch_size = 2
max_batch_size = 4

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

def get_normal_direction(pc, file_index):
    file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
    if os.path.exists(file_path):
        normals = load_pickle(file_path)
        if  normals.shape[0]!=pc.shape[0]:
            normals = estimate_suction_direction(pc,
                                                 view=False)  # inference time on local computer = 1.3 s        if file_index is not None:
            file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
            save_pickle(file_path, normals)
    else:
        normals = estimate_suction_direction(pc,
                                            view=False)  # inference time on local computer = 1.3 s        if file_index is not None:
        file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
        save_pickle(file_path, normals)

    normals = torch.from_numpy(normals).to('cuda')
    return normals


class TrainGraspGAN:
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
        self.gripper_sampler_statistics = None
        self.critic_statistics = None
        self.data_tracker = None


        self.sampling_centroid = None
        self.diversity_momentum = 1.0

        self.sample_from_latent=True

        self.action_net = None
        self.load_action_model()

    def load_action_model(self):
        try:
            '''load  models'''
            actions_net = ModelWrapper(model=ActionNet(), module_key=action_module_key)
            actions_net.ini_model(train=False)
            self.action_net = actions_net.model
        except Exception as e:
            print(str(e))

    def initialize_ref_generator(self):
        model_wrapper = ModelWrapper(model=copy.deepcopy(self.gan.generator), module_key='ref_generator')
        model_wrapper.optimizer = torch.optim.Adam(model_wrapper.model.parameters(), lr=self.learning_rate/10,
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

    def step_ref_generator_training(self, depth, mask, gripper_pose, objects_mask):

        assert objects_mask.sum() > 0
        approach_direction = gripper_pose[:, 0:3, ...].detach().clone()

        self.ref_generator.model.zero_grad(set_to_none=True)

        gripper_pose_ref = self.ref_generator.model(
            depth.clone(), approach_direction, detach_backbone=True)

        gripper_pose_ref2 = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :][mask]

        assert not torch.isnan(gripper_pose_ref2).any(),f'{gripper_pose_ref2},---{gripper_pose}'
        assert not torch.isnan(gripper_pose).any(),f'{gripper_pose}'

        # gripper_pose2 = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask].detach().clone()

        # models_separation = torch.abs(
        #     gripper_pose_ref2[:, 3:][objects_mask] - gripper_pose2[:, 3:][objects_mask]).mean()
        # print(f'models_separation={models_separation}')

        # if (self.sample_from_latent and self.moving_out_of_scope.val>0.01):

        p=max((1-self.moving_anneling_factor.val)**2,self.moving_collision_rate.val,self.moving_out_of_scope.val)
        if p>np.random.rand() or self.sample_from_latent :
            beta_dist_width=self.generate_random_beta_dist_widh(gripper_pose[:, 0, ...].numel())
            beta_dist_width = reshape_for_layer_norm(beta_dist_width, camera=camera, reverse=True)
            sampled_pose=torch.cat([approach_direction,beta_dist_width],dim=1)
            print(Fore.CYAN,f'Random label, p={p}',Fore.RESET)

            # sampled_pose[:,3:,...]=sampled_pose[:,3:,...]*p+(1-p)*gripper_pose[:,3:,...].detach().clone()
            # sampled_pose[:, 3:5, ...]=F.normalize(sampled_pose[:, 3:5, ...], dim=1)
            # sampled_pose[:, 5:, ...]=torch.clamp(sampled_pose[:, 5:, ...],0.,1)
            return True, sampled_pose,0

        print(Fore.CYAN,f'Second G label, p={p}',Fore.RESET)

        return False, gripper_pose_ref,0



    def initialize(self, n_samples=None):
        self.n_samples = n_samples
        self.prepare_data_loader()

        '''Moving rates'''
        self.moving_collision_rate = MovingRate(N_gripper_sampling_module_key + '_collision', decay_rate=0.01, initial_val=1.)
        self.moving_firmness = MovingRate(N_gripper_sampling_module_key + '_firmness', decay_rate=0.01, initial_val=0.)
        self.moving_out_of_scope = MovingRate(N_gripper_sampling_module_key + '_out_of_scope', decay_rate=0.01, initial_val=1.)
        self.relative_sampling_timing = MovingRate(N_gripper_sampling_module_key + '_relative_sampling_timing', decay_rate=0.01,
                                                   initial_val=1.)
        self.superior_A_model_moving_rate=MovingRate(N_gripper_sampling_module_key + '_superior_A_model', decay_rate=0.01,
                                                   initial_val=0.)
        self.moving_anneling_factor = MovingRate(N_gripper_sampling_module_key + '_anneling_factor', decay_rate=0.01,
                                                 initial_val=0.)


        '''initialize statistics records'''


        self.gripper_sampler_statistics = TrainingTracker(name=N_gripper_sampling_module_key + '_gripper_sampler',
                                                          iterations_per_epoch=len(self.data_loader),
                                                          track_label_balance=False)

        self.critic_statistics = TrainingTracker(name=N_gripper_sampling_module_key + '_critic',
                                                 iterations_per_epoch=len(self.data_loader), track_label_balance=False)

        self.data_tracker = DataTracker(name=gripper_grasp_tracker)

        # gripper_scope = ModelWrapper(model=scope_net_vanilla(in_size=6), module_key=gripper_scope_module_key)
        # gripper_scope.ini_model(train=False)
        # self.gripper_arm_reachability_net = gripper_scope.model

    def prepare_data_loader(self):
        file_ids = training_buffer.get_indexes()
        # file_ids = sample_positive_buffer(size=self.n_samples, dict_name=gripper_grasp_tracker,
        #                                   disregard_collision_samples=True,sample_with_probability=False)
        print(Fore.CYAN, f'Buffer size = {len(file_ids)}', Fore.RESET)
        dataset = GraspGANDataset2(data_pool=training_buffer, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers,
                                                  shuffle=True)
        self.size = len(dataset)
        self.data_loader = data_loader

    def prepare_model_wrapper(self):
        '''load  models'''
        gan = GANWrapper(N_gripper_sampling_module_key, G, D)
        gan.ini_models(train=False)
        # gan.critic_sgd_optimizer(learning_rate=self.learning_rate ,weight_decay_=0.0)
        gan.generator_sgd_optimizer(learning_rate=self.learning_rate*10,weight_decay_=0.0)
        gan.critic_adam_optimizer(learning_rate=self.learning_rate,beta1=0.9,weight_decay_=0.0)
        # gan.generator_adam_optimizer(learning_rate=self.learning_rate, beta1=0.9)

        return gan

    def simulate_elevation_variations(self, original_depth,objects_mask, max_elevation=0.15, exponent=2.0):
        '''Elevation-based Augmentation'''
        shift_entities_mask = objects_mask & (original_depth > 0.0001)
        new_depth = original_depth.clone().detach()
        new_depth[shift_entities_mask] -= max_elevation * (np.random.rand() ** exponent) * camera.scale

        return new_depth

    def analytical_bin_mask(self, pc, file_ids):
        try:
            bin_mask,floor_elevation = bin_planes_detection(pc, sides_threshold=0.005, floor_threshold=0.0015, view=False,
                                            file_index=file_ids[0], cache_name='bin_planes2')
            return bin_mask, floor_elevation
        except Exception as error_message:
            print(file_ids[0])
            print(error_message)
            return None, None


    def step_discriminator_learning(self, depth, pc, mask, bin_mask, gripper_pose, gripper_pose_ref,
                                    is_noise_label,altered_objects_elevation,gamma_col,valid_pose_index,floor_elevation):
        '''self supervised critic learning'''
        with torch.no_grad():
            generated_grasps_cat = torch.cat([gripper_pose, gripper_pose_ref], dim=0)
            depth_cat = depth.repeat(2, 1, 1, 1)
        critic_score = self.gan.critic(depth_cat, generated_grasps_cat)

        # counter = 0
        tracked_indexes = []

        gripper_pose2 = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask]
        ref_gripper_pose2 = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :][mask]
        gen_scores_ = critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        ref_scores_ = critic_score.permute(0, 2, 3, 1)[1, :, :, 0][mask]


        if self.sampling_centroid is None:
            selection_p = torch.rand_like(ref_gripper_pose2[:,0])
        else:
            gamma_dive = (1.001 - F.cosine_similarity(ref_gripper_pose2.detach().clone(),
                                               self.sampling_centroid[None, :], dim=-1))/2
            gamma_dive *= ((1.001 - F.cosine_similarity(gripper_pose2.detach().clone(),
                                               self.sampling_centroid[None, :], dim=-1))/2)#**2.0
            # gamma_small_change = 1.006 - ((1.0 - F.cosine_similarity(gripper_pose2[:,3:5].detach().clone(),
            #                                                         ref_gripper_pose2[:,3:5].detach().clone(), dim=-1)) / 2)
            # gamma_firmness=torch.clamp(ref_gripper_pose2[:,-2].detach().clone(),0.001,0.99)**0.3
            gamma_rand=torch.rand_like(gamma_dive)

            # selection_p=(gamma_dive*gamma_pairwise_dist*gamma_rand*gamma_firmness*gamma_col)**(1/6)

            if not altered_objects_elevation:
                d_gamma=1-torch.from_numpy(pc[:,-1]-pc[bin_mask <= 0.5][:,-1].min()).cuda()
                d_gamma=d_gamma/d_gamma[bin_mask <= 0.5].max()
                selection_p = (gamma_dive *   gamma_rand * d_gamma) ** (1 / 5)
            else:
                selection_p = (gamma_dive *   gamma_rand *) ** (1 / 4)

            assert not torch.isnan(selection_p).any(), f'{ref_gripper_pose2}'

            # if not altered_objects_elevation:
            #     d_gamma=1-torch.from_numpy(pc[:,-1]-pc[bin_mask <= 0.5][:,-1].min()).cuda()
            #     d_gamma=d_gamma/d_gamma[bin_mask <= 0.5].max()
            #     selection_p*=(d_gamma)

        selection_mask = torch.from_numpy(bin_mask <= 0.5).cuda()
        n = int(min(max_n, selection_mask.sum()))

        loss_terms_counters = [0, 0, 0, 0, 0]
        l_collision = torch.tensor(0., device=gripper_pose.device)
        l_firmness = torch.tensor(0., device=gripper_pose.device)
        col_loss_counter=0
        firm_loss_counter=0
        for t in range(n):
            dist = MaskedCategorical(probs=selection_p, mask=selection_mask)
            if t==0 and valid_pose_index is not None:
                target_index=valid_pose_index
            else:
                target_index = dist.sample()
            selection_mask[target_index] = False
            target_point = pc[target_index]

            target_generated_pose = gripper_pose2[target_index]
            target_ref_pose = ref_gripper_pose2[target_index]
            label_ = ref_scores_[target_index]
            prediction_ = gen_scores_[target_index]
            c_, s_, f_ = evaluate_grasps3(target_point, target_generated_pose, target_ref_pose, pc, visualize=True,floor_elevation=floor_elevation)

            if (s_[0] == 0) and (c_[0] == 0 or (s_[1] == 0 and c_[1] == 0) ):
                self.moving_collision_rate.update(int(c_[0] > 0))
            if sum(s_) == 0 and sum(c_) == 0:
                self.moving_firmness.update(int(f_[0] > f_[1]))
            self.moving_out_of_scope.update(int(s_[0] > 0))

            prediction_uniqness=  ((1 - F.cosine_similarity(target_generated_pose.detach().clone()[None, :],
                                                                            self.sampling_centroid[None, :], dim=-1)) / 2).item()**0.5    if      self.sampling_centroid is not None else 0.

            ref_width=target_ref_pose[-1]
            ref_dist=target_ref_pose[-2]
            if (s_[0] > 0. or c_[0] > 0.) and s_[1] == 0 and c_[1] == 0:
                if not is_noise_label: self.superior_A_model_moving_rate.update(0.)
            elif (s_[1] > 0. or c_[1] > 0.) and s_[0] == 0 and c_[0] == 0:
                if not is_noise_label: self.superior_A_model_moving_rate.update(1.0)
            counted=False
            if col_loss_counter<max_batch_size:
                l_c, counted, loss_terms_counters,_ = collision_loss(c_, s_, f_, prediction_,
                                                                   label_, loss_terms_counters,prediction_uniqness,ref_dist,ref_width)
            if counted:
                col_loss_counter+=1

            if not counted and firm_loss_counter<max_batch_size:
                '''firmness loss'''
                l_f, counted, loss_terms_counters,_ = firmness_loss(c_, s_, f_, prediction_,
                                                                  label_, loss_terms_counters,prediction_uniqness)
                if counted:
                    # print(target_ref_pose[3:].detach(), '--', target_generated_pose[3:].detach(), 'c--', c_, 's--', s_,
                    #       'f--', f_)
                    firm_loss_counter+=1

            if counted:
                if t==0 and valid_pose_index is not None:
                    print(Fore.GREEN)
                    print(target_ref_pose[3:].detach().cpu(), '--', target_generated_pose[3:].detach().cpu(), 'c--', c_, 's--', s_,
                          'f--', f_,Fore.RESET)
                else:
                    print(target_ref_pose[3:].detach().cpu(), '--', target_generated_pose[3:].detach().cpu(), 'c--', c_, 's--', s_,
                          'f--', f_)

                avoid_collision = (s_[0] > 0. or c_[0] > 0.)
                A_is_collision_free = None
                A_is_more_firm = None

                if (s_[0] > 0. or c_[0] > 0.) and s_[1] == 0 and c_[1] == 0:
                    A_is_collision_free = False
                    # if not is_noise_label: self.superior_A_model_moving_rate.update(0.)
                elif (s_[1] > 0. or c_[1] > 0.) and s_[0] == 0 and c_[0] == 0:
                    A_is_collision_free = True
                    # if not is_noise_label: self.superior_A_model_moving_rate.update(1.0)
                elif sum(c_) == 0 and sum(s_) == 0:
                    if f_[1] > f_[0]:
                        A_is_more_firm = False
                    elif f_[0] > f_[1]:
                        A_is_more_firm = True
                tracked_indexes.append((target_index, avoid_collision, A_is_collision_free, A_is_more_firm))

                if self.sampling_centroid is None:
                    self.sampling_centroid = target_generated_pose.detach().clone()
                else:
                    diffrence = ((1 - F.cosine_similarity(target_generated_pose.detach().clone()[None, :],
                                                          self.sampling_centroid[None, :], dim=-1)) / 2) ** 2.0
                    self.diversity_momentum = self.diversity_momentum * 0.99 + diffrence.item() * 0.01
                    self.sampling_centroid = self.sampling_centroid * 0.99 + target_generated_pose.detach().clone() * 0.01

            if col_loss_counter+firm_loss_counter==2* max_batch_size :
                self.relative_sampling_timing.update((t + 1) / n)
                break

        # curc_loss=torch.clamp(torch.abs(ref_scores_-gen_scores_)-1.,0.).mean()
        # if     curc_loss.item()  >0.: print(f'curclium loss = {curc_loss.item()}')
        if col_loss_counter>=min_batch_size :
            print(loss_terms_counters)
            loss = l_collision / (loss_terms_counters[0] + loss_terms_counters[1]+loss_terms_counters[2])
            if firm_loss_counter>=min_batch_size :
                loss += l_firmness / (loss_terms_counters[3]+loss_terms_counters[4])

            self.critic_statistics.loss = loss.item()

            return True, tracked_indexes
        else:
            print('pass, counter/max_Batch_size=', loss_terms_counters, '/', max_batch_size)
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




    def pixel_to_point_index(self,depth,mask,gripper_pixel_index):
        tmp = torch.zeros_like(depth)[0, 0]
        tmp[gripper_pixel_index[0], gripper_pixel_index[1]] = 1.0
        tmp = tmp[mask]

        point_index = (tmp == 1).nonzero(as_tuple=False)
        return point_index.item()

    def begin(self):

        pi = progress_indicator('Begin new training round: ', max_limit=len(self.data_loader))
        gripper_pose = None
        for i, batch in enumerate(self.data_loader, 0):
            depth, file_ids ,pose_7,valid_gripper_pose,gripper_pixel_index= batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pose_7 = pose_7.cuda().float()[0]
            gripper_pixel_index=gripper_pixel_index[0]
            valid_gripper_pose=valid_gripper_pose[0]
            # pose_7[-2]=torch.clamp(pose_7[-2]-0.1,0.)
            valid_pose_index=None
            pi.step(i)
            print(' ')
            print(f'i={i}')
            print('')



            pc, mask = depth_to_point_clouds(depth[0, 0].cpu().numpy(), camera)
            pc = transform_to_camera_frame(pc, reverse=True)
            if valid_gripper_pose:
                valid_pose_index=self.pixel_to_point_index(depth,mask,gripper_pixel_index)

            '''background detection head'''
            bin_mask,floor_elevation = self.analytical_bin_mask(pc, file_ids)
            if bin_mask is None: continue
            objects_mask_numpy = bin_mask <= 0.5
            objects_mask = torch.from_numpy(objects_mask_numpy).cuda()
            objects_mask_pixel_form=torch.ones_like(depth)
            objects_mask_pixel_form[0,0][mask]=objects_mask_pixel_form[0,0][mask]*objects_mask
            objects_mask_pixel_form=objects_mask_pixel_form>0.5

            # view_features(reshape_for_layer_norm(objects_mask_pixel_form, camera=camera, reverse=False))

            '''Elevation-based augmentation'''
            if np.random.rand()>0.7:
                depth=self.simulate_elevation_variations(depth,objects_mask_pixel_form,exponent=5.0)
                pc, mask = depth_to_point_clouds(depth[0, 0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)
                altered_objects_elevation=True
            else:
                altered_objects_elevation=False

            '''normal approach'''
            approach_pointwise_form=get_normal_direction(pc, file_ids[0]).float()
            random_mask=torch.rand_like(approach_pointwise_form[:,0])>0.5
            approach_pointwise_form[random_mask]*=0.0
            approach_pointwise_form[random_mask,2]=1.0
            approach=torch.zeros_like(depth)
            approach=approach.repeat(1,3,1,1)
            approach=approach.permute(0,2,3,1)

            approach[0][mask] = approach[0][mask] +approach_pointwise_form
            approach=approach.permute(0,3,1,2)

            for k in range(max_samples_per_image):
                with torch.no_grad():
                    gripper_pose = self.gan.generator(
                        depth.clone(),approach,  detach_backbone=True) # [1,7,h,w]

                if k == 0:
                    collision_scores = self.action_net.collision_prediction(depth.clone(), gripper_pose)
                    gamma_col = 1.0 - collision_scores[0, 0][mask]
                    # print(f'i={i}, k={k}, pc shape={pc.shape}')
                    # dense_grasps_visualization(pc, gripper_pose[0].permute(1, 2, 0)[mask],
                    #                            view_mask=objects_mask,
                    #                            sampling_p=gamma_col**2, view_all=False)
                    # break

                is_noise_label, gripper_pose_ref,reg_loss = self.step_ref_generator_training(depth, mask, gripper_pose,
                                                                                             objects_mask  )

                if valid_gripper_pose:
                    pixels_A=gripper_pixel_index[0]
                    pixels_B=gripper_pixel_index[1]
                    gripper_pose_ref[0,:,pixels_A,pixels_B]=pose_7
                    print(Fore.GREEN,'inject label pose',Fore.RESET)

                if i % 10 == 0 and i != 0 and k==0:
                    gripper_poses = gripper_pose[0].permute(1, 2, 0)[mask].detach()  # .cpu().numpy()
                    self.view_result(gripper_poses)
                    self.export_check_points()
                    self.save_statistics()
                    self.load_action_model()
                if i % 5 == 0 and k==0:
                    gripper_pose2 = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask].detach()
                    beta_angles = torch.atan2(gripper_pose2[:, 3], gripper_pose2[:, 4])
                    print(f'beta angles variance = {beta_angles.std()}')
                    subindexes = torch.randperm(gripper_pose2.size(0))
                    if subindexes.shape[0] > 1000: subindexes = subindexes[:1000]
                    all_values = gripper_pose2[subindexes].detach()
                    dist_values=torch.clamp((all_values[:,5:6]),0.,1.).clone()
                    width_values=torch.clamp((all_values[:,6:7]),0.,1.).clone()
                    beta_values=(all_values[:,3:5]).clone()
                    beta_diversity=(1.001 - F.cosine_similarity(beta_values[None,...],beta_values[:,None,:],dim=-1)).mean()/2
                    print(f'Mean separation for beta : {beta_diversity}')
                    print(f'Mean separation for width : {torch.cdist(width_values, width_values, p=1).mean()}')
                    print(f'Mean separation for distance : {torch.cdist(dist_values, dist_values, p=1).mean()}')
                    self.moving_anneling_factor.update(beta_diversity)

                '''zero grad'''
                self.gan.critic.zero_grad(set_to_none=True)
                self.gan.generator.zero_grad(set_to_none=True)

                train_generator, tracked_indexes = self.step_discriminator_learning(depth, pc, mask, bin_mask, gripper_pose,
                                                                                    gripper_pose_ref.detach().clone(),
                                                                                    is_noise_label,altered_objects_elevation,gamma_col,valid_pose_index,floor_elevation)




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


        self.critic_statistics.save()
        self.gripper_sampler_statistics.save()

        self.data_tracker.save()

    def export_check_points(self):
        self.gan.export_models()
        self.gan.export_optimizers()

    def clear(self):

        self.gripper_sampler_statistics.clear()
        self.critic_statistics.clear()
def train_N_grasp_GAN(n=1):
    lr = 1e-4
    Train_grasp_GAN = TrainGraspGAN(n_samples=None, learning_rate=lr)
    torch.cuda.empty_cache()
    # torch.autograd.set_detect_anomaly(True)

    for i in range(n):
        with torch.no_grad():
            cuda_memory_report()
            Train_grasp_GAN.initialize(n_samples=None)
            Train_grasp_GAN.begin()

    # del Train_grasp_GAN

if __name__ == "__main__":
    train_N_grasp_GAN(n=10000)
