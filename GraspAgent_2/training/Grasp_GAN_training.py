import numpy as np
import spconv.pytorch as spconv
from colorama import Fore
from filelock import FileLock
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_scatter import scatter_mean

from Configurations.config import workers
from GraspAgent_2.data_loader.YPG_data_laoder import YPGDataset2
from GraspAgent_2.model.YPG_GAN import YPG_model_key, YPG_G, YPG_D
from GraspAgent_2.utils.Voxel_operations import occupancy_to_pointcloud, crop_cube
from Online_data_audit.data_tracker import gripper_grasp_tracker, DataTracker
from check_points.check_point_conventions import GANWrapper
from lib.IO_utils import custom_print
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data2
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds, transform_to_camera_frame_torch
from lib.image_utils import view_image
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from records.training_satatistics import TrainingTracker, MovingRate
from registration import camera
from training.Actions_learning import suction_sampler_loss
from training.learning_objectives.gripper_collision import evaluate_grasps3, gripper_bin_collision_loss, \
    gripper_object_collision_loss, gripper_quality_loss
from visualiztion import dense_grasps_visualization, view_npy_open3d
from GraspAgent_2.utils.backgorund_detection import analytical_bin_mask
from GraspAgent_2.training.sample_random_grasp import pose_interpolation
from GraspAgent_2.training.Sample_contrastive_pairs import sample_contrastive_pairs

lock = FileLock("file.lock")
max_samples_per_image = 1

freeze_backbone = False

training_buffer = online_data2()
training_buffer.main_modality = training_buffer.depth

bce_loss = nn.BCELoss()

print = custom_print

cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
cache_name = 'normals'
discrepancy_distance = 1.0

collision_expo = 1.0
firmness_expo = 1.0
generator_expo = 1.0

m = 0.2

import torch

def euclidean_triplet_loss(anchor, positive, negative, margin_signal):
    margin = torch.as_tensor(margin_signal * 1.0, device=anchor.device)
    d_pos  = (anchor - positive).pow(2).sum(dim=-1).sqrt()   # L2 distance
    d_neg  = (anchor - negative).pow(2).sum(dim=-1).sqrt()
    loss   = F.relu(d_pos - d_neg + margin)
    return loss.mean()

def hinge_loss(positive, negative,margin,k=.3):
    loss = torch.clamp((negative.squeeze() - positive.squeeze())   + margin*k, 0.)
    return loss

def cosine_triplet_loss(anchor, positive, negative, margin_signal):
    d_pos=(anchor*positive).sum()
    d_neg=(anchor*negative).sum()
    loss=torch.clamp(d_neg-d_pos+margin_signal*0.3,0)

    # margin_signal=torch.tensor([margin_signal],device=anchor.device)
    # d_pos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    # d_neg = 1 - F.cosine_similarity(anchor, negative, dim=-1)
    # loss = F.relu(d_pos - d_neg + margin_signal*0.3)
    return loss.mean()

def balanced_sampling(values, mask=None, exponent=2.0, balance_indicator=1.0):
    with torch.no_grad():
        max_ = values.max().item()
        min_ = values.min().item()
        range_ = max_ - min_
        # assert range_>0.0001


        if (not range_>0.) :
            # print(Fore.RED,'Warning: mode collapse detected',Fore.RESET)
            selection_probability=torch.rand_like(values)
        else:
            pivot_point = np.sqrt(np.abs(balance_indicator)) * np.sign(balance_indicator)
            xa = ((max_ - values) / range_) * pivot_point
            # xa=(1-values)* pivot_point
            selection_probability = ((1 - pivot_point) / 2 + xa + 0.5 * (1 - abs(pivot_point)))
            selection_probability = selection_probability ** exponent

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

        self.batch_size = 2

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
        self.background_detector_statistics = None
        self.grasp_quality_statistics=None

        '''initialize statistics records'''
        self.gripper_sampler_statistics = None
        self.critic_statistics = None
        self.data_tracker = None

        self.sampling_centroid = None
        self.diversity_momentum = 1.0

        self.skip_rate=None


    def initialize(self, n_samples=None):
        self.n_samples = n_samples
        self.prepare_data_loader()

        '''Moving rates'''

        self.skip_rate = MovingRate(YPG_model_key + '_skip_rate',
                                                   decay_rate=0.01,
                                                    initial_val=1.)

        self.gradient_moving_rate = MovingRate(YPG_model_key + '_gradient', decay_rate=0.01, initial_val=1000.)

        self.moving_collision_rate = MovingRate(YPG_model_key + '_collision', decay_rate=0.01,
                                                initial_val=1.)
        self.moving_firmness = MovingRate(YPG_model_key + '_firmness', decay_rate=0.01, initial_val=0.)
        self.moving_out_of_scope = MovingRate(YPG_model_key + '_out_of_scope', decay_rate=0.01,
                                              initial_val=1.)
        self.relative_sampling_timing = MovingRate(YPG_model_key + '_relative_sampling_timing',
                                                   decay_rate=0.01,
                                                    initial_val=1.)
        self.superior_A_model_moving_rate = MovingRate(YPG_model_key + '_superior_A_model',
                                                       decay_rate=0.01,
                                                       initial_val=0.)
        self.moving_anneling_factor = MovingRate(YPG_model_key + '_anneling_factor', decay_rate=0.01,
                                                 initial_val=0.)

        # self.moving_scores_std = MovingRate(YPG_model_key + '_scores_std', decay_rate=0.01,
        #                                     initial_val=.01)

        '''initialize statistics records'''

        self.bin_collision_statistics = TrainingTracker(name=YPG_model_key + '_bin_collision',
                                                        track_label_balance=True)
        self.objects_collision_statistics = TrainingTracker(name=YPG_model_key + '_objects_collision',
                                                            track_label_balance=True)

        self.background_detector_statistics = TrainingTracker(name=YPG_model_key + '_background_detector',
                                                              track_label_balance=False)


        self.gripper_sampler_statistics = TrainingTracker(name=YPG_model_key + '_gripper_sampler',
                                                          track_label_balance=False)

        self.grasp_quality_statistics = TrainingTracker(name=YPG_model_key + '_grasp_quality',
                                                        track_label_balance=True,decay_rate=0.001)

        self.critic_statistics = TrainingTracker(name=YPG_model_key + '_critic',
                                                 track_label_balance=False)

        self.data_tracker = DataTracker(name=YPG_model_key)

    def prepare_data_loader(self):
        file_ids = training_buffer.get_indexes()

        print(Fore.CYAN, f'Buffer size = {len(file_ids)}', Fore.RESET)
        dataset = YPGDataset2(data_pool=training_buffer, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers,
                                                  shuffle=True)
        self.size = len(dataset)
        self.data_loader = data_loader

    def prepare_model_wrapper(self):
        '''load  models'''
        gan = GANWrapper(YPG_model_key, YPG_G, YPG_D)
        gan.ini_models(train=True)


        gan.critic_adamW_optimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.999,weight_decay_=0)
        # gan.critic_sgd_optimizer(learning_rate=self.learning_rate*10,momentum=0.,weight_decay_=0)
        gan.generator_adamW_optimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999)
        # gan.generator_sgd_optimizer(learning_rate=self.learning_rate,momentum=0.9)

        return gan

    def step_discriminator(self,depth, mask,    gripper_pose, gripper_pose_ref ,pairs,cropped_spheres ):
        self.gan.critic.zero_grad()
        self.gan.critic_optimizer.zero_grad()

        '''self supervised critic learning'''
        # with torch.no_grad():
        #     generated_grasps_cat = torch.cat([gripper_pose, gripper_pose_ref], dim=0)

        with torch.no_grad():
            generated_grasps_stack = []
            for pair in pairs:
                index = pair[0]

                pred_pose = gripper_pose[index]
                label_pose = gripper_pose_ref[index]
                pair_pose = torch.stack([pred_pose, label_pose])
                generated_grasps_stack.append(pair_pose)
            generated_grasps_stack = torch.stack(generated_grasps_stack)
        # cropped_voxels=torch.stack(cropped_voxels)[:,None,...]

        # print(cropped_voxels.shape)
        anchor,positive_negative,scores = self.gan.critic(depth[None,None], generated_grasps_stack,pairs,mask[None,None],cropped_spheres,backbone=self.gan.generator.back_bone_)

        # print(score)
        # exit()
        #score[600,600]
        #score[mask # n_p
        # gen_scores_ = score.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        # ref_scores_ = score.permute(0, 2, 3, 1)[1, :, :, 0][mask]


        # generated_embedding=positive_negative[:,0]
        # ref_embedding=positive_negative[:,1]

        generated_scores=scores[:,0]
        ref_scores=scores[:,1]

        loss = torch.tensor(0., device=depth.device)

        for j in range(len(pairs)):
            target_index = pairs[j][0]
            margin = pairs[j][1]
            k = pairs[j][2]
            if k>0:
                loss+=(hinge_loss(positive=ref_scores[j],negative=generated_scores[j],margin=margin)**2)/self.batch_size
                # positive=ref_embedding[j]
                # negative=generated_embedding[j]
            else:
                loss+=(hinge_loss(positive=generated_scores[j],negative=ref_scores[j],margin=margin)**2)/self.batch_size
                # positive = generated_embedding[j]
                # negative = ref_embedding[j]



            # loss+=(cosine_triplet_loss(anchor, positive, negative, margin_signal=margin)**2)/batch_size

            # label = ref_score[j].squeeze()
            # pred_ = generated_score[j].squeeze()
            c=1
            # if k>0 and (label-pred_)>1:
            #     print(Fore.LIGHTMAGENTA_EX,'curriculum loss activated',Fore.RESET)
            #     c=-1
            #     margin=0


            # loss+=(torch.clamp((pred_ - label) * k *c+ margin, 0.)**2)/batch_size
        # loss=self.RGAN_D_loss(pairs,gen_scores_,ref_scores_)
        loss.backward()

        self.critic_statistics.loss=loss.item()
        self.gan.critic_optimizer.step()

        self.gan.critic.zero_grad()
        self.gan.critic_optimizer.zero_grad()

        print(Fore.LIGHTYELLOW_EX,
              f'd_loss={loss.item()}',
              Fore.RESET)

    def WGAN_D_loss(self,pairs,gen_scores_,ref_scores_):
        s=[]
        i=[]
        for j in range(len(pairs)):
            target_index = pairs[j][0]
            margin = pairs[j][1]
            k = pairs[j][2]
            if k==1:
                s.append(ref_scores_[target_index])
                i.append(gen_scores_[target_index])
            else:
                i.append(ref_scores_[target_index])
                s.append(gen_scores_[target_index])

        s=torch.stack(s).mean()
        i=torch.stack(i).mean()

        loss=i-s

        return loss

    def RGAN_D_loss(self,pairs,gen_scores_,ref_scores_):
        s=[]
        i=[]
        for j in range(len(pairs)):
            target_index = pairs[j][0]
            margin = pairs[j][1]
            k = pairs[j][2]
            if k==1:
                s.append(ref_scores_[target_index])
                i.append(gen_scores_[target_index])
            else:
                i.append(ref_scores_[target_index])
                s.append(gen_scores_[target_index])

        s=torch.stack(s)
        i=torch.stack(i)

        loss=torch.clamp(i-s.mean()+1,0).mean()+torch.clamp(i.mean()-s+1,0).mean()

        return loss



    def get_generator_loss(self, depth,mask, gripper_pose, gripper_pose_ref, pairs, objects_mask,cropped_spheres):


        # generated_grasps_cat = torch.cat([gripper_pose, gripper_pose_ref], dim=0)

        gripper_pose = gripper_pose[0].permute(1, 2, 0)[mask]

        generated_grasps_stack = []
        for pair in pairs:
            index = pair[0]
            pred_pose = gripper_pose[index]

            label_pose = gripper_pose_ref[index]

            pair_pose = torch.stack([pred_pose, label_pose])
            generated_grasps_stack.append(pair_pose)

        generated_grasps_stack = torch.stack(generated_grasps_stack)
        # cropped_voxels=torch.stack(cropped_voxels)[:,None,...]
        # print(cropped_voxels.shape)

        # critic_score = self.gan.critic(depth[None,None], generated_grasps_cat)
        anchor,positive_negative,scores = self.gan.critic(depth[None,None], generated_grasps_stack, pairs,mask[None,None],cropped_spheres,detach_backbone=True,backbone=self.gan.generator.back_bone_)

        # gen_scores_ = critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        # ref_scores_ = critic_score.permute(0, 2, 3, 1)[1, :, :, 0][mask]

        # gripper_pose = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask]
        # gripper_pose_ref = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :][mask]

        # gen_embedding = positive_negative[:,0]
        # ref_embedding = positive_negative[:,1]

        gen_scores = scores[:,0]
        ref_scores = scores[:,1]

        loss = torch.tensor(0., device=depth.device).float()

        for j in range(len(pairs)):
            target_index = pairs[j][0]
            margin = pairs[j][1]
            k = pairs[j][2]

            # loss+=(cosine_triplet_loss(anchor, positive=gen_embedding[j], negative=ref_embedding[j], margin_signal=0.))/batch_size
            loss += (hinge_loss(positive=gen_scores[j], negative=ref_scores[j], margin=0.) ** 1) / self.batch_size

            target_generated_pose = gripper_pose[target_index].detach()
            target_ref_pose = gripper_pose_ref[target_index].detach()

            # label = ref_score[j].squeeze()
            # pred_ = gen_scores_[j].squeeze()

            if k==1:
                print(Fore.LIGHTCYAN_EX,f'{target_ref_pose.cpu()} {target_generated_pose.cpu().detach()}  m={margin}',Fore.RESET)
            else:
                print(Fore.LIGHTGREEN_EX,f'{target_ref_pose.cpu()} {target_generated_pose.cpu().detach()}  m={margin}',Fore.RESET)
            print()

            # w=1 if k>0 else 0
            # loss += ((torch.clamp( label - pred_, 0.)) **1)/ batch_size

        return loss


    def visualize(self, pc,depth,mask,bin_mask,grasp_quality,grasp_collision):
        with torch.no_grad():
            # '''get parameters'''
            # pc, mask = depth_to_point_clouds(depth[0, 0].cpu().numpy(), camera)
            # pc = transform_to_camera_frame(pc, reverse=True)

            # downsampling_mask=np.random.random(pc.shape[0])>0.5

            # elevation_mask = (pc[:, -1] < 0.15)  # & downsampling_mask
            # mask[mask] = (mask[mask]) & elevation_mask
            # pc = pc[elevation_mask]

            '''generated grasps'''
            gripper_pose, _,_,_,gripper_pose2 = self.gan.generator(
                depth[None,None], mask[None,None],detach_backbone=True)

            gripper_pose = gripper_pose[0].permute(1, 2, 0)[mask]#.cpu().numpy()
            collision_with_objects_predictions = grasp_collision[0, 0][mask]
            collision_with_bin_predictions = grasp_collision[0, 1][mask]
            grasp_quality=grasp_quality[0,0][mask]
            grasp_quality=torch.clamp(grasp_quality,0.01,1)
            collision_mask = (collision_with_objects_predictions < .7) & (collision_with_bin_predictions<0.9)

            # sampling_p = 1. - collision_with_objects_predictions #** 2.0
            # dense_grasps_visualization(pc, gripper_poses,
            #                            view_mask=(gripper_sampling_mask & torch.from_numpy(objects_mask).cuda()),
            #                            sampling_p=sampling_p, view_all=False,exclude_collision=True)
            grasp_quality=torch.rand_like(grasp_quality)
            collision_mask=torch.rand_like(collision_mask.float())>0.

            dense_grasps_visualization(pc, gripper_pose,
                                       view_mask=(~bin_mask) ,
                                       sampling_p=grasp_quality, view_all=False, exclude_collision=True)

            # suction_head_predictions[~torch.from_numpy(objects_mask).cuda()] *= 0.

    def crop_square_patch(self,img: torch.Tensor, center: tuple, size: int, pad_value: float = 0.0):
        """
        img: [H, W] depth image (torch tensor)
        center: (row, col) pixel coordinates
        size: patch size (must be odd ideally, so it's symmetric)
        pad_value: value to use for padding (default=0.0)
        """
        r, c = center
        half = size // 2

        # Pad image on all sides
        padded = F.pad(img.unsqueeze(0).unsqueeze(0),  # add [N, C] dims
                       pad=(half, half, half, half),  # (left, right, top, bottom)
                       mode="constant", value=pad_value)
        padded = padded[0, 0]  # back to [H, W]

        # Shift center because of padding
        r += half
        c += half

        # Crop fixed-size patch
        patch = padded[r - half:r + half , c - half:c + half ]
        return patch

    def step_generator(self,depth,mask,bin_mask,pc,objects_mask,gripper_pose_ref,pairs,cropped_spheres):
        '''zero grad'''
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator.zero_grad(set_to_none=True)
        '''generated grasps'''
        gripper_pose, grasp_quality, background_detection, grasp_collision,gripper_pose2 = self.gan.generator(depth[None, None, ...],mask[None,None],
                                                                                                detach_backbone=freeze_backbone)

        gripper_pose_PW = gripper_pose[0].permute(1, 2, 0)[mask]
        grasp_collision = grasp_collision[0].permute(1, 2, 0)[mask]
        grasp_quality = grasp_quality[0].permute(1, 2, 0)[mask]

        '''loss computation'''
        gripper_collision_loss = torch.tensor(0., device=pc.device)
        gripper_quality_loss_ = torch.tensor(0., device=pc.device)
        background_detection_loss = torch.tensor(0., device=pc.device)



        label = bin_mask.float()
        # pc_numpy=pc
        # print(background_detection.shape)
        # print(label.shape)
        background_detection_loss += bce_loss(background_detection.squeeze()[mask], label)
        self.background_detector_statistics.update_confession_matrix(label.detach(),
                                                                     background_detection.squeeze().detach()[mask])

        for k in range(self.batch_size*2):
            '''gripper-object collision - 1'''
            while True:
                gripper_target_index = balanced_sampling(grasp_collision[:, 0].detach(),
                                                         mask=objects_mask.detach(),
                                                         exponent=30.0,
                                                         balance_indicator=self.objects_collision_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index].detach()  # .cpu().numpy()
                gripper_prediction_ = grasp_collision[gripper_target_index]
                gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()
                loss, counted = gripper_object_collision_loss(gripper_target_pose, gripper_target_point, pc,
                                                              objects_mask, gripper_prediction_,
                                                              self.objects_collision_statistics)
                if counted: break
            gripper_collision_loss += loss / self.batch_size

        for k in range(self.batch_size*2):
            '''gripper-bin collision - 1'''
            while True:
                gripper_target_index = balanced_sampling(grasp_collision[:, 1].detach(),
                                                         mask=objects_mask.detach(),
                                                         exponent=30.0,
                                                         balance_indicator=self.bin_collision_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index].detach()  # .cpu().numpy()
                gripper_prediction_ = grasp_collision[gripper_target_index]
                gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

                loss, counted = gripper_bin_collision_loss(gripper_target_pose, gripper_target_point, pc,
                                                           objects_mask, gripper_prediction_,
                                                           self.bin_collision_statistics)

                if counted: break
            gripper_collision_loss += loss / self.batch_size

        for k in range(self.batch_size*2):
            '''grasp quality'''
            while True:
                gripper_target_index = balanced_sampling(grasp_quality[:, 0].detach(),
                                                         mask=objects_mask.detach(),
                                                         exponent=30.0,
                                                         balance_indicator=self.grasp_quality_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index].detach()  # .cpu().numpy()
                gripper_prediction_ = grasp_quality[gripper_target_index]
                gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()
                loss, counted = gripper_quality_loss(gripper_target_pose, gripper_target_point, pc,
                                                      gripper_prediction_,
                                                     self.grasp_quality_statistics)

                if counted: break
            gripper_quality_loss_ += loss / self.batch_size


        gripper_sampling_loss = self.get_generator_loss(
            depth, mask, gripper_pose, gripper_pose_ref,
            pairs, objects_mask,cropped_spheres)

        assert not torch.isnan(gripper_sampling_loss).any(), f'{gripper_sampling_loss}'

        # gripper_sampling_loss-=weight*pred_/m

        print(Fore.LIGHTYELLOW_EX,
              f'g_loss={gripper_sampling_loss.item()}',
              Fore.RESET)

        loss = gripper_sampling_loss + background_detection_loss  + gripper_collision_loss  + gripper_quality_loss_

        with torch.no_grad():
            self.gripper_sampler_statistics.loss = gripper_sampling_loss.item()
        # if abs(loss.item())>0.0:
        # try:

        loss.backward()

        self.gan.generator_optimizer.step()

        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator_optimizer.zero_grad(set_to_none=True)
        self.gan.critic_optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            self.background_detector_statistics.loss = background_detection_loss.item()

    def step_generator_without_sampler(self,depth,mask,bin_mask,pc,objects_mask):
        '''zero grad'''
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator.zero_grad(set_to_none=True)
        '''generated grasps'''
        gripper_pose, grasp_quality, background_detection, grasp_collision,gripper_pose2 = self.gan.generator(depth[None, None, ...],mask[None,None],
                                                                                                detach_backbone=freeze_backbone)

        gripper_pose_PW = gripper_pose[0].permute(1, 2, 0)[mask]
        grasp_collision = grasp_collision[0].permute(1, 2, 0)[mask]
        grasp_quality = grasp_quality[0].permute(1, 2, 0)[mask]

        '''loss computation'''
        gripper_collision_loss = torch.tensor(0., device=pc.device)
        gripper_quality_loss_ = torch.tensor(0., device=pc.device)
        background_detection_loss = torch.tensor(0., device=pc.device)


        label = bin_mask.float()
        # pc_numpy=pc
        # print(background_detection.shape)
        # print(label.shape)
        background_detection_loss += bce_loss(background_detection.squeeze()[mask], label)
        self.background_detector_statistics.update_confession_matrix(label.detach(),
                                                                     background_detection.squeeze().detach()[mask])

        for k in range(self.batch_size*2):
            '''gripper-object collision - 1'''
            while True:
                gripper_target_index = balanced_sampling(grasp_collision[:, 0].detach(),
                                                         mask=objects_mask.detach(),
                                                         exponent=30.0,
                                                         balance_indicator=self.objects_collision_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index].detach()  # .cpu().numpy()
                gripper_prediction_ = grasp_collision[gripper_target_index]
                gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()
                loss, counted = gripper_object_collision_loss(gripper_target_pose, gripper_target_point, pc,
                                                              objects_mask, gripper_prediction_,
                                                              self.objects_collision_statistics)
                if counted: break
            gripper_collision_loss += loss / self.batch_size

        for k in range(self.batch_size*2):
            '''gripper-bin collision - 1'''
            while True:
                gripper_target_index = balanced_sampling(grasp_collision[:, 1].detach(),
                                                         mask=objects_mask.detach(),
                                                         exponent=30.0,
                                                         balance_indicator=self.bin_collision_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index].detach()  # .cpu().numpy()
                gripper_prediction_ = grasp_collision[gripper_target_index]
                gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

                loss, counted = gripper_bin_collision_loss(gripper_target_pose, gripper_target_point, pc,
                                                           objects_mask, gripper_prediction_,
                                                           self.bin_collision_statistics)

                if counted: break
            gripper_collision_loss += loss / self.batch_size

        for k in range(self.batch_size*2):
            '''grasp quality'''
            while True:
                gripper_target_index = balanced_sampling(grasp_quality[:, 0].detach(),
                                                         mask=objects_mask.detach(),
                                                         exponent=30.0,
                                                         balance_indicator=self.grasp_quality_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index].detach()  # .cpu().numpy()
                gripper_prediction_ = grasp_quality[gripper_target_index]
                gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()
                loss, counted = gripper_quality_loss(gripper_target_pose, gripper_target_point, pc,
                                                      gripper_prediction_,
                                                     self.grasp_quality_statistics)

                if counted: break
            gripper_quality_loss_ += loss / self.batch_size



        loss =  background_detection_loss  + gripper_collision_loss  + gripper_quality_loss_

        loss.backward()

        self.gan.generator_optimizer.step()

        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator_optimizer.zero_grad(set_to_none=True)
        self.gan.critic_optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            self.background_detector_statistics.loss = background_detection_loss.item()

    def depth_standardization(self,depth,mask):
        mean_ = depth[mask].mean()
        depth_ = (depth.clone() - mean_) / 30
        depth_[~mask] = 0.
        return depth_

    def begin(self):

        pi = progress_indicator('Begin new training round: ', max_limit=len(self.data_loader))
        gripper_pose = None
        # apply_static_spectral_norm(self.gan.generator.back_bone)
        # self.export_check_points()
        # exit()
        for i, batch in enumerate(self.data_loader, 0):
            depth, file_ids = batch # [b,480,712]
            depth = depth.cuda().float()[0]

            pi.step(i)

            with torch.no_grad():


                pc, mask = depth_to_point_clouds(depth, camera)
                mask.requires_grad_(False)
                pc = transform_to_camera_frame_torch(pc, reverse=True).requires_grad_(False)

                # print(pc.max(dim=0))
                # print(pc.min(dim=0))
                # print(pc.mean(dim=0))
                #
                # exit()
                # depth=self.depth_standardization(depth,mask)#+torch.normal(mean=0.0, std=0.1, size=(1,),device='cuda')
                # depth[~mask]=0.

                '''background detection head'''
                bin_mask = analytical_bin_mask(pc, file_ids)

                if bin_mask is None: continue
                bin_mask.requires_grad_(False)
                objects_mask = bin_mask <= 0.5

            for k in range(max_samples_per_image):
                print('------------------------------------------------------------------')

                with torch.no_grad():
                    gripper_pose,grasp_quality,_,grasp_collision ,gripper_pose2= self.gan.generator(
                        depth[None,None,...],mask[None,None], detach_backbone=True)  # [1,7,h,w]




                    # view_image(depth.detach().cpu().numpy())
                    # print(grasp_quality.mean().item())
                    # view_image(grasp_quality[0,0].detach().cpu().numpy())
                    # continue

                    # self.visualize(pc, depth, mask, bin_mask,grasp_quality.detach_(),grasp_collision.detach_())
                    # continue

                    x=self.superior_A_model_moving_rate.val

                    # tou = 4*x* (x-1)+1
                    # tou=max(0.01, min(tou, 0.99))**0.5
                    tou=1-x

                    f=self.grasp_quality_statistics.accuracy*((1-grasp_quality.detach())**2)+(1-self.grasp_quality_statistics.accuracy)
                    annealing_factor=tou*f
                    print(f'mean_annealing_factor= {annealing_factor.mean()}, tou={tou}')

                    gripper_pose_ref = pose_interpolation(gripper_pose, objects_mask,annealing_factor=annealing_factor,tou=tou)

                    if i % int(100) == 0 and i != 0 and k == 0:
                        try:
                            self.export_check_points()
                            self.save_statistics()
                            # self.load_action_model()
                        except Exception as e:
                            print(Fore.RED, str(e), Fore.RESET)
                    if i % 10 == 0 and k == 0:
                        self.analysis_view(gripper_pose,mask)

                    assert not torch.isnan(gripper_pose).any(), f'{gripper_pose}'
                    assert not torch.isnan(gripper_pose_ref).any(), f'{gripper_pose_ref[0,1000:1110]},    {gripper_pose[0,1000:1110]}'

                    counted,pairs,sampling_centroid,sampler_samples=sample_contrastive_pairs(pc,mask, bin_mask, gripper_pose, gripper_pose_ref,
                                              self.sampling_centroid, self.batch_size,tou,grasp_quality.detach(),self.superior_A_model_moving_rate)
                    if not counted:
                        self.skip_rate.update(1)
                        continue
                    else:
                        self.skip_rate.update(0)

                    self.sampling_centroid=sampling_centroid

                    '''prepare cropped point clouds'''
                    # cropped_voxels = []
                    cropped_spheres=[]
                    radius=0.08
                    batch_features_list = []
                    batch_indices_list = []
                    space_range = 2.0
                    voxel_size =0.02
                    grid_size = int(space_range / voxel_size)
                    b=0
                    for pair in pairs:
                        index=pair[0]

                        center=pc[index]


                        sub_pc=crop_cube(pc, center, cube_size=2*radius)
                        # sub_pc = crop_sphere_torch(pc, center,s radius)
                        sub_pc -= center
                        sub_pc/=radius


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
                        )+b

                        indices = torch.cat([
                            batch_indices,
                            voxel_coords[:, [2, 1, 0]]  # z, y, x
                        ], dim=1)

                        batch_indices_list.append(indices)
                        batch_features_list.append(voxel_features)

                        b+=1

                    batch_features = torch.cat(batch_features_list, dim=0)
                    batch_indices = torch.cat(batch_indices_list, dim=0)

                    cropped_spheres = spconv.SparseConvTensor(
                        features=batch_features,  # (M, C=3)
                        indices=batch_indices,  # (M, 4)
                        spatial_shape=[grid_size] * 3,
                        batch_size=self.batch_size
                    )




                '''zero grad'''
                self.gan.critic.zero_grad(set_to_none=True)
                self.gan.generator.zero_grad(set_to_none=True)

                gripper_pose=gripper_pose[0].permute(1,2,0)[mask]
                gripper_pose_ref=gripper_pose_ref[0].permute(1,2,0)[mask]

                self.step_discriminator(depth,mask,  gripper_pose, gripper_pose_ref, pairs,cropped_spheres)

                # if sampler_samples==batch_size:
                self.step_generator(depth,mask,bin_mask,pc,objects_mask,gripper_pose_ref,pairs,cropped_spheres)
                # else:
                #     self.step_generator_without_sampler(depth,mask,bin_mask,pc,objects_mask)

                # continue

        pi.end()

        self.export_check_points()
        self.clear()

    def analysis_view(self,gripper_pose,mask):
        cuda_memory_report()

        gripper_poses = gripper_pose[0].permute(1, 2, 0)[mask].detach()  # .cpu().numpy()
        gripper_poses[:,0:3]=F.normalize(gripper_poses[:,0:3],p=2,dim=1,eps=1e-8)
        gripper_poses[:,3:5]=F.normalize(gripper_poses[:,3:5],p=2,dim=1,eps=1e-8)

        self.view_result(gripper_poses)
        beta_angles = torch.atan2(gripper_poses[:, 3], gripper_poses[:, 4])
        print(f'beta angles variance = {beta_angles.std()}')
        subindexes = torch.randperm(gripper_poses.size(0))
        if subindexes.shape[0] > 1000: subindexes = subindexes[:1000]
        all_values = gripper_poses[subindexes].detach()
        dist_values = torch.clamp((all_values[:, 5:6]), 0., 1.).clone()
        width_values = torch.clamp((all_values[:, 6:7]), 0., 1.).clone()
        beta_values = (all_values[:, 3:5]).clone()
        beta_diversity = (1.001 - F.cosine_similarity(beta_values[None, ...], beta_values[:, None, :],
                                                      dim=-1)).mean() / 2
        print(f'Mean separation for beta : {beta_diversity}')
        print(f'Mean separation for width : {torch.cdist(width_values, width_values, p=1).mean()}')
        print(f'Mean separation for distance : {torch.cdist(dist_values, dist_values, p=1).mean()}')
        self.moving_anneling_factor.update(beta_diversity.item())

    def view_result(self, values):
        with torch.no_grad():

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
            self.moving_firmness.view()
            self.moving_out_of_scope.view()
            self.relative_sampling_timing.view()
            self.moving_anneling_factor.view()
            self.superior_A_model_moving_rate.view()
            self.skip_rate.view()

            self.bin_collision_statistics.print()
            self.objects_collision_statistics.print()
            self.grasp_quality_statistics.print()

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
        self.skip_rate.save()

        self.critic_statistics.save()
        self.gripper_sampler_statistics.save()

        self.data_tracker.save()

        self.gradient_moving_rate.save()

        self.bin_collision_statistics.save()
        self.objects_collision_statistics.save()
        self.background_detector_statistics.save()
        self.grasp_quality_statistics.save()


    def export_check_points(self):
        self.gan.export_models()
        self.gan.export_optimizers()


    def clear(self):
        self.critic_statistics.clear()

        self.bin_collision_statistics.clear()
        self.objects_collision_statistics.clear()
        self.gripper_sampler_statistics.clear()
        self.background_detector_statistics.clear()
        self.grasp_quality_statistics.clear()

def train_N_grasp_GAN(n=1):
    lr = 1e-4
    Train_grasp_GAN = TrainGraspGAN(n_samples=None, learning_rate=lr)
    torch.cuda.empty_cache()
    # torch.autograd.set_detect_anomaly(True)

    for i in range(n):
        # try:
            cuda_memory_report()
            Train_grasp_GAN.initialize(n_samples=None)
            # fix_weight_scales(Train_grasp_GAN.gan.generator.grasp_quality)
            # exit()
            Train_grasp_GAN.begin()
        # except Exception as e:
        #     print(Fore.RED, str(e), Fore.RESET)
        #     del Train_grasp_GAN
        #     torch.cuda.empty_cache()
        #     Train_grasp_GAN = TrainGraspGAN(n_samples=None, learning_rate=lr)

    # del Train_grasp_GAN

if __name__ == "__main__":
    # grasp_quality_statistics = TrainingTracker(name=YPG_model_key + '_grasp_quality',
    #                                                 track_label_balance=True)
    train_N_grasp_GAN(n=10000)

