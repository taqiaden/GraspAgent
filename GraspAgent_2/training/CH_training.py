import os
import numpy as np
from colorama import Fore
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from GraspAgent_2.model.CH_model import CH_model_key, CH_D, CH_G
from GraspAgent_2.sim_hand_s.Casia_hand_env import CasiaHandEnv
from GraspAgent_2.training.sample_random_grasp import ch_pose_interpolation
from GraspAgent_2.utils.Online_clustering import OnlingClustering
from GraspAgent_2.utils.quat_operations import quat_rotate_vector
from GraspAgent_2.utils.weigts_normalization import scale_all_weights, fix_weight_scales
from Online_data_audit.data_tracker import gripper_grasp_tracker, DataTracker
from check_points.check_point_conventions import GANWrapper
from lib.IO_utils import custom_print
from lib.cuda_utils import cuda_memory_report
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from records.training_satatistics import TrainingTracker, MovingRate
import torch

iter_per_scene = 1

batch_size = 2
freeze_G_backbone = False
freeze_D_backbone = False

max_n = 50

bce_loss = nn.BCELoss()

print = custom_print

cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
cache_name = 'normals'
discrepancy_distance = 1.0

collision_expo = 1.0
firmness_expo = 1.0
generator_expo = 1.0

m = 0.2


def balanced_sampling(values, mask=None, exponent=2.0, balance_indicator=1.0,eps=1e-4):
    with torch.no_grad():
        max_ = values.max().item()
        min_ = values.min().item()
        range_ = max_ - min_

        if (not range_>0.) :
            selection_probability=torch.rand_like(values)
        else:
            pivot_point = np.sqrt(np.abs(balance_indicator)) * np.sign(balance_indicator)
            xa=(1-values)* pivot_point
            selection_probability = ((1 - pivot_point) / 2 + xa + 0.5 * (1 - abs(pivot_point)))
            selection_probability = selection_probability ** exponent
            selection_probability+=eps
        # try:
        if mask is None:
            dist = Categorical(probs=selection_probability)
        else:
            dist = MaskedCategorical(probs=selection_probability, mask=mask)

        target_index = dist.sample()
        # except Exception as e:
        #     print(str(e))
        #     print(selection_probability)
        #     print(mask)
        #     print(mask.sum())
        #     print(values.mean())
        #     print(values.max())
        #     print(values.min())
        #     print(values.std())
        #     exit()

        return target_index

class TrainGraspGAN:
    def __init__(self, n_samples=None, epochs=1, learning_rate=5e-5):

        self.n_samples = n_samples
        self.size = n_samples
        self.epochs = epochs
        self.learning_rate = learning_rate

        '''model wrapper'''
        self.gan = self.prepare_model_wrapper()

        '''Moving rates'''
        self.moving_collision_rate = None
        self.relative_sampling_timing = None
        self.superior_A_model_moving_rate = None

        self.bin_collision_statistics = None
        self.objects_collision_statistics = None
        self.grasp_quality_statistics=None

        '''initialize statistics records'''
        self.gripper_sampler_statistics = None
        self.critic_statistics = None
        self.critic_statistics = None
        self.data_tracker = None


        self.tmp_pose_record=[]

        self.last_pose_center_path=CH_model_key+'_pose_center'
        if os.path.exists(self.last_pose_center_path):
            self.sampling_centroid = torch.load(self.last_pose_center_path).cuda()
            if self.sampling_centroid.shape!=10: self.sampling_centroid = torch.tensor([0, 1, 0, 0, 0.5,0.5,0.5,  0.,0.,0.],
                                                        device='cuda')
        else: self.sampling_centroid = torch.tensor([0, 1, 0, 0, 0.5,0.5,0.5,  0.,0.,0.],
                                                        device='cuda')
        root_dir = os.getcwd()  # current working directory

        self.ch_env = CasiaHandEnv(root=root_dir + "/GraspAgent_2/sim_hand_s/speed_hand/",max_obj_per_scene=1)

        self.tou = 1

        self.quat_centers=OnlingClustering(key_name=CH_model_key+'_quat',number_of_centers=16,vector_size=4,decay_rate=0.01,is_quat=True,dist_threshold=0.77)
        self.fingers_centers=OnlingClustering(key_name=CH_model_key+'_fingers',number_of_centers=9,vector_size=3,decay_rate=0.01,use_euclidean_dist=True,dist_threshold=0.2)

    def initialize(self, n_samples=None):
        self.n_samples = n_samples

        '''Moving rates'''


        self.moving_collision_rate = MovingRate(CH_model_key + '_collision', decay_rate=0.01,
                                                initial_val=1.)
        self.relative_sampling_timing = MovingRate(CH_model_key + '_relative_sampling_timing',
                                                   decay_rate=0.01,
                                                    initial_val=1.)
        self.superior_A_model_moving_rate = MovingRate(CH_model_key + '_superior_A_model',
                                                       decay_rate=0.01,
                                                       initial_val=0.)

        # self.superior_A_model_moving_rate.moving_rate=0
        # self.superior_A_model_moving_rate.save()
        # exit()

        '''initialize statistics records'''
        self.bin_collision_statistics = TrainingTracker(name=CH_model_key + '_bin_collision',
                                                        track_label_balance=True)
        self.objects_collision_statistics = TrainingTracker(name=CH_model_key + '_objects_collision',
                                                            track_label_balance=True)

        self.gripper_sampler_statistics = TrainingTracker(name=CH_model_key + '_gripper_sampler',
                                                          track_label_balance=False)

        self.grasp_quality_statistics = TrainingTracker(name=CH_model_key + '_grasp_quality',
                                                        track_label_balance=True,decay_rate=0.001)

        self.critic_statistics = TrainingTracker(name=CH_model_key + '_critic',
                                                  track_label_balance=False)

        self.data_tracker = DataTracker(name=CH_model_key)

    def prepare_model_wrapper(self):
        '''load  models'''
        gan = GANWrapper(CH_model_key, CH_G, CH_D)
        gan.ini_models(train=True)

        gan.critic_adamW_optimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.999)
        # gan.critic_sgd_optimizer(learning_rate=self.learning_rate*1,momentum=0.)
        gan.generator_adamW_optimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999)
        # gan.generator_sgd_optimizer(learning_rate=self.learning_rate*10,momentum=0.)

        return gan

    def step_discriminator(self,depth,   gripper_pose, gripper_pose_ref ,pairs,floor_mask ):
        '''zero grad'''
        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.critic.zero_grad()
        self.gan.critic_optimizer.zero_grad()

        '''self supervised critic learning'''
        with torch.no_grad():

            generated_grasps_stack=[]
            for pair in pairs:
                index=pair[0]

                pred_pose=gripper_pose[index]
                label_pose=gripper_pose_ref[index]
                pair_pose=torch.stack([pred_pose,label_pose])
                generated_grasps_stack.append(pair_pose)
            generated_grasps_stack=torch.stack(generated_grasps_stack)
        # cropped_voxels=torch.stack(cropped_voxels)[:,None,...]

        # print(cropped_voxels.shape)
        score = self.gan.critic(depth[None,None,...], generated_grasps_stack,pairs,~floor_mask.view(1,1,600,600),detach_backbone=freeze_D_backbone)

        # print(score)
        # exit()
        # gen_scores_ = score.permute(0, 2, 3, 1)[0, :, :, 0].reshape(-1)
        # ref_scores_ = score.permute(0, 2, 3, 1)[1, :, :, 0].reshape(-1)

        gen_scores_=score[:,0]
        ref_scores_=score[:,1]

        loss = torch.tensor(0., device=depth.device)

        for j in range(len(pairs)):
            target_index = pairs[j][0]
            k = pairs[j][1]
            margin=pairs[j][2]

            label = ref_scores_[j].squeeze()
            pred_ = gen_scores_[j].squeeze()

            c = 1
            if k > 0 and (label - pred_) > 1:
                print(Fore.LIGHTMAGENTA_EX, 'curriculum loss activated', Fore.RESET)
                c = -1
                margin = 0

            loss+=(torch.clamp((pred_ - label) * k *c+ margin, 0.)**1  )/batch_size
            # loss+=l/batch_size
        # loss=self.RGAN_D_loss(pairs,gen_scores_,ref_scores_)
        loss.backward()

        self.critic_statistics.loss=loss.item()
        self.gan.critic_optimizer.step()

        self.gan.critic.zero_grad()
        self.gan.critic_optimizer.zero_grad()

        print(  Fore.LIGHTYELLOW_EX,f'd_loss={loss.item()}',
              Fore.RESET)

    def get_generator_loss(self, depth, gripper_pose, gripper_pose_ref, pairs,floor_mask):

        gripper_pose = gripper_pose[0].permute(1, 2, 0).reshape(360000, 10)

        # generated_grasps_cat = torch.cat([gripper_pose, gripper_pose_ref], dim=0)

        generated_grasps_stack=[]
        for pair in pairs:
            index=pair[0]
            pred_pose=gripper_pose[index]

            label_pose=gripper_pose_ref[index]
            pair_pose=torch.stack([pred_pose,label_pose])
            generated_grasps_stack.append(pair_pose)

        generated_grasps_stack=torch.stack(generated_grasps_stack)
        # cropped_voxels=torch.stack(cropped_voxels)[:,None,...]
        # print(cropped_voxels.shape)
        # cuda_memory_report()
        critic_score = self.gan.critic(depth[None,None,...], generated_grasps_stack,pairs,~floor_mask.view(1,1,600,600),detach_backbone=True)

        # cuda_memory_report()
        # critic_score = self.gan.critic(pc, generated_grasps_cat, detach_backbone=True)

        # gen_scores_ = critic_score.permute(0, 2, 3, 1)[0, :, :, 0].reshape(-1)
        # ref_scores_ = critic_score.permute(0, 2, 3, 1)[1, :, :, 0].reshape(-1)

        # gripper_pose = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(-1)
        # gripper_pose_ref = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :].reshape(-1)

        gen_scores_ = critic_score[:,0]
        ref_scores_ = critic_score[:,1]

        loss = torch.tensor(0., device=depth.device)

        for j in range(len(pairs)):
            target_index = pairs[j][0]
            k = pairs[j][1]
            margin=pairs[j][2]

            target_generated_pose = gripper_pose[target_index].detach()
            target_ref_pose = gripper_pose_ref[target_index].detach()

            label = ref_scores_[j].squeeze()
            pred_ = gen_scores_[j].squeeze()

            if margin!=1:
                print(Fore.LIGHTYELLOW_EX,f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()} s=[{label.cpu()},{pred_.cpu()}, m={margin}] ',Fore.RESET)
            elif k==1:
                print(Fore.LIGHTCYAN_EX,f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()} s=[{label.cpu()},{pred_.cpu()}] ',Fore.RESET)
            else:
                print(Fore.LIGHTGREEN_EX,f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()} s=[{label.cpu()},{pred_.cpu()}] ',Fore.RESET)

            v2 = quat_rotate_vector(target_ref_pose[0:4].cpu().tolist(), [0, 1, 0])
            v3 = quat_rotate_vector(target_generated_pose[0:4].cpu().tolist(), [0, 1, 0])

            print('ref approach: ',v2,' ge approach: ',v3)
            print()

            # w=1 if k>0 else 0
            loss += ((torch.clamp( label - pred_, 0.)) **2)/ batch_size

        return loss

    def step_generator(self,depth,floor_mask,pc,gripper_pose_ref,pairs):
        '''zero grad'''
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator.zero_grad(set_to_none=True)
        # self.gan.critic.eval()

        '''generated grasps'''
        # cuda_memory_report()
        gripper_pose, grasp_quality,  grasp_collision = self.gan.generator(depth[None, None, ...],~floor_mask.view(1,1,600,600),
                                                                                                detach_backbone=freeze_G_backbone)
        # cuda_memory_report()
        # exit()
        gripper_pose_PW = gripper_pose[0].permute(1, 2, 0).reshape(360000,10)
        object_collision = grasp_collision[0,0].reshape(-1)
        floor_collision = grasp_collision[0,1].reshape(-1)

        grasp_quality = grasp_quality[0,0].reshape(-1)

        '''loss computation'''
        gripper_collision_loss = torch.tensor(0., device=depth.device)
        gripper_quality_loss_ = torch.tensor(0., device=depth.device)



        for k in range(batch_size*2):
            '''gripper-object collision'''
            gripper_target_index = balanced_sampling(object_collision.detach(),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.objects_collision_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = object_collision[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

            contact_with_obj , contact_with_floor=self.check_collision(gripper_target_point,gripper_target_pose,view=False)

            label = torch.ones_like(gripper_prediction_) if contact_with_obj else torch.zeros_like(gripper_prediction_)

            object_collision_loss = bce_loss(gripper_prediction_, label)

            with torch.no_grad():
                self.objects_collision_statistics.loss = object_collision_loss.item()
                self.objects_collision_statistics.update_confession_matrix(label.detach(),
                                                                      gripper_prediction_.detach())

            gripper_collision_loss+=object_collision_loss/batch_size


        for k in range(batch_size*2):
            '''gripper-bin collision'''
            gripper_target_index = balanced_sampling(floor_collision.detach(),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.bin_collision_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = floor_collision[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

            contact_with_obj , contact_with_floor = self.check_collision(gripper_target_point, gripper_target_pose, view=False)

            label = torch.ones_like(gripper_prediction_) if contact_with_floor else torch.zeros_like(gripper_prediction_)

            floor_collision_loss = bce_loss(gripper_prediction_, label)

            with torch.no_grad():
                self.bin_collision_statistics.loss = floor_collision_loss.item()
                self.bin_collision_statistics.update_confession_matrix(label.detach(),
                                                                           gripper_prediction_.detach())

            gripper_collision_loss += floor_collision_loss / batch_size

        for k in range(batch_size*2):
            '''grasp quality'''
            '''gripper-bin collision'''
            gripper_target_index = balanced_sampling(grasp_quality.detach(),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.grasp_quality_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = grasp_quality[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

            grasp_success,initial_collision = self.evaluate_grasp(gripper_target_point, gripper_target_pose, view=False)

            label = torch.ones_like(gripper_prediction_) if grasp_success else torch.zeros_like(gripper_prediction_)

            grasp_quality_loss = bce_loss(gripper_prediction_, label)

            with torch.no_grad():
                self.grasp_quality_statistics.loss = grasp_quality_loss.item()
                self.grasp_quality_statistics.update_confession_matrix(label.detach(),
                                                                       gripper_prediction_.detach())

            gripper_quality_loss_ += grasp_quality_loss / batch_size

        gripper_sampling_loss = self.get_generator_loss(
            depth,  gripper_pose, gripper_pose_ref,
            pairs,floor_mask)

        assert not torch.isnan(gripper_sampling_loss).any(), f'{gripper_sampling_loss}'

        print(Fore.LIGHTYELLOW_EX,
              f'g_loss={gripper_sampling_loss.item()}',
              Fore.RESET)

        loss = gripper_sampling_loss+gripper_collision_loss+gripper_quality_loss_ #+ background_detection_loss * 30 + gripper_collision_loss + 10 * gripper_quality_loss_

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


    def step_generator_without_sampler(self,depth,floor_mask,pc):
        '''zero grad'''
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator.zero_grad(set_to_none=True)
        # self.gan.critic.eval()

        '''generated grasps'''
        # cuda_memory_report()
        gripper_pose, grasp_quality,  grasp_collision = self.gan.generator(depth[None, None, ...],~floor_mask.view(1,1,600,600),
                                                                                                detach_backbone=freeze_G_backbone)
        # cuda_memory_report()
        # exit()
        gripper_pose_PW = gripper_pose[0].permute(1, 2, 0).reshape(360000,10)
        object_collision = grasp_collision[0,0].reshape(-1)
        floor_collision = grasp_collision[0,1].reshape(-1)

        grasp_quality = grasp_quality[0,0].reshape(-1)

        '''loss computation'''
        gripper_collision_loss = torch.tensor(0., device=depth.device)
        gripper_quality_loss_ = torch.tensor(0., device=depth.device)



        for k in range(batch_size*2):
            '''gripper-object collision'''
            gripper_target_index = balanced_sampling(object_collision.detach(),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.objects_collision_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = object_collision[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

            contact_with_obj , contact_with_floor=self.check_collision(gripper_target_point,gripper_target_pose,view=False)

            label = torch.ones_like(gripper_prediction_) if contact_with_obj else torch.zeros_like(gripper_prediction_)

            object_collision_loss = bce_loss(gripper_prediction_, label)

            with torch.no_grad():
                self.objects_collision_statistics.loss = object_collision_loss.item()
                self.objects_collision_statistics.update_confession_matrix(label.detach(),
                                                                      gripper_prediction_.detach())

            gripper_collision_loss+=object_collision_loss/batch_size


        for k in range(batch_size*2):
            '''gripper-bin collision'''
            gripper_target_index = balanced_sampling(floor_collision.detach(),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.bin_collision_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = floor_collision[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

            contact_with_obj , contact_with_floor = self.check_collision(gripper_target_point, gripper_target_pose, view=False)

            label = torch.ones_like(gripper_prediction_) if contact_with_floor else torch.zeros_like(gripper_prediction_)

            floor_collision_loss = bce_loss(gripper_prediction_, label)

            with torch.no_grad():
                self.bin_collision_statistics.loss = floor_collision_loss.item()
                self.bin_collision_statistics.update_confession_matrix(label.detach(),
                                                                           gripper_prediction_.detach())

            gripper_collision_loss += floor_collision_loss / batch_size

        for k in range(batch_size*2):
            '''grasp quality'''
            '''gripper-bin collision'''
            gripper_target_index = balanced_sampling(grasp_quality.detach(),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.grasp_quality_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = grasp_quality[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

            grasp_success,initial_collision = self.evaluate_grasp(gripper_target_point, gripper_target_pose, view=False)

            label = torch.ones_like(gripper_prediction_) if grasp_success else torch.zeros_like(gripper_prediction_)

            grasp_quality_loss = bce_loss(gripper_prediction_, label)

            with torch.no_grad():
                self.grasp_quality_statistics.loss = grasp_quality_loss.item()
                self.grasp_quality_statistics.update_confession_matrix(label.detach(),
                                                                       gripper_prediction_.detach())

            gripper_quality_loss_ += grasp_quality_loss / batch_size

        loss = gripper_collision_loss+gripper_quality_loss_ #+ background_detection_loss * 30 + gripper_collision_loss + 10 * gripper_quality_loss_

        loss.backward()
        self.gan.generator_optimizer.step()

        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator_optimizer.zero_grad(set_to_none=True)
        self.gan.critic_optimizer.zero_grad(set_to_none=True)

    def check_collision(self,target_point,target_pose,view=False):
        with torch.no_grad():
            quat, fingers, shifted_point = self.process_pose(target_point, target_pose, view=view)

        return self.ch_env.check_collision(hand_pos=shifted_point,hand_quat=quat,hand_fingers=None,view=False)

    def process_pose(self,target_point, target_pose, view=False):
        target_pose_ = target_pose.clone()
        target_point_ = np.copy(target_point)

        quat = target_pose_[:4].cpu().tolist()

        fingers = target_pose_[4:-1].cpu().tolist()

        transition = target_pose_[4+3:].cpu().numpy() / 100

        projected_transition = quat_rotate_vector(quat, [1, 0, 0])*transition[0]

        # approach = quat_rotate_vector(np.array(quat), np.array([0, 0, 1]))
        # projected_transition = approach * transition

        shifted_point = (target_point_ + projected_transition).tolist()

        if view:
            print()
            print('quat: ',quat)
            print('fingers: ',fingers)
            print('transition: ',transition)
            # print('projected_transition: ',projected_transition)
            print('shifted_point: ',shifted_point)

        return quat,fingers,shifted_point

    def evaluate_grasp(self, target_point, target_pose, view=False,shake_intensity=None):

        with torch.no_grad():
            quat,fingers,shifted_point=self.process_pose(target_point, target_pose, view=view)

            in_scope, grasp_success, contact_with_obj, contact_with_floor = self.ch_env.check_graspness(
                hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers,
               view=view,shake_intensity=shake_intensity)

            initial_collision=contact_with_obj or contact_with_floor

            # print('in_scope, grasp_success, contact_with_obj, contact_with_floor :',in_scope, grasp_success, contact_with_obj, contact_with_floor )

            if grasp_success is not None:
                if grasp_success and not contact_with_obj and not contact_with_floor:
                    return True and in_scope,initial_collision

        return False, initial_collision


    def sample_contrastive_pairs(self,pc,  floor_mask, gripper_pose, gripper_pose_ref,
                                 sampling_centroid, batch_size, annealing_factor, grasp_quality,
                                 superior_A_model_moving_rate):

        pairs = []

        selection_mask = ~floor_mask
        grasp_quality=grasp_quality[0,0].reshape(-1)
        gripper_pose_PW = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000,10)
        clipped_gripper_pose_PW=gripper_pose_PW.clone()
        clipped_gripper_pose_PW[:,4:4+3]=torch.clip(clipped_gripper_pose_PW[:,4:4+3],0,1)
        gripper_pose_ref_PW = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000,10)

        # grasp_quality = grasp_quality.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        # max_ = grasp_quality.max()
        # min_ = grasp_quality.min()
        # grasp_quality = (grasp_quality - min_) / (max_ - min_)
        def norm_(gamma ,expo_=1.0,min=0.01):
            gamma = (gamma - gamma.min()) / (
                    gamma.max() - gamma.min())
            gamma = gamma ** expo_
            gamma=torch.clamp(gamma,min)
            return gamma

        gamma_dive = norm_((1.001 - F.cosine_similarity(clipped_gripper_pose_PW,
                                                        sampling_centroid[None, :], dim=-1) ) /2 ,1)
        gamma_dive *= norm_((1.001 - F.cosine_similarity(gripper_pose_ref_PW,
                                                        sampling_centroid[None, :], dim=-1) ) /2 ,1)

        # selection_p = compute_sampling_probability(sampling_centroid, gripper_pose_ref_PW, gripper_pose_PW, pc, bin_mask,grasp_quality)
        # selection_p = torch.rand_like(gripper_pose_PW[:, 0])
        selection_p = gamma_dive**(1/2)

        avaliable_iterations = selection_mask.sum()
        if avaliable_iterations<3: return False, None,None,None

        n = int(min(max_n, avaliable_iterations))

        counter = 0

        sampler_samples=0

        t = 0
        while t < n:
            t += 1

            dist = MaskedCategorical(probs=selection_p, mask=selection_mask)

            target_index = dist.sample().item()

            selection_mask[target_index] *= False
            avaliable_iterations -= 1
            target_point = pc[target_index]

            target_generated_pose = gripper_pose_PW[target_index]
            target_ref_pose = gripper_pose_ref_PW[target_index]

            margin=1.


            ref_success ,ref_initial_collision= self.evaluate_grasp(target_point,target_ref_pose,view=False)
            gen_success,gen_initial_collision = self.evaluate_grasp(target_point, target_generated_pose,view=False)
            if ref_success and gen_success :
                print('                                                         ...1')
                ref_success,  ref_initial_collision = self.evaluate_grasp(target_point, target_ref_pose,
                                                                                       view=False, shake_intensity=0.05)
                gen_success,  gen_initial_collision = self.evaluate_grasp(target_point,
                                                                                       target_generated_pose,
                                                                                       view=False, shake_intensity=0.05)

            if ref_success and not gen_success:
                superior_A_model_moving_rate.update(0.)
            elif gen_success and not ref_success:
                superior_A_model_moving_rate.update(1.)

            if ref_success and gen_success:
                print('                                                         ...2')
                max_ref=target_ref_pose[4:-1].max().item()
                max_gen=target_generated_pose[4:-1].max().item()
                if abs(max_ref-max_gen)<0.1: continue
                print('                                                         ...3')

                ref_success=max_ref<max_gen
                gen_success=not ref_success
                margin=abs(max_ref-max_gen)/(max_ref+max_gen)

            if ref_success == gen_success:continue


            if not ref_success :
                if np.random.rand() > ((1 - torch.clamp(grasp_quality[target_index], 0, 1).item()) * annealing_factor):
                    print('                                                         ...5')
                    continue


            print(f'ref_success={ref_success}, gen_success={gen_success}')
            # print(f'ref_in_scope={ref_in_scope}, gen_in_scope={gen_in_scope}')

            k=1 if ref_success and not gen_success else -1
            if k == 1:
                sampler_samples+=1

            counter += 1
            t = 0
            hh = (counter / batch_size) ** 2
            n = int(min(hh * max_n + n, avaliable_iterations))

            self.relative_sampling_timing.update(t)

            pairs.append((target_index,  k,margin))

            self.tmp_pose_record.append(target_generated_pose.detach().clone())

            superior_pose = target_ref_pose if k > 0 else target_generated_pose

            quat=superior_pose[0:4].clone()

            self.quat_centers.update(quat)
            superior_pose[4:4 + 3]=torch.clip(superior_pose[4:4+3],0,1)
            fingers=superior_pose[4:4+3].clone()
            fingers=torch.clip(fingers,0,1)
            self.fingers_centers.update(fingers)


            if sampling_centroid is None:
                sampling_centroid = superior_pose.detach().clone()
            else:
                sampling_centroid = sampling_centroid * 0.999 + superior_pose.detach().clone() * 0.001

            if counter == batch_size: break


        if counter == batch_size:
            return True, pairs, sampling_centroid,sampler_samples
        else:
            return False, pairs, sampling_centroid,sampler_samples

    def step(self,i):
        self.ch_env.drop_new_obj()

        '''get scene perception'''
        depth, pc, floor_mask = self.ch_env.get_scene_preception(view=False)
        # return

        depth = torch.from_numpy(depth).cuda()  # [600.600]
        floor_mask = torch.from_numpy(floor_mask).cuda()

        for k in range(iter_per_scene):

            with torch.no_grad():

                gripper_pose, grasp_quality, grasp_collision = self.gan.generator(
                    depth[None, None, ...],~floor_mask.view(1,1,600,600),detach_backbone=True)

                # gripper_pose = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000, 10)
                # print(torch.abs(gripper_pose[~floor_mask]).mean(dim=0))
                # print(torch.abs(gripper_pose[floor_mask]).mean(dim=0))
                # exit()

                f = self.grasp_quality_statistics.accuracy * ((1 - grasp_quality.detach()) ** 2) + (
                            1 - self.grasp_quality_statistics.accuracy)
                annealing_factor = self.tou * f
                print(f'mean_annealing_factor= {annealing_factor.mean()}, tou={self.tou}')

                gripper_pose_ref = ch_pose_interpolation(gripper_pose, self.sampling_centroid,
                                                         annealing_factor=annealing_factor,quat_centers=self.quat_centers.centers,finger_centers=self.fingers_centers.centers)  # [b,10,600,600]

                if i % int(100) == 0 and i != 0 and k == 0:
                    try:
                        self.export_check_points()
                        self.save_statistics()
                        # self.load_action_model()
                    except Exception as e:
                        print(Fore.RED, str(e), Fore.RESET)
                if i % 10 == 0 and k == 0:
                    self.view_result(gripper_pose, floor_mask)

                self.tmp_pose_record = []
                counted, pairs, sampling_centroid ,sampler_samples= self.sample_contrastive_pairs(pc, floor_mask, gripper_pose,
                                                                                  gripper_pose_ref,
                                                                                  self.sampling_centroid, batch_size,
                                                                                  self.tou, grasp_quality.detach(),
                                                                                  self.superior_A_model_moving_rate)
                if not counted:
                    self.ch_env.update_obj_info(0.1)
                    continue
                else:
                    self.ch_env.update_obj_info(0.9)

                x = self.superior_A_model_moving_rate.val
                self.tou = 1 - x

                self.sampling_centroid = sampling_centroid


            gripper_pose = gripper_pose[0].permute(1, 2, 0).reshape(360000, 10)
            gripper_pose_ref = gripper_pose_ref[0].permute(1, 2, 0).reshape(360000, 10)

            self.step_discriminator(depth, gripper_pose, gripper_pose_ref, pairs,floor_mask)

            # if sampler_samples==batch_size:
            self.step_generator(depth, floor_mask, pc, gripper_pose_ref, pairs)
            # else:
            #     self.step_generator_without_sampler(depth, floor_mask, pc)
            # continue


    def view_result(self, gripper_poses,floor_mask):
        with torch.no_grad():

            print('Center pos: ',self.sampling_centroid.cpu().numpy())


            cuda_memory_report()

            values = gripper_poses[0].permute(1, 2, 0).reshape(360000, 10).detach()  # .cpu().numpy()
            values=values[~floor_mask]

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
            self.relative_sampling_timing.view()
            self.superior_A_model_moving_rate.view()

            self.bin_collision_statistics.print()
            self.objects_collision_statistics.print()
            self.grasp_quality_statistics.print()



            self.quat_centers.view()
            self.fingers_centers.view()

    def save_statistics(self):
        self.moving_collision_rate.save()
        self.relative_sampling_timing.save()
        self.superior_A_model_moving_rate.save()

        self.critic_statistics.save()
        self.gripper_sampler_statistics.save()

        self.data_tracker.save()


        self.bin_collision_statistics.save()
        self.objects_collision_statistics.save()
        self.grasp_quality_statistics.save()

        torch.save(self.sampling_centroid,self.last_pose_center_path)

        self.quat_centers.save()
        self.fingers_centers.save()

        self.ch_env.save_obj_dict()

    def export_check_points(self):
        self.gan.export_models()
        self.gan.export_optimizers()

    def clear(self):
        self.critic_statistics.clear()

        self.bin_collision_statistics.clear()
        self.objects_collision_statistics.clear()
        self.gripper_sampler_statistics.clear()
        self.grasp_quality_statistics.clear()

    def begin(self,iterations=10):
        pi = progress_indicator('Begin new training round: ', max_limit=iterations)

        for i in range(iterations):
            # cuda_memory_report()
            try:
                self.step(i)
                pi.step(i)
            except Exception as e:
                print(Fore.RED, str(e), Fore.RESET)
                torch.cuda.empty_cache()
                self.ch_env.update_obj_info(0.1)

        pi.end()

        self.export_check_points()
        self.save_statistics()
        self.clear()

def train_N_grasp_GAN(n=1):
    lr = 1e-5
    Train_grasp_GAN = TrainGraspGAN(n_samples=None, learning_rate=lr)
    torch.cuda.empty_cache()

    for i in range(n):
        cuda_memory_report()
        Train_grasp_GAN.initialize(n_samples=None)
        # fix_weight_scales(Train_grasp_GAN.gan.generator.grasp_collision_)
        # exit()
        # scale_all_weights(Train_grasp_GAN.gan.generator.back_bone_,5)
        # Train_grasp_GAN.export_check_points()
        # exit()
        Train_grasp_GAN.begin(iterations=100)


if __name__ == "__main__":
    train_N_grasp_GAN(n=10000)
