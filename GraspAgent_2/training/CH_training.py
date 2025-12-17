import argparse
import configparser
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
from GraspAgent_2.utils.focal_loss import FocalLoss
from GraspAgent_2.utils.model_init import init_weights_he_normal, gan_init_with_norms
from GraspAgent_2.utils.quat_operations import quat_rotate_vector, grasp_frame_to_quat, quat_between

from Online_data_audit.data_tracker import gripper_grasp_tracker, DataTracker
from check_points.check_point_conventions import GANWrapper
from lib.IO_utils import custom_print
from lib.cuda_utils import cuda_memory_report
from lib.loss.D_loss import binary_l1, binary_smooth_l1
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from records.training_satatistics import TrainingTracker, MovingRate
import torch


freeze_G_backbone = False
freeze_D_backbone = False



hard_level_factor=0

max_n = 30

bce_loss = nn.BCELoss()
bce_with_logits=nn.BCEWithLogitsLoss()
focal_loss=FocalLoss()

print = custom_print

cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
cache_name = 'normals'
discrepancy_distance = 1.0

collision_expo = 1.0
firmness_expo = 1.0
generator_expo = 1.0

m = 0.2

def cosine_triplet_loss(anchor, positive, negative, margin_signal):

    margin_signal=torch.tensor([margin_signal],device=anchor.device)
    d_pos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    d_neg = 1 - F.cosine_similarity(anchor, negative, dim=-1)
    loss = F.relu(d_pos - d_neg + margin_signal*0.3)
    return loss.mean()

def balanced_sampling(values, mask=None, exponent=2.0, balance_indicator=1.0,eps=1e-4):
    with torch.no_grad():
        max_ = values.max().item()
        min_ = values.min().item()
        range_ = max_ - min_

        if not range_ > 0.:
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


        return target_index

def quat_from_dir_and_roll_torch(alpha, beta, eps=1e-8, canonicalize=True):
    """
    alpha : (N, 3)  main direction vectors
    beta  : (N, 2)  roll vectors (XY plane)
    return: (N, 4)  quaternions (w, x, y, z)
    """

    def normalize(v):
        return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

    # z-axis
    z = normalize(alpha)

    # lift 2D roll → 3D
    x_ref = torch.cat([beta, torch.zeros_like(beta[..., :1])], dim=-1)
    x_ref = normalize(x_ref)

    # Gram–Schmidt to make x ⟂ z
    x = x_ref - (x_ref * z).sum(-1, keepdim=True) * z
    x = normalize(x)

    # y-axis
    y = torch.cross(z, x, dim=-1)

    # rotation matrix (columns = x y z)
    R = torch.stack([x, y, z], dim=-1)  # (N,3,3)

    # rotmat → quaternion (vectorized, stable)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    qw = torch.sqrt((trace + 1.0).clamp_min(0)) / 2
    qx = (R[..., 2, 1] - R[..., 1, 2]) / (4 * qw.clamp_min(eps))
    qy = (R[..., 0, 2] - R[..., 2, 0]) / (4 * qw.clamp_min(eps))
    qz = (R[..., 1, 0] - R[..., 0, 1]) / (4 * qw.clamp_min(eps))

    q = torch.stack([qw, qx, qy, qz], dim=-1)
    q = normalize(q)

    # canonicalize: enforce w ≥ 0 (important for learning)
    if canonicalize:
        sign = torch.where(q[..., :1] < 0, -1.0, 1.0)
        q = q * sign

    return q

def process_pose(target_point, target_pose, view=False):
    target_pose_ = target_pose.clone()
    target_point_ = np.copy(target_point)

    # quat = target_pose_[:4].cpu().tolist()
    alpha=target_pose_[:3]
    beta=target_pose_[3:5]

    approach_ref=torch.tensor([0.866, -0.5, 0],device='cuda')

    default_quat = quat_between(approach_ref, torch.tensor([0., 0., -1.],device='cuda'))
    quat=grasp_frame_to_quat(alpha, beta, default_quat).cpu().tolist()

    fingers = torch.clip(target_pose_[5:5+3],0,1).cpu().tolist()

    transition = target_pose_[5+3:5+3+1].cpu().numpy() / 100
    projected_transition = quat_rotate_vector(quat, approach_ref.tolist())*transition[0]

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


class TrainGraspGAN:
    def __init__(self, args,n_samples=None, epochs=1, learning_rate=5e-5):

        self.args=args

        self.batch_size=1
        self.max_G_norm = 500
        self.max_D_norm = 100
        self.iter_per_scene = 1

        self.n_samples = n_samples
        self.size = n_samples
        self.epochs = epochs

        '''model wrapper'''
        self.gan = self.prepare_model_wrapper()

        '''Moving rates'''
        self.moving_collision_rate = None
        self.skip_rate = None
        self.superior_A_model_moving_rate = None

        self.G_grad_norm_MR=None
        self.D_grad_norm_MR=None

        self.bin_collision_statistics = None
        self.objects_collision_statistics = None
        self.collision_statistics = None

        self.grasp_quality_statistics=None

        '''initialize statistics records'''
        self.gripper_sampler_statistics = None
        self.critic_statistics = None
        self.critic_statistics = None
        self.data_tracker = None

        self.tmp_pose_record=[]

        self.n_param = 9

        self.last_pose_center_path=CH_model_key+'_pose_center'
        if os.path.exists(self.last_pose_center_path):
            self.sampling_centroid = torch.load(self.last_pose_center_path).cuda()
            if self.sampling_centroid.shape!=self.n_param: self.sampling_centroid = torch.tensor([0, 1,0, 1, 0, 0.5,0.5,0.5,  0.],
                                                        device='cuda')
        else: self.sampling_centroid = torch.tensor([0, 1,0, 1, 0, 0.5,0.5,0.5,  0.],
                                                        device='cuda')
        root_dir = os.getcwd()  # current working directory


        self.ch_env = CasiaHandEnv(root=root_dir + "/GraspAgent_2/sim_hand_s/speed_hand/",max_obj_per_scene=10)

        self.tou = 1

        # self.quat_centers=OnlingClustering(key_name=CH_model_key+'_quat',number_of_centers=20,vector_size=4,decay_rate=0.01,is_quat=True,dist_threshold=0.77)
        # self.fingers_centers=OnlingClustering(key_name=CH_model_key+'_fingers',number_of_centers=10,vector_size=3,decay_rate=0.01,use_euclidean_dist=True,dist_threshold=0.2)
        # self.transition_centers=OnlingClustering(key_name=CH_model_key+'_transitions',number_of_centers=5,vector_size=1,decay_rate=0.01,use_euclidean_dist=True,dist_threshold=0.2)
        inti_centers=torch.tensor([[0, 1,0, 1, 0, 0.5,0.5,0.5,  0.]],device='cuda')
        self.taxonomies=OnlingClustering(key_name=CH_model_key+'_taxonomies',number_of_centers=30,vector_size=9,decay_rate=0.01,use_euclidean_dist=True,dist_threshold=0.2,inti_centers=inti_centers)

    def initialize(self, n_samples=None):
        self.n_samples = n_samples

        '''Moving rates'''
        self.moving_collision_rate = MovingRate(CH_model_key + '_collision', decay_rate=0.01,
                                                initial_val=1.)
        self.skip_rate = MovingRate(CH_model_key + '_skip_rate',
                                                   decay_rate=0.01,
                                                    initial_val=1.)
        self.superior_A_model_moving_rate = MovingRate(CH_model_key + '_superior_A_model',
                                                       decay_rate=0.01,
                                                       initial_val=0.)

        self.G_grad_norm_MR = MovingRate(CH_model_key + '_G_grad_norm',
                                                       decay_rate=0.01,
                                                       initial_val=0.)

        self.D_grad_norm_MR = MovingRate(CH_model_key + '_D_grad_norm',
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
        self.collision_statistics= TrainingTracker(name=CH_model_key + '_collision',
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

        # gan.generator.back_bone.apply(init_weights_he_normal)
        # gan_init_with_norms(gan.generator.CH_PoseSampler)

        gan.critic_adamW_optimizer(learning_rate=self.args.lr, beta1=0., beta2=0.999,weight_decay_=0,load_check_point=self.args.load_last_optimizer)
        # gan.critic_sgd_optimizer(learning_rate=self.learning_rate*10,momentum=0.,weight_decay_=0.)
        gan.generator_adamW_optimizer(learning_rate=self.args.lr, beta1=0.9, beta2=0.999,load_check_point=self.args.load_last_optimizer)
        # gan.generator_sgd_optimizer(learning_rate=self.learning_rate*10,momentum=0.)
        return gan

    def step_discriminator(self,depth,   gripper_pose, gripper_pose_ref ,pairs,floor_mask,latent_vector ):
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
        anchor,positive_negative = self.gan.critic(depth[None,None,...], generated_grasps_stack,pairs,~floor_mask.view(1,1,600,600),latent_vector=latent_vector,detach_backbone=freeze_D_backbone)

        # print(score)
        # exit()
        # gen_scores_ = score.permute(0, 2, 3, 1)[0, :, :, 0].reshape(-1)
        # ref_scores_ = score.permute(0, 2, 3, 1)[1, :, :, 0].reshape(-1)

        generated_embedding = positive_negative[:, 0]
        ref_embedding = positive_negative[:, 1]

        loss = torch.tensor(0., device=depth.device)

        for j in range(len(pairs)):
            target_index = pairs[j][0]
            k = pairs[j][1]
            margin=pairs[j][2]

            if k>0:
                positive=ref_embedding[j]
                negative=generated_embedding[j]
            else:
                positive = generated_embedding[j]
                negative = ref_embedding[j]



            loss+=(cosine_triplet_loss(anchor, positive, negative, margin_signal=margin)**1)/self.batch_size
            # c = 1
            # if k > 0 and (label - pred_) > 1:
            #     print(Fore.LIGHTMAGENTA_EX, 'curriculum loss activated', Fore.RESET)
            #     c = -1
            #     margin = 0

            # print(f'label score: {label.item()}, pred score: {pred_.item()}')

            # loss+=(torch.clamp((pred_ - label) * k *c+ margin, 0.)**2  )/self.batch_size
            # if k>0 and margin==1:
            #     loss+=(torch.abs(pred_+1)**2)/self.batch_size
            #     loss+=(torch.abs(label-1)**2)/self.batch_size
            # elif k<0 and margin==1:
            #     loss += (torch.abs(label + 1) ** 2) / self.batch_size
            #     loss += (torch.abs(pred_ - 1) ** 2) / self.batch_size

            # loss+=l/batch_size
        # loss=self.RGAN_D_loss(pairs,gen_scores_,ref_scores_)
        loss.backward()

        self.critic_statistics.loss=loss.item()

        '''GRADIENT CLIPPING'''
        params = list(self.gan.critic.back_bone.parameters())
        backbone_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf'))

        params = list(self.gan.critic.att_block_.parameters())
        decoder_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf'))

        norm=torch.nn.utils.clip_grad_norm_(self.gan.critic.parameters(), max_norm=self.max_D_norm)

        self.D_grad_norm_MR.update(norm.item())

        print(Fore.LIGHTGREEN_EX,f' D  norm : {norm}, backbone norm : {backbone_norm}, decoder norm : {decoder_norm}',Fore.RESET)

        self.gan.critic_optimizer.step()

        self.gan.critic.zero_grad()
        self.gan.critic_optimizer.zero_grad()

        print(  Fore.LIGHTYELLOW_EX,f'd_loss={loss.item()}',
              Fore.RESET)

    def get_generator_loss(self, depth, gripper_pose, gripper_pose_ref, pairs,floor_mask,latent_vector):

        gripper_pose = gripper_pose[0].permute(1, 2, 0).reshape(360000, self.n_param)

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
        anchor,positive_negative  = self.gan.critic(depth[None,None,...], generated_grasps_stack,pairs,~floor_mask.view(1,1,600,600),latent_vector=latent_vector,detach_backbone=True)

        # cuda_memory_report()
        # critic_score = self.gan.critic(pc, generated_grasps_cat, detach_backbone=True)

        # gen_scores_ = critic_score.permute(0, 2, 3, 1)[0, :, :, 0].reshape(-1)
        # ref_scores_ = critic_score.permute(0, 2, 3, 1)[1, :, :, 0].reshape(-1)

        # gripper_pose = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(-1)
        # gripper_pose_ref = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :].reshape(-1)

        gen_embedding = positive_negative[:, 0]
        ref_embedding = positive_negative[:, 1]

        loss = torch.tensor(0., device=depth.device)

        for j in range(len(pairs)):
            target_index = pairs[j][0]
            k = pairs[j][1]
            margin=pairs[j][2]

            target_generated_pose = gripper_pose[target_index].detach()
            target_ref_pose = gripper_pose_ref[target_index].detach()

            loss+=(cosine_triplet_loss(anchor, positive=gen_embedding[j], negative=ref_embedding[j], margin_signal=0.))/self.batch_size


            # label = ref_scores_[j].squeeze()
            # pred_ = gen_scores_[j].squeeze()

            if margin!=1:
                print(Fore.LIGHTYELLOW_EX,f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()} , m={margin}] ',Fore.RESET)
            elif k==1:
                print(Fore.LIGHTCYAN_EX,f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()}  ',Fore.RESET)
            else:
                print(Fore.LIGHTGREEN_EX,f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()} ',Fore.RESET)

            # v2 = quat_rotate_vector(target_ref_pose[0:4].cpu().tolist(), [0, 1, 0])
            # v3 = quat_rotate_vector(target_generated_pose[0:4].cpu().tolist(), [0, 1, 0])

            # print('ref approach: ',v2,' ge approach: ',v3)
            # print()

            # w=1 if k>0 else 0
            # loss += ((torch.clamp( label - pred_, 0.)) **1)/ self.batch_size
            # loss += (torch.abs(1-pred_) **1)/ self.batch_size


        return loss

    def step_generator(self,depth,floor_mask,pc,gripper_pose_ref,pairs,latent_vector):
        '''zero grad'''
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator.zero_grad(set_to_none=True)
        # self.gan.critic.eval()

        '''generated grasps'''
        # cuda_memory_report()
        gripper_pose, grasp_quality_logits,  grasp_collision_logits = self.gan.generator(depth[None, None, ...],~floor_mask.view(1,1,600,600),latent_vector,
                                                                                                detach_backbone=freeze_G_backbone,backbone=self.gan.critic.back_bone)
        # cuda_memory_report()
        # exit()
        gripper_pose_PW = gripper_pose[0].permute(1, 2, 0).reshape(360000,self.n_param)
        object_collision_logits = grasp_collision_logits[0,0].reshape(-1)
        floor_collision_logits = grasp_collision_logits[0,1].reshape(-1)
        collision_logits = grasp_collision_logits[0,2].reshape(-1)

        grasp_quality_logits = grasp_quality_logits[0,0].reshape(-1)

        '''loss computation'''
        gripper_collision_loss = torch.tensor(0., device=depth.device)
        gripper_quality_loss_ = torch.tensor(0., device=depth.device)

        for k in range(self.batch_size*4):
            '''gripper collision'''
            gripper_target_index = balanced_sampling(F.sigmoid(collision_logits.detach()),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.collision_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = collision_logits[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

            contact_with_obj , contact_with_floor=self.check_collision(gripper_target_point,gripper_target_pose,view=False)

            label = torch.ones_like(gripper_prediction_) if contact_with_obj or contact_with_floor else torch.zeros_like(gripper_prediction_)

            object_collision_loss = bce_with_logits(gripper_prediction_, label)

            with torch.no_grad():
                self.collision_statistics.loss = object_collision_loss.item()
                self.collision_statistics.update_confession_matrix(label.detach(),
                                                                      F.sigmoid(gripper_prediction_.detach()))

            gripper_collision_loss+=object_collision_loss/self.batch_size



        for k in range(self.batch_size*4):
            '''gripper-object collision'''
            gripper_target_index = balanced_sampling(F.sigmoid(object_collision_logits.detach()),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.objects_collision_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = object_collision_logits[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

            contact_with_obj , contact_with_floor=self.check_collision(gripper_target_point,gripper_target_pose,view=False)

            label = torch.ones_like(gripper_prediction_) if contact_with_obj else torch.zeros_like(gripper_prediction_)

            object_collision_loss = bce_with_logits(gripper_prediction_, label)

            with torch.no_grad():
                self.objects_collision_statistics.loss = object_collision_loss.item()
                self.objects_collision_statistics.update_confession_matrix(label.detach(),
                                                                      F.sigmoid(gripper_prediction_.detach()))

            gripper_collision_loss+=object_collision_loss/self.batch_size


        for k in range(self.batch_size*4):
            '''gripper-bin collision'''
            gripper_target_index = balanced_sampling(F.sigmoid(floor_collision_logits.detach()),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.bin_collision_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = floor_collision_logits[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

            contact_with_obj , contact_with_floor = self.check_collision(gripper_target_point, gripper_target_pose, view=False)

            label = torch.ones_like(gripper_prediction_) if contact_with_floor else torch.zeros_like(gripper_prediction_)

            floor_collision_loss = bce_with_logits(gripper_prediction_, label)

            with torch.no_grad():
                self.bin_collision_statistics.loss = floor_collision_loss.item()
                self.bin_collision_statistics.update_confession_matrix(label.detach(),
                                                                           F.sigmoid(gripper_prediction_.detach()))

            gripper_collision_loss += floor_collision_loss / self.batch_size

        for k in range(self.batch_size*4):
            '''grasp quality'''
            gripper_target_index = balanced_sampling(F.sigmoid(grasp_quality_logits.detach()),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.grasp_quality_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = grasp_quality_logits[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

            grasp_success,initial_collision,n_grasp_contact,self_collide,stable_grasp  = self.evaluate_grasp(gripper_target_point, gripper_target_pose, view=False,hard_level=(1-self.tou)*hard_level_factor,shake=False)

            label = torch.ones_like(gripper_prediction_) if grasp_success else torch.zeros_like(gripper_prediction_)

            grasp_quality_loss = bce_with_logits(gripper_prediction_, label)

            with torch.no_grad():
                self.grasp_quality_statistics.loss = grasp_quality_loss.item()
                self.grasp_quality_statistics.update_confession_matrix(label.detach(),
                                                                       F.sigmoid(gripper_prediction_.detach()))

            gripper_quality_loss_ += grasp_quality_loss / self.batch_size

        gripper_sampling_loss = self.get_generator_loss(
            depth,  gripper_pose, gripper_pose_ref,
            pairs,floor_mask,latent_vector)

        assert not torch.isnan(gripper_sampling_loss).any(), f'{gripper_sampling_loss}'

        print(Fore.LIGHTYELLOW_EX,
              f'g_loss={gripper_sampling_loss.item()}',
              Fore.RESET)

        loss = gripper_sampling_loss+gripper_collision_loss+gripper_quality_loss_

        with torch.no_grad():
            self.gripper_sampler_statistics.loss = gripper_sampling_loss.item()
        # if abs(loss.item())>0.0:
        # try:
        loss.backward()

        '''GRADIENT CLIPPING'''
        params=list(self.gan.generator.back_bone.parameters()) + \
         list(self.gan.generator.CH_PoseSampler.parameters())
        norm=torch.nn.utils.clip_grad_norm_(params, max_norm=self.max_G_norm)
        self.G_grad_norm_MR.update(norm.item())
        print(Fore.LIGHTGREEN_EX,f' G norm : {norm}',Fore.RESET)

        params2 = list(self.gan.generator.back_bone2_.parameters()) + \
                 list(self.gan.generator.grasp_quality.parameters())+ \
                 list(self.gan.generator.grasp_collision.parameters())+ \
                 list(self.gan.generator.grasp_collision2_.parameters())+ \
                 list(self.gan.generator.grasp_collision3_.parameters())
        norm = torch.nn.utils.clip_grad_norm_(params2, max_norm=float('inf'))
        print(Fore.LIGHTGREEN_EX, f' G norm 2 : {norm}', Fore.RESET)

        self.gan.generator_optimizer.step()

        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator_optimizer.zero_grad(set_to_none=True)
        self.gan.critic_optimizer.zero_grad(set_to_none=True)

    def step_generator_without_sampler(self,depth,floor_mask,pc,latent_vector):
        '''zero grad'''
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator.zero_grad(set_to_none=True)
        # self.gan.critic.eval()

        '''generated grasps'''
        # cuda_memory_report()
        gripper_pose, grasp_quality_logits,  grasp_collision_logits = self.gan.generator(depth[None, None, ...],~floor_mask.view(1,1,600,600),latent_vector,
                                                                                                detach_backbone=freeze_G_backbone,backbone=self.gan.critic.back_bone)
        # cuda_memory_report()
        # exit()
        gripper_pose_PW = gripper_pose[0].permute(1, 2, 0).reshape(360000,self.n_param)
        object_collision_logits = grasp_collision_logits[0,0].reshape(-1)
        floor_collision_logits = grasp_collision_logits[0,1].reshape(-1)
        collision_logits = grasp_collision_logits[0,2].reshape(-1)

        grasp_quality_logits = grasp_quality_logits[0,0].reshape(-1)

        '''loss computation'''
        gripper_collision_loss = torch.tensor(0., device=depth.device)
        gripper_quality_loss_ = torch.tensor(0., device=depth.device)

        for k in range(self.batch_size*2):
            '''gripper collision'''
            gripper_target_index = balanced_sampling(F.sigmoid(collision_logits.detach()),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.collision_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = collision_logits[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()


            contact_with_obj , contact_with_floor=self.check_collision(gripper_target_point,gripper_target_pose,view=False)

            label = torch.ones_like(gripper_prediction_) if contact_with_obj or contact_with_floor else torch.zeros_like(gripper_prediction_)

            object_collision_loss = bce_with_logits(gripper_prediction_, label)

            with torch.no_grad():
                self.collision_statistics.loss = object_collision_loss.item()
                self.collision_statistics.update_confession_matrix(label.detach(),
                                                                      F.sigmoid(gripper_prediction_.detach()))

            gripper_collision_loss+=object_collision_loss/self.batch_size



        for k in range(self.batch_size*2):
            '''gripper-object collision'''
            gripper_target_index = balanced_sampling(F.sigmoid(object_collision_logits.detach()),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.objects_collision_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = object_collision_logits[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

            contact_with_obj , contact_with_floor=self.check_collision(gripper_target_point,gripper_target_pose,view=False)

            label = torch.ones_like(gripper_prediction_) if contact_with_obj else torch.zeros_like(gripper_prediction_)

            object_collision_loss = bce_with_logits(gripper_prediction_, label)

            with torch.no_grad():
                self.objects_collision_statistics.loss = object_collision_loss.item()
                self.objects_collision_statistics.update_confession_matrix(label.detach(),
                                                                      F.sigmoid(gripper_prediction_.detach()))

            gripper_collision_loss+=object_collision_loss/self.batch_size


        for k in range(self.batch_size*2):
            '''gripper-bin collision'''
            gripper_target_index = balanced_sampling(F.sigmoid(floor_collision_logits.detach()),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.bin_collision_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = floor_collision_logits[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

            contact_with_obj , contact_with_floor = self.check_collision(gripper_target_point, gripper_target_pose, view=False)

            label = torch.ones_like(gripper_prediction_) if contact_with_floor else torch.zeros_like(gripper_prediction_)

            floor_collision_loss = bce_with_logits(gripper_prediction_, label)

            with torch.no_grad():
                self.bin_collision_statistics.loss = floor_collision_loss.item()
                self.bin_collision_statistics.update_confession_matrix(label.detach(),
                                                                           F.sigmoid(gripper_prediction_.detach()))

            gripper_collision_loss += floor_collision_loss / self.batch_size

        for k in range(self.batch_size*2):
            '''grasp quality'''
            gripper_target_index = balanced_sampling(F.sigmoid(grasp_quality_logits.detach()),
                                                     mask=~floor_mask.detach(),
                                                     exponent=30.0,
                                                     balance_indicator=self.grasp_quality_statistics.label_balance_indicator)
            gripper_target_point = pc[gripper_target_index]
            gripper_prediction_ = grasp_quality_logits[gripper_target_index].squeeze()
            gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

            grasp_success,initial_collision,n_grasp_contact,self_collide,stable_grasp  = self.evaluate_grasp(gripper_target_point, gripper_target_pose, view=False,hard_level=(1-self.tou)*hard_level_factor,shake=False)

            label = torch.ones_like(gripper_prediction_) if grasp_success else torch.zeros_like(gripper_prediction_)

            grasp_quality_loss = bce_with_logits(gripper_prediction_, label)

            with torch.no_grad():
                self.grasp_quality_statistics.loss = grasp_quality_loss.item()
                self.grasp_quality_statistics.update_confession_matrix(label.detach(),
                                                                       F.sigmoid(gripper_prediction_.detach()))

            gripper_quality_loss_ += grasp_quality_loss / self.batch_size



        loss = gripper_collision_loss+gripper_quality_loss_

        print(Fore.LIGHTYELLOW_EX,
              f'g_loss without sampler={loss.item()}',
              Fore.RESET)

        loss.backward()

        '''GRADIENT CLIPPING'''
        params=list(self.gan.generator.back_bone.parameters()) + \
         list(self.gan.generator.CH_PoseSampler.parameters())
        norm=torch.nn.utils.clip_grad_norm_(params, max_norm=self.max_G_norm)
        self.G_grad_norm_MR.update(norm.item())
        print(Fore.LIGHTGREEN_EX,f' G norm : {norm}',Fore.RESET)

        params2 = list(self.gan.generator.back_bone2_.parameters()) + \
                 list(self.gan.generator.grasp_quality.parameters())+ \
                 list(self.gan.generator.grasp_collision.parameters())+ \
                 list(self.gan.generator.grasp_collision2_.parameters())+ \
                 list(self.gan.generator.grasp_collision3_.parameters())
        norm = torch.nn.utils.clip_grad_norm_(params2, max_norm=float('inf'))
        print(Fore.LIGHTGREEN_EX, f' G norm 2 : {norm}', Fore.RESET)

        self.gan.generator_optimizer.step()

        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator_optimizer.zero_grad(set_to_none=True)
        self.gan.critic_optimizer.zero_grad(set_to_none=True)


    def check_collision(self,target_point,target_pose,view=False):
        with torch.no_grad():
            quat, fingers, shifted_point = process_pose(target_point, target_pose, view=view)

        return self.ch_env.check_collision(hand_pos=shifted_point,hand_quat=quat,hand_fingers=None,view=False)

    def evaluate_grasp(self, target_point, target_pose, view=False,hard_level=0,shake=True):

        with torch.no_grad():
            quat,fingers,shifted_point= process_pose(target_point, target_pose, view=view)

            in_scope, grasp_success, contact_with_obj, contact_with_floor,n_grasp_contact,self_collide,stable_grasp  = self.ch_env.check_graspness(
                hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers,
               view=view,hard_level=hard_level,shake=shake)

            initial_collision=contact_with_obj or contact_with_floor

            # print('in_scope, grasp_success, contact_with_obj, contact_with_floor :',in_scope, grasp_success, contact_with_obj, contact_with_floor )

            if grasp_success is not None:
                if grasp_success and not contact_with_obj and not contact_with_floor:
                    return True ,initial_collision,n_grasp_contact,self_collide,stable_grasp

        return False, initial_collision,n_grasp_contact,self_collide,stable_grasp


    def sample_contrastive_pairs(self,pc,  floor_mask, gripper_pose, gripper_pose_ref,
                                 sampling_centroid,  annealing_factor, grasp_quality,grasp_collision,
                                 superior_A_model_moving_rate,latent_vector):

        pairs = []

        selection_mask = (~floor_mask) #& (latent_vector.reshape(-1)!=0)
        grasp_quality=grasp_quality[0,0].reshape(-1)
        grasp_collision=grasp_collision[0,2].reshape(-1)
        gripper_pose_PW = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000,self.n_param)
        clipped_gripper_pose_PW=gripper_pose_PW.clone()
        clipped_gripper_pose_PW[:,5:5+3]=torch.clip(clipped_gripper_pose_PW[:,5:5+3],0,1)
        gripper_pose_ref_PW = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000,self.n_param)

        # grasp_quality = grasp_quality.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        # max_ = grasp_quality.max()
        # min_ = grasp_quality.min()
        # grasp_quality = (grasp_quality - min_) / (max_ - min_)
        def norm_(gamma ,expo_=1.0,min=0.01):

            gamma = (gamma - gamma.min()) / (
                    gamma.max() - gamma.min()+1e-6)

            gamma = gamma ** expo_

            gamma=torch.clamp(gamma,min)

            return gamma
        gamma_dive = norm_((1.001 - F.cosine_similarity(clipped_gripper_pose_PW,
                                                        sampling_centroid[None, :], dim=-1) ) /2 ,1)

        gamma_dive *= norm_((1.001 - F.cosine_similarity(gripper_pose_ref_PW,
                                                        sampling_centroid[None, :], dim=-1) ) /2 ,1)

        # selection_p = compute_sampling_probability(sampling_centroid, gripper_pose_ref_PW, gripper_pose_PW, pc, bin_mask,grasp_quality)
        # selection_p = torch.rand_like(gripper_pose_PW[:, 0])
        # gamma_col=norm_(1-grasp_collision)*self.skip_rate()+(1-self.skip_rate())
        gamma_dive=gamma_dive*self.skip_rate()+(1-self.skip_rate())
        selection_p = (gamma_dive*torch.rand_like(gamma_dive)) ** (1/4) #* self.tou + (1 - self.tou)*torch.rand_like(gamma_dive)

        avaliable_iterations = selection_mask.sum()
        if avaliable_iterations<3: return False, None,None,None

        n = int(min(max_n, avaliable_iterations))


        print(Fore.LIGHTBLACK_EX,'         # Available candidates =',avaliable_iterations.item(),Fore.RESET)

        counter = 0
        counter2=0

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

            # assert target_ref_pose[0]>0,f'{target_ref_pose}'

            ref_success ,ref_initial_collision,ref_n_grasp_contact,ref_self_collide,stable_ref_grasp = self.evaluate_grasp(target_point,target_ref_pose,view=False,hard_level=(1-self.tou)*hard_level_factor)
            gen_success,gen_initial_collision,gen_n_grasp_contact,gen_self_collide,stable_gen_grasp  = self.evaluate_grasp(target_point, target_generated_pose,view=False,hard_level=(1-self.tou)*hard_level_factor)
            gen_success=gen_success and target_generated_pose[0].item()>0
            if t==1 and self.skip_rate()>0.9:
                print(f' ref ---- {target_ref_pose}, {ref_success ,ref_initial_collision,ref_n_grasp_contact,ref_self_collide}')
                print(f' gen ---- {target_generated_pose}, {ref_success ,ref_initial_collision,ref_n_grasp_contact,ref_self_collide}')

            if not ref_success and not gen_success:
                    continue
            print(f'ref_success={ref_success}, gen_success={gen_success}')

            # if self.skip_rate.val > 0.7:
            if ref_success and gen_success:
                ref_success=ref_success and stable_ref_grasp
                gen_success=gen_success and stable_gen_grasp

            if ref_success and gen_success and ref_n_grasp_contact!= gen_n_grasp_contact:
                ref_success=ref_success and ref_n_grasp_contact>gen_n_grasp_contact
                gen_success=gen_success and gen_n_grasp_contact>ref_n_grasp_contact
                margin=abs(ref_n_grasp_contact-gen_n_grasp_contact)/max(ref_n_grasp_contact,gen_n_grasp_contact)

            else:
                max_ref_fingers=min(1,target_ref_pose[5:5+3].max().item())
                max_gen_fingers=min(1,target_generated_pose[5:5+3].max().item())
                if abs(max_ref_fingers-max_gen_fingers)>0.3 and max(max_ref_fingers,max_gen_fingers)>0.7:
                    ref_success = ref_success and max_ref_fingers <max_gen_fingers
                    gen_success = gen_success and max_gen_fingers < max_ref_fingers
                    margin=abs(max_ref_fingers-max_gen_fingers)/max(max_ref_fingers,max_gen_fingers)

            if ref_success and not gen_success and counter2<self.batch_size:
                superior_A_model_moving_rate.update(0.)
                counter2+=1
            elif gen_success and not ref_success and counter2<self.batch_size:
                superior_A_model_moving_rate.update(1.)
                counter2+=1

            u=(1.001 - F.cosine_similarity(target_generated_pose,
                                         sampling_centroid, dim=0)) / 2

            if not ref_success or (ref_success and gen_success):
                if np.random.rand() > u*((1 - torch.clamp(grasp_quality[target_index], 0, 1).item()) * annealing_factor):
                    print('                                                         ...5')
                    continue

            if ref_success and gen_success:continue


            if ref_success == gen_success:continue

            # print(f'ref_in_scope={ref_in_scope}, gen_in_scope={gen_in_scope}')

            k=1 if ref_success and not gen_success else -1
            if k == 1:
                sampler_samples+=1



            counter += 1
            t = 0
            hh = (counter / self.batch_size) ** 2
            n = int(min(hh * max_n + n, avaliable_iterations))


            self.skip_rate.update(t)

            pairs.append((target_index,  k,margin))

            self.tmp_pose_record.append(target_generated_pose.detach().clone())

            superior_pose = target_ref_pose if k > 0 else target_generated_pose


            self.taxonomies.update(superior_pose.detach().clone())



            if sampling_centroid is None:
                sampling_centroid = superior_pose.detach().clone()
            else:
                sampling_centroid = sampling_centroid * 0.999 + superior_pose.detach().clone() * 0.001

            if counter == self.batch_size: break

        if counter == self.batch_size:
            return True, pairs, sampling_centroid,sampler_samples
        else:
            return False, pairs, sampling_centroid,sampler_samples

    def step(self,i):
        self.ch_env.drop_new_obj(stablize=np.random.random()>0.)

        '''get scene perception'''
        depth, pc, floor_mask = self.ch_env.get_scene_preception(view=False)


        depth = torch.from_numpy(depth).cuda()  # [600.600]
        # torch.save(depth, 'depth_ch_tmp')
        floor_mask = torch.from_numpy(floor_mask).cuda()
        # torch.save(floor_mask, 'floor_mask_ch_tmp')
        # exit()

        latent_vector=torch.randn((1,8,depth.shape[0],depth.shape[1]),device='cuda')


        for k in range(self.iter_per_scene):




            with torch.no_grad():

                gripper_pose, grasp_quality_logits, grasp_collision_logits = self.gan.generator(
                    depth[None, None, ...],~floor_mask.view(1,1,600,600),latent_vector,detach_backbone=True,backbone=self.gan.critic.back_bone)
                # break
                # gripper_pose = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000, self.n_param)
                # print(torch.abs(gripper_pose[~floor_mask]).mean(dim=0))
                # print(torch.abs(gripper_pose[floor_mask]).mean(dim=0))
                # exit()
                grasp_quality=F.sigmoid(grasp_quality_logits)
                grasp_collision=F.sigmoid(grasp_collision_logits)


                # grasp_quality=torch.clamp(grasp_quality_logits,0,1)

                f =(1 - grasp_quality.detach())
                n=max(self.tou,self.skip_rate.val)**2
                annealing_factor = n+(1-n)*f
                print(Fore.LIGHTYELLOW_EX,f'mean_annealing_factor= {annealing_factor.mean()}, tou={self.tou}, skip rate={self.skip_rate.val}',Fore.RESET)

                self.max_G_norm=max(self.G_grad_norm_MR.val* self.tou**2 + (1 - self.tou**2)*5,5)
                self.max_D_norm= max(self.D_grad_norm_MR.val * self.tou**2 + (1 - self.tou**2)*1,1)

                # if self.tou<0.6:
                #     self.batch_size=1
                #     self.learning_rate=1e-5
                # elif self.tou>0.7:
                #     self.batch_size = 1
                #     self.learning_rate=1e-4


                gripper_pose_ref = ch_pose_interpolation(gripper_pose, self.sampling_centroid,
                                                         annealing_factor=annealing_factor,taxonomies=        self.taxonomies.centers)  # [b,self.n_param,600,600]

                if i % int(50) == 0 and i != 0 and k == 0:
                    try:
                        self.export_check_points()
                        self.save_statistics()
                    except Exception as e:
                        print(Fore.RED, str(e), Fore.RESET)
                if i % 10 == 0 and k == 0:
                    self.view_result(gripper_pose, floor_mask)

                self.tmp_pose_record = []
                counted, pairs, sampling_centroid ,sampler_samples= self.sample_contrastive_pairs(pc, floor_mask, gripper_pose,
                                                                                  gripper_pose_ref,
                                                                                  self.sampling_centroid,
                                                                                  self.tou, grasp_quality.detach(),grasp_collision.detach(),
                                                                                  self.superior_A_model_moving_rate,latent_vector)
            if not counted:

                # self.superior_A_model_moving_rate.update(0)
                self.tou = 1 - self.superior_A_model_moving_rate.val
                # self.step_generator_without_sampler(depth,floor_mask,pc,latent_vector)
                if k==0:
                    self.ch_env.update_obj_info(0.1)
                    self.skip_rate.update(1.)
                    self.ch_env.remove_obj()

                break
            else:
                self.skip_rate.update(0.)
                self.ch_env.update_obj_info(0.9)



            self.tou = 1 - self.superior_A_model_moving_rate.val

            self.sampling_centroid = sampling_centroid


            gripper_pose = gripper_pose[0].permute(1, 2, 0).reshape(360000, self.n_param)
            gripper_pose_ref = gripper_pose_ref[0].permute(1, 2, 0).reshape(360000, self.n_param)

            self.step_discriminator(depth, gripper_pose, gripper_pose_ref, pairs,floor_mask,latent_vector=latent_vector)

            # if sampler_samples==batch_size:
            self.step_generator(depth, floor_mask, pc, gripper_pose_ref, pairs,latent_vector)
            # else:
            #     self.step_generator_without_sampler(depth, floor_mask, pc,latent_vector)
            # continue


    def view_result(self, gripper_poses,floor_mask):
        with torch.no_grad():

            print('Center pos: ',self.sampling_centroid.cpu().numpy())


            cuda_memory_report()

            values = gripper_poses[0].permute(1, 2, 0).reshape(360000, self.n_param).detach()  # .cpu().numpy()
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
            self.skip_rate.view()
            self.superior_A_model_moving_rate.view()

            self.G_grad_norm_MR.view()
            self.D_grad_norm_MR.view()

            self.bin_collision_statistics.print()
            self.objects_collision_statistics.print()
            self.collision_statistics.print()

            self.grasp_quality_statistics.print()


            self.taxonomies.view()

            # self.quat_centers.view()
            # self.fingers_centers.view()
            # self.transition_centers.view()

    def save_statistics(self):
        self.moving_collision_rate.save()
        self.skip_rate.save()
        self.superior_A_model_moving_rate.save()

        self.G_grad_norm_MR.save()
        self.D_grad_norm_MR.save()

        self.critic_statistics.save()
        self.gripper_sampler_statistics.save()

        self.data_tracker.save()


        self.bin_collision_statistics.save()
        self.collision_statistics.save()
        self.grasp_quality_statistics.save()

        torch.save(self.sampling_centroid,self.last_pose_center_path)

        self.taxonomies.save()

        # self.quat_centers.save()
        # self.fingers_centers.save()
        # self.transition_centers.save()

        self.ch_env.save_obj_dict()

    def export_check_points(self):
        self.gan.export_models()
        self.gan.export_optimizers()

    def clear(self):
        self.critic_statistics.clear()

        self.bin_collision_statistics.clear()
        self.collision_statistics.clear()
        self.gripper_sampler_statistics.clear()
        self.grasp_quality_statistics.clear()

    def begin(self,iterations=10):
        pi = progress_indicator('Begin new training round: ', max_limit=iterations)

        for i in range(iterations):
            if self.skip_rate.val > 0.7:
                self.batch_size = 1
                self.iter_per_scene = 5
                self.ch_env.max_obj_per_scene = 1
            elif self.skip_rate.val < 0.2:
                self.batch_size = 2
                self.iter_per_scene = 1
                self.ch_env.max_obj_per_scene = 10
            # cuda_memory_report()
            if args.catch_exceptions:
                try:
                    self.step(i)
                    pi.step(i)
                except Exception as e:
                    print(Fore.RED, str(e), Fore.RESET)
                    torch.cuda.empty_cache()
                    self.ch_env.update_obj_info(0.1)
                    self.ch_env.remove_all_objects()
            else:
                self.step(i)
                pi.step(i)

        pi.end()

        self.export_check_points()
        self.save_statistics()
        self.clear()

def train_N_grasp_GAN(args,n=1):
    lr = 1e-5

    Train_grasp_GAN = TrainGraspGAN(args,n_samples=None, learning_rate=lr)
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

def read_config(path):
    config = configparser.ConfigParser()
    config.read(path)
    return config

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.ini",
        help="Path to the config file"
    )

    parser.add_argument(
        "--load_last_optimizer",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Load last optimizer state (default: True). Use true/false."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate"
    )


    parser.add_argument(
        "--catch_exceptions",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Wrap the execution with try and except (default: True). Use true/false."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Normalize filename (avoid config.ini.ini)
    config_path = args.config
    if not config_path.lower().endswith(".ini"):
        config_path += ".ini"

    # Read config
    config = read_config(config_path)

    print("Config path:", os.path.abspath(config_path))
    print("load_last_optimizer:", args.load_last_optimizer)

    train_N_grasp_GAN(args,n=10000)
