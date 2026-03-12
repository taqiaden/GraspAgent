import argparse
import configparser
import os
import random
import time
import traceback
import numpy as np
from colorama import Fore
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import spconv.pytorch as spconv
from torch_scatter import scatter_mean
from GraspAgent_2.kinematic_utils.path_check import kinematic_checker
from GraspAgent_2.model.CH_model import CH_model_key, CH_D, CH_G
from GraspAgent_2.sim_hand_s.Casia_hand_env import CasiaHandEnv
from GraspAgent_2.training.sample_random_grasp import ch_pose_interpolation
from GraspAgent_2.utils.Voxel_operations import crop_sphere_torch, crop_cube, view_3d_occupancy_grid
from GraspAgent_2.utils.dynamic_dataset import DynamicDataManagement, SynthesisedData
from GraspAgent_2.utils.focal_loss import FocalLoss
from GraspAgent_2.utils.quat_operations import quat_rotate_vector, grasp_frame_to_quat, quat_between
from Online_data_audit.data_tracker import gripper_grasp_tracker, DataTracker
from check_points.check_point_conventions import GANWrapper
from lib.IO_utils import custom_print
from lib.cuda_utils import cuda_memory_report
from lib.image_utils import view_image
from lib.loss.D_loss import binary_l1, binary_smooth_l1
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from records.training_satatistics import TrainingTracker, MovingRate
import torch
from visualiztion import view_npy_open3d
from collections import deque

freeze_G_backbone = False
freeze_D_backbone = False

test_mode=False
view=False
synthesizie_only=False

hard_level_factor=0

max_n = 20
k_d=1
k_g=1

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

def c_loss(pred,label):
    loss=binary_l1(pred+0.5,label)**2
    # loss=bce_with_logits(pred,label)
    # loss=sigmoid_focal_loss(pred, label, gamma=2.0, alpha=0.5)
    return loss

def c_loss2(pred, label):
    loss=binary_l1(pred+0.5,label)**2
    # loss=bce_with_logits(pred,label)
    # loss=sigmoid_focal_loss(pred, label, gamma=2.0, alpha=0.5)
    return loss

    # if balance_indicator is None: return loss
    # else:
    #     if balance_indicator>0:
    #         y=abs(balance_indicator)
    #         w=(1 - label) * y + (1 - y)
    #
    #     else:
    #         y = abs(balance_indicator)
    #         w=label*y+(1-y)
    #
    #     loss=loss*w
    #
    #     # weights = torch.exp(-loss.detach())
    #     # loss = (weights * loss)
    #     return loss
def logits_to_probs(logits):
    return torch.clamp(logits+0.5,0,1)
    # return F.sigmoid(logits)
def logits_to_probs2(logits):
    return torch.clamp(logits+0.5,0,1)
    # return F.sigmoid(logits)
    # return torch.clamp(logits,0,1)



def hinge_loss(positive, negative,margin,k=1.):
    loss = torch.clamp((negative.squeeze() - positive.squeeze())   + margin*k, 0.)
    return loss

def euclidean_triplet_loss(anchor, positive, negative, margin_signal):
    margin = torch.as_tensor(margin_signal * 1.0, device=anchor.device)
    d_pos  = (anchor - positive).pow(2).sum(dim=-1).sqrt()   # L2 distance
    d_neg  = (anchor - negative).pow(2).sum(dim=-1).sqrt()
    loss   = F.relu(d_pos - d_neg + margin)
    return loss.mean()

def cosine_triplet_loss(anchor, positive, negative, margin_signal):
    d_pos=(anchor*positive).sum()
    d_neg=(anchor*negative).sum()
    loss=torch.clamp(d_neg-d_pos+margin_signal*0.3,0)

    # margin_signal=torch.tensor([margin_signal],device=anchor.device)
    # d_pos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
    # d_neg = 1 - F.cosine_similarity(anchor, negative, dim=-1)
    # loss = F.relu(d_pos - d_neg + margin_signal*0.3)
    return loss.mean()

def cosine_repulsion_loss(feat, eps=1e-6):
    """
    feat: Tensor of shape [N, 64]
    """
    n = feat.size(0)

    if n > 100:
        idx = torch.randperm(n, device=feat.device)[:100]
        feat = feat[idx]

    # L2 normalize
    feat = feat / (feat.norm(dim=1, keepdim=True) + eps)

    # Cosine similarity matrix
    sim = torch.matmul(feat, feat.T)  # [N, N]

    # Remove diagonal (self-similarity)
    N = feat.size(0)
    mask = ~torch.eye(N, dtype=torch.bool, device=feat.device)

    # Squared cosine similarity
    loss = (sim[mask] ** 2).mean()
    return loss

def visualize_pointcloud_with_index_open3d(pc, idx):
    """
    pc: (N, 3) numpy array
    idx: int
    """
    import open3d as o3d
    assert pc.ndim == 2 and pc.shape[1] == 3
    assert 0 <= idx < pc.shape[0]

    # Base point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # Color all points gray
    colors = np.tile([0.6, 0.6, 0.6], (pc.shape[0], 1))
    colors[idx] = [1.0, 0.0, 0.0]  # red target
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Optional: add a small sphere at the indexed point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    sphere.translate(pc[idx])
    sphere.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd, sphere])


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
def balanced_sampling(values, mask=None, exponent=2.0, balance_indicator=.0,eps=1e-4):
    with torch.no_grad():
        max_ = values.max().item()
        min_ = values.min().item()
        range_ = max_ - min_

        if not range_ > 0.:
            selection_probability=torch.rand_like(values)

        else:
            s=(values-min_)/range_
            # s = (1 - abs(balance_indicator))*s + torch.rand_like(values) * abs(balance_indicator)

            pivot_point = np.sqrt(np.abs(balance_indicator)) * np.sign(balance_indicator)
            xa=(1-s)* pivot_point
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

def random_sampling(values, mask=None):
    with torch.no_grad():
        selection_probability=torch.rand_like(values)

        if mask is None:
            dist = Categorical(probs=selection_probability)
        else:
            dist = MaskedCategorical(probs=selection_probability, mask=mask)

        target_index = dist.sample()


        return target_index


def half_way_unit_vector(v: torch.Tensor) -> torch.Tensor:
    """
    v: torch.Tensor, shape (2,), (x, y)
    Returns unit vector that lies half-way (angle/2) **anticlockwise**
    from the positive x-axis to v.
    """
    v = v.float()
    angle = torch.atan2(v[1], v[0])  # −π … π
    angle = angle % (2 * torch.pi)  # map to 0 … 2π
    half = angle / 2
    return torch.stack([half.cos(), half.sin()])

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

def process_fingers(target_pose_):
    fingers = torch.clip(target_pose_[5:5 + 3] + 0.5, 0, 1)
    fingers = fingers
    return fingers

def process_pose(target_point, target_pose, view=False):
    target_pose_ = target_pose.clone()
    target_point_ = target_point.cpu().numpy() if torch.is_tensor(target_point) else target_point

    # quat = target_pose_[:4].cpu().tolist()
    alpha=target_pose_[:3]
    # alpha[-1]=torch.clip(alpha[-1],max=0.)

    # beta=half_way_unit_vector(target_pose_[3:5])
    beta=target_pose_[3:5]

    alpha = F.normalize(alpha, p=2, dim=0, eps=1e-8)
    beta = F.normalize(beta, p=2, dim=0, eps=1e-8)

    approach_ref=torch.tensor([0.866, -0.5, 0],device='cuda')

    default_quat = quat_between(approach_ref, torch.tensor([0., 0., -1.],device='cuda'))
    quat=grasp_frame_to_quat(alpha, beta, default_quat).cpu().tolist()

    fingers=process_fingers(target_pose_).cpu().tolist()

    fingers[0] = 1.
    fingers[1] = 1.
    fingers[2] = 1.

    transition=torch.clip(target_pose_[5+3:5+3+1],0,1)
    # transition=target_pose_[5+3:5+3+1]
    transition = (transition.cpu().numpy()) / 10
    projected_transition = quat_rotate_vector(quat, approach_ref.tolist())*transition[0]

    # approach = quat_rotate_vector(np.array(quat), np.array([0, 0, 1]))
    # projected_transition = approach * transition

    shifted_point = (target_point_ + projected_transition).tolist()
    # shifted_point = (target_point_ ).tolist()

    assert all(x == x for x in quat), f"quat contains NaN, {quat}"
    assert all(x == x for x in fingers), f"fingers contains NaN, {fingers}"
    assert all(x == x for x in shifted_point), f"shifted_point contains NaN, {shifted_point}"

    if view:
        print()
        print('quat: ',quat)
        print('fingers: ',fingers)
        print('transition: ',transition)
        print('target_point_: ',target_point_)
        print('projected_transition: ',projected_transition)
        print('shifted_point: ',shifted_point)

    return quat,fingers,shifted_point


class TrainGraspGAN:
    def __init__(self, args,n_samples=None, epochs=1, learning_rate=5e-5):

        self.args=args

        self.batch_size=2
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

        self.kinematics = kinematic_checker()

        self.DDM=DynamicDataManagement(key=CH_model_key+'_synthesized_dynamic_data')

        self.loaded_synthesised_data=None

        # self.quat_centers=OnlingClustering(key_name=CH_model_key+'_quat',number_of_centers=20,vector_size=4,decay_rate=0.01,is_quat=True,dist_threshold=0.77)
        # self.fingers_centers=OnlingClustering(key_name=CH_model_key+'_fingers',number_of_centers=10,vector_size=3,decay_rate=0.01,use_euclidean_dist=True,dist_threshold=0.2)
        # self.transition_centers=OnlingClustering(key_name=CH_model_key+'_transitions',number_of_centers=5,vector_size=1,decay_rate=0.01,use_euclidean_dist=True,dist_threshold=0.2)
        # inti_centers=torch.tensor([[0, 1,0, 1, 0, 0.5,0.5,0.5,  0.]],device='cuda')
        # self.taxonomies=OnlingClustering(key_name=CH_model_key+'_taxonomies',number_of_centers=30,vector_size=9,decay_rate=0.01,use_euclidean_dist=True)
        # self.alpha=OnlingClustering(key_name=CH_model_key+'_alpha',number_of_centers=8,vector_size=3,decay_rate=0.01,use_euclidean_dist=False)
        # self.beta=OnlingClustering(key_name=CH_model_key+'_beta',number_of_centers=10,vector_size=2,decay_rate=0.01,use_euclidean_dist=False)
        # self.fingers=OnlingClustering(key_name=CH_model_key+'_fingers',number_of_centers=7,vector_size=3,decay_rate=0.01,use_euclidean_dist=True)
        # self.transition=OnlingClustering(key_name=CH_model_key+'_transition',number_of_centers=3,vector_size=1,decay_rate=0.01,use_euclidean_dist=True)

    def initialize(self, n_samples=None):
        self.n_samples = n_samples

        '''Moving rates'''
        self.moving_collision_rate = MovingRate(CH_model_key + '_collision', decay_rate=0.01,
                                                initial_val=1.)
        self.skip_rate = MovingRate(CH_model_key + '_skip_rate',
                                                   decay_rate=0.01,
                                                    initial_val=1.)
        self.superior_A_model_moving_rate = MovingRate(CH_model_key + '_superior_A_model',
                                                       decay_rate=0.1,
                                                       initial_val=0.)

        self.G_grad_norm_MR = MovingRate(CH_model_key + '_G_grad_norm',
                                                       decay_rate=0.1,
                                                       initial_val=0.)

        self.D_grad_norm_MR = MovingRate(CH_model_key + '_D_grad_norm',
                                                       decay_rate=0.1,
                                                       initial_val=0.)

        self.Ave_samples_per_scene = MovingRate(CH_model_key + 'Ave_samples_per_scene',
                                                       decay_rate=0.1,
                                                       initial_val=0.)
        self.Ave_importance = MovingRate(CH_model_key + 'Ave_importance',
                                                       decay_rate=0.1,
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
                                                            track_label_balance=True,decay_rate=0.01)

        self.gripper_sampler_statistics = TrainingTracker(name=CH_model_key + '_gripper_sampler',
                                                          track_label_balance=False)

        self.grasp_quality_statistics = TrainingTracker(name=CH_model_key + '_grasp_quality',
                                                        track_label_balance=True,decay_rate=0.01)

        self.critic_statistics = TrainingTracker(name=CH_model_key + '_critic',
                                                  track_label_balance=False)

        self.data_tracker = DataTracker(name=CH_model_key)

    def prepare_model_wrapper(self):
        '''load  models'''
        gan = GANWrapper(CH_model_key, CH_G, CH_D)
        gan.ini_models(train=True)

        # gan.generator.back_bone2_.apply(init_weights_he_normal)
        # gan.generator.grasp_quality.apply(init_weights_he_normal)
        # gan.generator.grasp_collision.apply(init_weights_he_normal)
        # gan.generator.grasp_collision2_.apply(init_weights_he_normal)
        # gan.generator.grasp_collision3_.apply(init_weights_he_normal)

        sampler_params = []
        sampler_params += list(gan.generator.CH_PoseSampler.parameters())
        sampler_params += list(gan.generator.back_bone.parameters())

        policy_params = []
        policy_params += list(gan.generator.grasp_quality_.parameters())
        policy_params += list(gan.generator.grasp_collision_.parameters())
        policy_params += list(gan.generator.grasp_collision2.parameters())
        policy_params += list(gan.generator.grasp_collision3.parameters())
        policy_params += list(gan.generator.back_bone2_.parameters())
        # policy_params += list(gan.generator.back_bone3_.parameters())

        gan.critic_adam_optimizer(learning_rate=self.args.lr, beta1=0.9, beta2=0.999)
        # gan.critic_sgd_optimizer(learning_rate=self.args.lr*10,momentum=0.,weight_decay_=0.)
        gan.generator_adam_optimizer(param_group=policy_params,learning_rate=self.args.lr, beta1=0.9, beta2=0.999)
        # gan.generator_sgd_optimizer(learning_rate=self.args.lr*10,momentum=0.,weight_decay_=0.)
        gan.sampler_optimizer = torch.optim.SGD(sampler_params, lr=self.args.lr*10,
                                               momentum=0)
        return gan

    def step_discriminator(self,cropped_spheres,depth,clean_depth,   gripper_pose, gripper_pose_ref ,pairs,floor_mask,grasp_quality,latent_vector ):

        grasp_quality=grasp_quality.reshape(-1)
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
        anchor,positive_negative,scores = self.gan.critic(clean_depth[None,None,...], generated_grasps_stack,pairs,~floor_mask.view(1,1,600,600),cropped_spheres,latent_vector=latent_vector,detach_backbone=freeze_D_backbone)

        # print(score)
        # exit()
        # gen_scores_ = score.permute(0, 2, 3, 1)[0, :, :, 0].reshape(-1)
        # ref_scores_ = score.permute(0, 2, 3, 1)[1, :, :, 0].reshape(-1)

        # generated_embedding = positive_negative[:, 0]
        # ref_embedding = positive_negative[:, 1]

        generated_scores = scores[:, 0]
        ref_scores = scores[:, 1]

        loss = torch.tensor(0., device=depth.device)

        for j in range(len(pairs)):
            target_index = pairs[j][0]
            k = pairs[j][1]
            margin=pairs[j][2]
            sim_factor=pairs[j][3]

            assert margin>=0 and sim_factor>0, f'margin=,{margin,sim_factor}'

            # m=sim_factor*(1-grasp_quality[target_index].item()) if k>0 else (1-sim_factor)*grasp_quality[target_index].item()
            # print(f'-------------------------------------------------m={m}, k={k}')

            if k>0:
                loss+=(hinge_loss(positive=ref_scores[j],negative=generated_scores[j],margin=margin)**k_d)/self.batch_size
                # positive=ref_embedding[j]
                # negative=generated_embedding[j]
            else:
                loss+=(hinge_loss(positive=generated_scores[j],negative=ref_scores[j],margin=margin)**k_d)/self.batch_size
                # positive = generated_embedding[j]
                # negative = ref_embedding[j]


            # loss+=(cosine_triplet_loss(anchor, positive, negative, margin_signal=margin)**1)/self.batch_size
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

        params = list(self.gan.critic.att_block.parameters())
        # params += list(self.gan.critic.contx_proj.parameters())

        decoder_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf'))

        norm=torch.nn.utils.clip_grad_norm_(self.gan.critic.parameters(), max_norm=float('inf'))

        self.D_grad_norm_MR.update(norm.item())

        print(Fore.LIGHTGREEN_EX,f' D  norm : {norm}, backbone_norm:{backbone_norm}, decoder_norm={decoder_norm}',Fore.RESET)

        self.gan.critic_optimizer.step()

        self.gan.critic.zero_grad()
        self.gan.critic_optimizer.zero_grad()

        print(  Fore.LIGHTYELLOW_EX,f'd_loss={loss.item()}',
              Fore.RESET)

    def print_pairs_info(self,pairs,gripper_pose,gripper_pose_ref):
        for j in range(len(pairs)):
            target_index = pairs[j][0]
            k = pairs[j][1]
            margin=pairs[j][2]
            dist_factor=pairs[j][3]

            target_generated_pose = gripper_pose[target_index].detach()
            target_ref_pose = gripper_pose_ref[target_index].detach()

            if k<0:
                print(Fore.LIGHTGREEN_EX,f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()} , m={margin} ',Fore.RESET)
            elif k>0:
                print(Fore.LIGHTCYAN_EX,f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()} , m={margin} ',Fore.RESET)
            # else:
            #     print(Fore.LIGHTGREEN_EX,f'{target_ref_pose.cpu().numpy()} {target_generated_pose.cpu().detach().numpy()} ',Fore.RESET)

    def get_generator_loss(self, cropped_spheres,depth,clean_depth, gripper_pose, gripper_pose_ref, pairs,floor_mask,latent_vector):

        gripper_pose = gripper_pose[0].permute(1, 2, 0).reshape(360000, self.n_param)
        gripper_pose_ref = gripper_pose_ref[0].permute(1, 2, 0).reshape(360000, self.n_param)

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
        anchor,positive_negative,scores  = self.gan.critic(clean_depth[None,None,...], generated_grasps_stack,pairs,~floor_mask.view(1,1,600,600),cropped_spheres,latent_vector=latent_vector,detach_backbone=True)

        # cuda_memory_report()
        # critic_score = self.gan.critic(pc, generated_grasps_cat, detach_backbone=True)

        # gen_scores_ = critic_score.permute(0, 2, 3, 1)[0, :, :, 0].reshape(-1)
        # ref_scores_ = critic_score.permute(0, 2, 3, 1)[1, :, :, 0].reshape(-1)

        # gripper_pose = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(-1)
        # gripper_pose_ref = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :].reshape(-1)

        # gen_embedding = positive_negative[:, 0]
        # ref_embedding = positive_negative[:, 1]

        gen_scores = scores[:, 0]
        ref_scores = scores[:, 1]

        loss = torch.tensor(0., device=depth.device)

        for j in range(len(pairs)):
            target_index = pairs[j][0]
            k = pairs[j][1]
            margin=pairs[j][2]
            dist_factor=pairs[j][3]


            # loss+=(cosine_triplet_loss(anchor, positive=gen_embedding[j], negative=ref_embedding[j], margin_signal=0.))/self.batch_size
            loss += (hinge_loss(positive=gen_scores[j], negative=ref_scores[j], margin=0.) ** k_g) / self.batch_size


            # label = ref_scores_[j].squeeze()
            # pred_ = gen_scores_[j].squeeze()


            # v2 = quat_rotate_vector(target_ref_pose[0:4].cpu().tolist(), [0, 1, 0])
            # v3 = quat_rotate_vector(target_generated_pose[0:4].cpu().tolist(), [0, 1, 0])

            # print('ref approach: ',v2,' ge approach: ',v3)
            # print()

            # w=1 if k>0 else 0
            # loss += ((torch.clamp( label - pred_, 0.)) **1)/ self.batch_size
            # loss += (torch.abs(1-pred_) **1)/ self.batch_size


        return loss


    def step_generator(self,cropped_spheres,depth,clean_depth,floor_mask,pc,gripper_pose_ref,pairs,latent_vector):
        '''zero grad'''
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.generator_optimizer.zero_grad(set_to_none=True)
        self.gan.sampler_optimizer.zero_grad(set_to_none=True)
        # self.gan.critic.eval()

        '''generated grasps'''
        # cuda_memory_report()
        gripper_pose, grasp_quality_logits,  grasp_collision_logits,features,model_B_quality_logits = self.gan.generator(depth[None, None, ...],~floor_mask.view(1,1,600,600),latent_vector,
                                                                                                model_B_poses=gripper_pose_ref,detach_backbone=freeze_G_backbone)
        # cuda_memory_report()
        # exit()

        obj_features=features[0].reshape(64,-1).permute(1,0)[~floor_mask]
        repulsive_loss=cosine_repulsion_loss(obj_features)*0.1 if len(pairs)==self.batch_size else torch.tensor([0.],device=obj_features.device)

        gripper_pose_PW = gripper_pose[0].permute(1, 2, 0).reshape(360000,self.n_param)
        gripper_pose_ref_PW = gripper_pose_ref[0].permute(1, 2, 0).reshape(360000,self.n_param)

        object_collision_logits = grasp_collision_logits[0,0].reshape(-1)
        floor_collision_logits = grasp_collision_logits[0,1].reshape(-1)
        collision_logits = grasp_collision_logits[0,2].reshape(-1)

        grasp_quality_logits = grasp_quality_logits[0,0].reshape(-1)
        model_B_quality_logits = model_B_quality_logits[0,0].reshape(-1)

        '''loss computation'''
        gripper_collision_loss = torch.tensor(0., device=depth.device)
        gripper_quality_loss_ = torch.tensor(0., device=depth.device)
        n=2

        start = time.time()
        positive_counter = 0
        negative_counter = 0
        s = 1
        for k in range(n):
            '''gripper collision'''
            while True:
                # gripper_target_index = balanced_sampling(logits_to_probs2(collision_logits.detach()),
                #                                          mask=~floor_mask.detach(),
                #                                          exponent=2.0,
                #                                          balance_indicator=self.collision_statistics.label_balance_indicator)
                gripper_target_index = random_sampling(logits_to_probs(collision_logits.detach()),
                                                       mask=~floor_mask.detach(), )
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = collision_logits[gripper_target_index].squeeze()
                gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

                contact_with_obj , contact_with_floor=self.check_collision(gripper_target_point,gripper_target_pose,view=False)
                collision=contact_with_obj or contact_with_floor
                label = torch.ones_like(gripper_prediction_) if collision else torch.zeros_like(gripper_prediction_)

                object_collision_loss = c_loss2(gripper_prediction_, label)

                if  time.time()-start>5 or self.skip_rate.val>0.9:
                    print(Fore.RED,f'collision classifier exploration timeout',Fore.RESET)
                    break
                if collision and positive_counter >= s:continue
                if (not collision) and negative_counter >= s:continue
                break

            with torch.no_grad():
                self.collision_statistics.loss = object_collision_loss.item()
                self.collision_statistics.update_confession_matrix(label.detach(),
                                                                      logits_to_probs2(gripper_prediction_.detach()))
            if collision: positive_counter+=1
            else: negative_counter+=1
            gripper_collision_loss+=object_collision_loss/(n)


        # for k in range(self.batch_size*2):
        #     '''gripper-object collision'''
        #     gripper_target_index = balanced_sampling(logits_to_probs2(object_collision_logits.detach()),
        #                                              mask=~floor_mask.detach(),
        #                                              exponent=30.0,
        #                                              balance_indicator=self.objects_collision_statistics.label_balance_indicator)
        #     gripper_target_point = pc[gripper_target_index]
        #     gripper_prediction_ = object_collision_logits[gripper_target_index].squeeze()
        #     gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()
        #
        #     contact_with_obj , contact_with_floor=self.check_collision(gripper_target_point,gripper_target_pose,view=False)
        #
        #     label = torch.ones_like(gripper_prediction_) if contact_with_obj else torch.zeros_like(gripper_prediction_)
        #
        #     object_collision_loss = c_loss2(gripper_prediction_, label)
        #
        #     with torch.no_grad():
        #         self.objects_collision_statistics.loss = object_collision_loss.item()
        #         self.objects_collision_statistics.update_confession_matrix(label.detach(),
        #                                                               logits_to_probs2(gripper_prediction_.detach()))
        #
        #     gripper_collision_loss+=object_collision_loss/self.batch_size
        #
        #
        # for k in range(self.batch_size*2):
        #     '''gripper-bin collision'''
        #     gripper_target_index = balanced_sampling(logits_to_probs2(floor_collision_logits.detach()),
        #                                              mask=~floor_mask.detach(),
        #                                              exponent=30.0,
        #                                              balance_indicator=self.bin_collision_statistics.label_balance_indicator)
        #     gripper_target_point = pc[gripper_target_index]
        #     gripper_prediction_ = floor_collision_logits[gripper_target_index].squeeze()
        #     gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()
        #
        #     contact_with_obj , contact_with_floor = self.check_collision(gripper_target_point, gripper_target_pose, view=False)
        #
        #     label = torch.ones_like(gripper_prediction_) if contact_with_floor else torch.zeros_like(gripper_prediction_)
        #
        #     floor_collision_loss = c_loss2(gripper_prediction_, label)
        #
        #     with torch.no_grad():
        #         self.bin_collision_statistics.loss = floor_collision_loss.item()
        #         self.bin_collision_statistics.update_confession_matrix(label.detach(),
        #                                                                    logits_to_probs2(gripper_prediction_.detach()))
        #
        #     gripper_collision_loss += floor_collision_loss / self.batch_size
        # start = time.time()
        # positive_counter=0
        # negative_counter=0
        # for k in range(self.batch_size*2):
        #     '''grasp quality'''
        #     while True:
        #         gripper_target_index = balanced_sampling(logits_to_probs(model_B_quality_logits.detach()),
        #                                                  mask=~floor_mask.detach(),
        #                                                  exponent=30.0,
        #                                                  balance_indicator=self.grasp_quality_statistics.label_balance_indicator)
        #
        #         gripper_target_point = pc[gripper_target_index]
        #         gripper_prediction_ = model_B_quality_logits[gripper_target_index].squeeze()
        #         gripper_target_pose = gripper_pose_ref_PW[gripper_target_index].detach()
        #
        #         grasp_success,initial_collision,n_grasp_contact,self_collide,stable_grasp,warning_flag,plan_found  = self.evaluate_grasp(gripper_target_point, gripper_target_pose, view=False,hard_level=(1-self.tou)*hard_level_factor,shake=False,check_kinematics=False,update_obj_prob=gripper_prediction_>0.5)
        #         # grasp_success=grasp_success and stable_grasp
        #         # if not grasp_success and self.grasp_quality_statistics.label_balance_indicator<(np.random.random()**2)*-1: continue
        #         # else: break
        #         if warning_flag: continue
        #         if  time.time()-start>5 or self.skip_rate.val>0.9:
        #             print(Fore.RED,f'quality policy exploration timeout',Fore.RESET)
        #             break
        #         if initial_collision: continue
        #
        #         label = torch.ones_like(gripper_prediction_) if grasp_success else torch.zeros_like(gripper_prediction_)
        #         self.grasp_quality_statistics.update_confession_matrix(label.detach(),logits_to_probs(gripper_prediction_.detach()))
        #
        #         # elif grasp_success and not stable_grasp: continue
        #         # elif not grasp_success and self.grasp_quality_statistics.label_balance_indicator<(np.random.random()**2)*-1: continue
        #         if grasp_success and positive_counter >= self.batch_size:continue
        #         if (not grasp_success) and negative_counter >= self.batch_size:continue
        #
        #         break
        #     if grasp_success: positive_counter+=1
        #     else: negative_counter+=1
        #     label = torch.ones_like(gripper_prediction_) if grasp_success   else torch.zeros_like(gripper_prediction_)
        #     grasp_quality_loss = c_loss(gripper_prediction_, label)
        #
        #     with torch.no_grad():
        #         self.grasp_quality_statistics.loss = grasp_quality_loss.item()
        #         self.objects_collision_statistics.update_confession_matrix(label.detach(),
        #                                                                logits_to_probs(gripper_prediction_.detach()))
        #
        #     gripper_quality_loss_ += grasp_quality_loss / self.batch_size
            # print(f'positive_counter: {positive_counter}, negative_counter: {negative_counter}')



        start = time.time()
        positive_counter=0
        negative_counter=0
        s=int(n/2)
        for k in range(n):
            '''grasp quality'''
            while True:
                # gripper_target_index = balanced_sampling(logits_to_probs(grasp_quality_logits.detach()),
                #                                          mask=~floor_mask.detach(),
                #                                          exponent=2.0,
                #                                          balance_indicator=self.grasp_quality_statistics.label_balance_indicator)
                gripper_target_index=random_sampling(logits_to_probs(grasp_quality_logits.detach()),
                                                         mask=~floor_mask.detach(),)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = grasp_quality_logits[gripper_target_index].squeeze()
                gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

                grasp_success,initial_collision,n_grasp_contact,self_collide,stable_grasp,warning_flag,plan_found ,grasped_obj = self.evaluate_grasp(gripper_target_point, gripper_target_pose, view=False,hard_level=(1-self.tou)*hard_level_factor,shake=False,check_kinematics=False,update_obj_prob=gripper_prediction_>0.5)
                # grasp_success=grasp_success and stable_grasp
                # if not grasp_success and self.grasp_quality_statistics.label_balance_indicator<(np.random.random()**2)*-1: continue
                # else: break
                if warning_flag: continue
                if  time.time()-start>5 or self.skip_rate.val>0.9:
                    print(Fore.RED,f'quality policy exploration timeout',Fore.RESET)
                    break
                # if initial_collision: continue

                label = torch.ones_like(gripper_prediction_) if grasp_success else torch.zeros_like(gripper_prediction_)
                self.grasp_quality_statistics.update_confession_matrix(label.detach(),logits_to_probs(gripper_prediction_.detach()))

                # elif grasp_success and not stable_grasp: continue
                # elif not grasp_success and self.grasp_quality_statistics.label_balance_indicator<(np.random.random()**2)*-1: continue
                if grasp_success and positive_counter >= s:continue
                if (not grasp_success) and negative_counter >= s:continue

                break
            if grasp_success: positive_counter+=1
            else: negative_counter+=1
            label = torch.ones_like(gripper_prediction_) if grasp_success   else torch.zeros_like(gripper_prediction_)
            grasp_quality_loss = c_loss(gripper_prediction_, label)



            with torch.no_grad():
                self.grasp_quality_statistics.loss = grasp_quality_loss.item()
                self.objects_collision_statistics.update_confession_matrix(label.detach(),
                                                                       logits_to_probs(gripper_prediction_.detach()))

            gripper_quality_loss_ += grasp_quality_loss / (n)

            # print(f'positive_counter: {positive_counter}, negative_counter: {negative_counter}')

        if len(pairs) == self.batch_size:
            gripper_sampling_loss = self.get_generator_loss(cropped_spheres,
                depth, clean_depth, gripper_pose, gripper_pose_ref,
                pairs,floor_mask,latent_vector)

            assert not torch.isnan(gripper_sampling_loss).any(), f'{gripper_sampling_loss}'

            print(Fore.LIGHTYELLOW_EX,
                  f'gripper_sampling_loss={gripper_sampling_loss.item()}, gripper_quality_loss_={gripper_quality_loss_.item()}, gripper_collision_loss={gripper_collision_loss.item()}',
                  Fore.RESET)
            with torch.no_grad():
                self.gripper_sampler_statistics.loss = gripper_sampling_loss.item()
        else: gripper_sampling_loss=0.

        loss = (gripper_sampling_loss+gripper_collision_loss+gripper_quality_loss_+repulsive_loss)


        # if abs(loss.item())>0.0:
        # try:
        loss.backward()

        '''GRADIENT CLIPPING'''
        # params=list(self.gan.generator.back_bone.parameters()) + \
        #  list(self.gan.generator.CH_PoseSampler.parameters())
        norm=torch.nn.utils.clip_grad_norm_(self.gan.generator.parameters(), max_norm=float('inf'))
        self.G_grad_norm_MR.update(norm.item())

        norm1=torch.nn.utils.clip_grad_norm_(self.gan.generator.back_bone.parameters(), max_norm=float('inf'))
        norm2=torch.nn.utils.clip_grad_norm_(self.gan.generator.back_bone2_.parameters(), max_norm=float('inf'))
        # norm3=torch.nn.utils.clip_grad_norm_(self.gan.generator.back_bone3_.parameters(), max_norm=float('inf'))

        print(Fore.LIGHTGREEN_EX,f' G norm : {norm}, backbone1:{norm1}, bacckbone2: {norm2},  repulsive_loss: {repulsive_loss.item()}',Fore.RESET)

        # params2 = list(self.gan.generator.back_bone.parameters()) + \
        #          list(self.gan.generator.grasp_quality.parameters())+ \
        #          list(self.gan.generator.grasp_collision.parameters())+ \
        #          list(self.gan.generator.grasp_collision2_.parameters())+ \
        #          list(self.gan.generator.grasp_collision3_.parameters())
        # norm = torch.nn.utils.clip_grad_norm_(params2, max_norm=float('inf'))
        # print(Fore.LIGHTGREEN_EX, f' G norm 2 : {norm}', Fore.RESET)

        self.gan.generator_optimizer.step()
        self.gan.sampler_optimizer.step()

        self.gan.generator.zero_grad(set_to_none=True)
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator_optimizer.zero_grad(set_to_none=True)
        self.gan.critic_optimizer.zero_grad(set_to_none=True)
        self.gan.sampler_optimizer.zero_grad(set_to_none=True)


    def check_collision(self,target_point,target_pose,view=False):
        with torch.no_grad():
            quat, fingers, shifted_point = process_pose(target_point, target_pose, view=view)


        return self.ch_env.check_collision(hand_pos=shifted_point,hand_quat=quat,hand_fingers=None,view=view)

    def evaluate_grasp(self, target_point, target_pose, view=False,hard_level=0,shake=True,check_kinematics=True,update_obj_prob=False):
        grasped_obj=None
        with torch.no_grad():
            quat,fingers,shifted_point= process_pose(target_point, target_pose, view=test_mode)

            if view:
                in_scope, grasp_success, contact_with_obj, contact_with_floor, n_grasp_contact, self_collide, stable_grasp = self.ch_env.view_grasp(
                    hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers,
                    view=view, hard_level=hard_level)
                warning_flag=False
            else:
                in_scope, grasp_success, contact_with_obj, contact_with_floor,n_grasp_contact,self_collide,stable_grasp,warning_flag,grasped_obj  = self.ch_env.check_graspness(
                    hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers,
                   view=view,hard_level=hard_level,shake=shake,update_obj_prob=update_obj_prob)

            initial_collision=contact_with_obj or contact_with_floor

            if warning_flag: print(Fore.RED,f' ----------------------------- warning_flag',Fore.RESET)

            # print('in_scope, grasp_success, contact_with_obj, contact_with_floor :',in_scope, grasp_success, contact_with_obj, contact_with_floor )

            if grasp_success is not None:
                if grasp_success and not contact_with_obj and not contact_with_floor:
                    plan_found=self.kinematics.kinematic_plan_exist(quat,shifted_point) if check_kinematics else True
                    return grasp_success ,initial_collision,n_grasp_contact,self_collide,stable_grasp,warning_flag,plan_found,grasped_obj

        return False, initial_collision,n_grasp_contact,self_collide,stable_grasp,warning_flag,None,grasped_obj



    def sample_contrastive_pairs(self,pc,  floor_mask, gripper_pose, gripper_pose_ref,
                                   annealing_factor, grasp_quality,grasp_collision,
                                 superior_A_model_moving_rate,latent_vector,model_b_quality):
        start = time.time()

        d_pairs = []
        g_pairs = []

        all_pairs=[]

        selection_mask = (~floor_mask) #& (latent_vector.reshape(-1)!=0)
        grasp_quality=grasp_quality[0,0].reshape(-1)
        model_b_quality=model_b_quality[0,0].reshape(-1)
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

        # batch_center=clipped_gripper_pose_PW[:,0:5].mean(dim=0)

        gamma_dive = norm_((1.001 - F.cosine_similarity(clipped_gripper_pose_PW[:,0:5],
                                                        self.sampling_centroid[None, 0:5], dim=-1) ) /2 ,1)

        gamma_dive *= norm_((1.001 - F.cosine_similarity(gripper_pose_ref_PW[:,0:5],
                                                        self.sampling_centroid[None, 0:5], dim=-1) ) /2 ,1)
        # gamma_dive *= norm_((1.001 - F.cosine_similarity(gripper_pose_ref_PW[:,0:5],
        #                                                 clipped_gripper_pose_PW[:,0:5], dim=-1) ) /2 ,1)

        # selection_p = compute_sampling_probability(sampling_centroid, gripper_pose_ref_PW, gripper_pose_PW, pc, bin_mask,grasp_quality)
        # gamma_transition=norm_(clipped_gripper_pose_PW[:,-1],expo_=2)

        # selection_p = norm_(1-grasp_quality,expo_=1)*norm_(model_b_quality,expo_=1)#*norm_(grasp_collision,expo_=1)#*gamma_transition
        # selection_p=(1-torch.abs(grasp_quality-0.5)*2).clamp(min=0.001)**2
        selection_p=torch.rand_like(grasp_quality)
        if test_mode: selection_p=grasp_quality**2
        # gamma_col=norm_(1-grasp_collision)*self.skip_rate()+(1-self.skip_rate())
        # gamma_graspness=grasp_quality*self.skip_rate()+(1-self.skip_rate())
        # gamma_quality=norm_(grasp_quality)

        # gamma_dive=gamma_dive*self.skip_rate()+(1-self.skip_rate())
        # selection_p = (torch.rand_like(gamma_dive)) #** (1/3) #* self.tou + (1 - self.tou)*torch.rand_like(gamma_dive)

        avaliable_iterations = selection_mask.sum()
        if avaliable_iterations<3: return [], [],None

        initial_mask=(grasp_quality>0.5) | (model_b_quality>0.5)
        initial_mask=initial_mask & selection_mask
        high_quality_samples = initial_mask.sum()

        n = int(min(max_n, avaliable_iterations))

        print(Fore.LIGHTBLACK_EX,'# Available candidates =',avaliable_iterations.item(), ' high_quality_samples =',high_quality_samples.item(),Fore.RESET)

        counter = 0
        counter2=0

        sampler_samples=0

        t = 0
        while t < n:
            # print(t)
            time_out=time.time()-start
            if time_out>5 and not  test_mode: break
            t += 1
            # if torch.any(initial_mask):
            #     dist = MaskedCategorical(probs=selection_p, mask=initial_mask)
            # else:
            # ideal_ref=False
            importance=None
            if self.loaded_synthesised_data is not None and len(self.loaded_synthesised_data)>0:
                # target_index=preferred_indexes.pop()
                target_index,_,_,importance,_=self.loaded_synthesised_data.sample_pop()
                # print(f'test ----- target_index={target_index}, ideal pose = {gripper_pose_ref_PW[target_index]}')
                # ideal_ref=True
            else:
                dist = MaskedCategorical(probs=selection_p, mask=selection_mask)
                target_index = dist.sample().item()

            selection_mask[target_index] *= False
            initial_mask[target_index] *= False

            avaliable_iterations -= 1
            target_point = pc[target_index]

            target_generated_pose = gripper_pose_PW[target_index]
            target_ref_pose = gripper_pose_ref_PW[target_index]

            margin=1.

            # assert target_ref_pose[0]>0,f'{target_ref_pose}'

            if test_mode:
                contact_with_obj , contact_with_floor=self.check_collision(target_point, target_generated_pose,view=False)
                if contact_with_obj or contact_with_floor: continue
                # contact_with_obj , contact_with_floor=self.check_collision(target_point, target_generated_pose,view=True)

                view_r=self.evaluate_grasp(
                    target_point, target_generated_pose, view=True, shake=True)
                #
                # print(Fore.LIGHTCYAN_EX,f'return f1: {view_r}')
                # view_r2 = self.evaluate_grasp(
                #     target_point, target_generated_pose, view=False, shake=True)
                # print(Fore.LIGHTCYAN_EX,f'return f2: {view_r2}')

                g_pairs.append((target_index,  1,1))
                d_pairs.append((target_index,  1,1))
                return  d_pairs,g_pairs,  1

            ref_quality = model_b_quality[target_index]
            gen_quality = grasp_quality[target_index]

            ref_success ,ref_initial_collision,ref_n_grasp_contact,ref_self_collide,stable_ref_grasp,warning_flag,ref_plan_found,ref_grasped_obj = self.evaluate_grasp(target_point,target_ref_pose,view=False,shake=False,update_obj_prob=False)
            # if ideal_ref : assert ref_success
            if ref_success and ( not ref_plan_found): continue
            # if ref_success and ( not stable_ref_grasp): continue

            # ref_success=ref_success   and ref_plan_found

            if warning_flag:
                break
                # continue
            gen_success, gen_initial_collision, gen_n_grasp_contact, gen_self_collide, stable_gen_grasp,warning_flag,gen_plan_found,gen_grasped_obj = self.evaluate_grasp(
                target_point, target_generated_pose, view=False, shake=False,check_kinematics=False,update_obj_prob=gen_quality>0.5)
            if gen_success and (not gen_plan_found): continue
            # gen_success=gen_success and stable_gen_grasp

            # print(f'ref_success:{ref_success}, gen_success:{gen_success}')

            ref_transition = torch.clamp(target_ref_pose[-1], 0, 1)
            gen_transition = torch.clamp(target_generated_pose[-1], 0, 1)

            if warning_flag:
                break
                # continue

            if t==1 and self.skip_rate()>0.9:
                print(f' ref ---- {target_ref_pose}, {ref_success ,ref_initial_collision,ref_n_grasp_contact,ref_self_collide}')
                print()
                print(f' gen ---- {target_generated_pose}, {ref_success ,ref_initial_collision,ref_n_grasp_contact,ref_self_collide}')


            # if gen_success and ref_success:
            #     ref_success = ref_success and stable_ref_grasp
            #     gen_success = gen_success and stable_gen_grasp
            #     margin = abs(ref_transition - gen_transition) / max(ref_transition, gen_transition)
                # margin = 0.999

            # if ref_success and gen_success:
            #     if gen_n_grasp_contact!=ref_n_grasp_contact:
            #         ref_success=ref_success and ref_n_grasp_contact>gen_n_grasp_contact
            #         gen_success=gen_success and gen_n_grasp_contact>ref_n_grasp_contact
            #         margin=abs(ref_transition-gen_transition)/max(ref_transition,gen_transition)
                    # margin=0.99

            # if ref_success and gen_success :
            #     if abs(ref_transition-gen_transition)>0.1:
            #         # margin=abs(ref_quality-gen_quality)#/(ref_quality+gen_quality)
            #         ref_success=ref_success and ref_transition>gen_transition
            #         gen_success=gen_success and gen_transition>ref_transition
            #         # margin=0.99

            # if ref_success and gen_success:
            #     max_ref_finger=target_ref_pose[5:8].max()+0.5
            #     max_gen_finger=target_generated_pose[5:8].max()+0.5
            #     if abs(max_ref_finger-max_gen_finger)>0.1:
            #         margin=abs(max_ref_finger-max_gen_finger)/max(max_ref_finger,max_gen_finger)
            #         ref_success=ref_success and max_gen_finger>max_ref_finger
            #         gen_success=gen_success and max_ref_finger>max_gen_finger

            if gen_success :
                if importance is not None and ref_success and importance>grasp_quality[target_index].item():
                    importance*=0.9
                    if importance > 0.1:
                        all_pairs.append((target_index, target_point, gripper_pose_ref_PW[target_index], importance,
                                          ref_grasped_obj))
                else:
                    importance = grasp_quality[target_index].item() if importance is None else importance*0.5+0.5*grasp_quality[target_index].item()
                    all_pairs.append((target_index,target_point,gripper_pose_PW[target_index],importance,gen_grasped_obj))
            elif ref_success :
                importance=1-grasp_quality[target_index].item() if importance is None else importance*0.9
                if importance>0.1:
                    all_pairs.append((target_index, target_point, gripper_pose_ref_PW[target_index],importance,ref_grasped_obj))



            if ref_success == gen_success:
                if not ref_success and not gen_success:
                    self.ch_env.update_obj_info(1e-2,decay=0.99)
                continue

            # if not ref_success and (len(d_pairs) == self.batch_size) :continue

            k=1 if ref_success and not gen_success else -1

            if k == 1:
                sampler_samples+=1

            counter += 1
            t = 0
            hh = (counter / self.batch_size) ** 2
            n = int(min(hh * max_n + n, avaliable_iterations))

            # self.skip_rate.update(t)

            cos_dist = (1 - F.cosine_similarity(target_generated_pose[0:5], target_ref_pose[0:5], dim=-1)) / 2
            scaler_dist=torch.linalg.norm(torch.clip(target_generated_pose[-1],0,1) - torch.clip(target_ref_pose[-1],0,1))
            scaler_dist2=torch.linalg.norm(torch.clip(target_generated_pose[5:8]+0.5,0,1) - torch.clip(target_ref_pose[5:8]+0.5,0,1))/np.sqrt(3)
            sim=(1-cos_dist.item())*(1-scaler_dist.item())*(1-scaler_dist2.item())

            # sim = sim ** 2
            # sim = 2 * sim * ((1 - sim) ** 0.3)
            if ref_success and not gen_success:  # and counter2 < self.batch_size:
                superior_A_model_moving_rate.update(0.)
                counter2 += 1
            elif gen_success and not ref_success:  # and counter2 < self.batch_size:
                superior_A_model_moving_rate.update(1.)
                counter2 += 1

            if len(d_pairs)<self.batch_size and not (ref_success and gen_success):
                if  gen_success :#and (grasp_quality[target_index]<max(superior_A_model_moving_rate.val,np.random.random()) ):#min(0.5,np.random.random()):#target_ref_pose[-1].item()>target_generated_pose[-1].item():
                    pass
                # elif ref_success and (grasp_quality[target_index]**2>max(np.random.random(),superior_A_model_moving_rate.val )):
                #     pass
                elif ref_success:
                    # margin=torch.clip(ref_quality-gen_quality,0) if ref_success else torch.clip(gen_quality-ref_quality,0)
                    # margin=1-gen_quality if ref_success else gen_quality
                    margin=importance
                    # margin=target_ref_pose[-1].item() if k>0 else target_generated_pose[-1].item()
                    d_pairs.append((target_index,  k,margin,sim,target_point))


            if len(g_pairs)<self.batch_size and ref_success and not gen_success:
                # if   grasp_quality[target_index]>0.5:#target_ref_pose[-1].item()>target_generated_pose[-1].item():
                #     pass
                # else:
                g_pairs.append((target_index,  k,margin,sim,target_point))
                # print(f'add g')

                # if self.batch_size==1:
                #     d_pairs=[(target_index,  k,margin,sim,target_point)]

            self.tmp_pose_record.append(target_generated_pose.detach().clone())

            superior_pose = target_ref_pose if k > 0 else target_generated_pose

            # self.taxonomies.update(superior_pose.detach().clone())
            # alpha=superior_pose[0:3].detach().clone()
            # self.alpha.update(alpha)
            # self.beta.update(superior_pose[3:5].detach().clone())
            # self.fingers.update(superior_pose[5:8].detach().clone())
            # self.transition.update(superior_pose[8:].detach().clone())

            if self.sampling_centroid is None:
                self.sampling_centroid = superior_pose.detach().clone()
            else:
                self.sampling_centroid = self.sampling_centroid * 0.9 + superior_pose.detach().clone() * 0.1

            if len(d_pairs) == self.batch_size and len(g_pairs) == self.batch_size: break

        # if len(d_pairs) == self.batch_size and len(g_pairs) == self.batch_size:
        #     return  d_pairs, g_pairs,  sampler_samples
        # else:
        if len(all_pairs)>0:
            '''Update dynamic data'''
            self.ch_env.restore_simulation_state()
            synthesised_data_obj=SynthesisedData()
            synthesised_data_obj.obj_ids=self.ch_env.objects
            synthesised_data_obj.obj_poses=self.ch_env.objects_poses

            assert 7*len(self.ch_env.objects)==len(self.ch_env.objects_poses)

            for pair in all_pairs:
                target_index, target_point,pose,importance,grasped_object=pair

                target_point = pc[target_index]

                synthesised_data_obj.target_indexes.append(target_index)
                synthesised_data_obj.grasp_target_points.append(target_point.cpu().numpy())
                synthesised_data_obj.grasp_parameters.append(pose.cpu().numpy())
                synthesised_data_obj.importance.append(importance)
                synthesised_data_obj.grasped_objects.append(grasped_object)


            if self.loaded_synthesised_data is not None:
                synthesised_data_obj.id=self.loaded_synthesised_data.id
                for n in range(len(self.loaded_synthesised_data.target_indexes)):
                    target_index=self.loaded_synthesised_data.target_indexes[n]
                    if target_index in synthesised_data_obj.target_indexes: continue
                    if self.loaded_synthesised_data.grasped_objects[n] is None: continue
                    synthesised_data_obj.target_indexes.append(target_index)
                    synthesised_data_obj.grasp_target_points.append(self.loaded_synthesised_data.grasp_target_points[n])
                    synthesised_data_obj.grasp_parameters.append(self.loaded_synthesised_data.grasp_parameters[n])
                    synthesised_data_obj.importance.append(self.loaded_synthesised_data.importance[n])
                    synthesised_data_obj.grasped_objects.append(self.loaded_synthesised_data.grasped_objects[n])

            if self.loaded_synthesised_data is None:
                if max(synthesised_data_obj.importance) >= 0.5:
                    self.DDM.save_data_point(synthesised_data_obj)
            else:
                ave_impo=sum(synthesised_data_obj.importance) / len(synthesised_data_obj.importance)
                self.Ave_importance.update(ave_impo)
                ave_samples=len(synthesised_data_obj.importance)
                self.Ave_samples_per_scene.update(ave_samples)

                self.DDM.update_old_record(synthesised_data_obj)
                if len(self.DDM)>=10000 and (max(synthesised_data_obj.importance)<0.5 or ave_impo*ave_impo<self.Ave_samples_per_scene.val*self.Ave_importance.val*0.7):
                    self.DDM.low_quality_samples_tracker.append(self.loaded_synthesised_data.id)

            self.skip_rate.update(0.)
        else:
            self.skip_rate.update(1.)
            if self.loaded_synthesised_data is not None:
                self.DDM.low_quality_samples_tracker.append(self.loaded_synthesised_data.id)


        return d_pairs, g_pairs,  sampler_samples

    def prepare_voxels(self,pairs,depth,pc, full_pointcloud,view=False):
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

    def step(self,i):

        self.ch_env.max_obj_per_scene=10

        if len(self.DDM.low_quality_samples_tracker)==0 and  ( ((np.random.rand() < self.skip_rate.val**2) or len(self.DDM)>=10000 ) and len(self.DDM) > 100) or test_mode:

            self.loaded_synthesised_data = self.DDM.load_random_sample()
            self.ch_env.objects=deque(self.loaded_synthesised_data.obj_ids)
            self.ch_env.objects_poses=self.loaded_synthesised_data.obj_poses

            # assert 7*len(self.ch_env.objects)==len(self.ch_env.objects_poses), f'{len(self.ch_env.objects)}, {len(self.ch_env.objects_poses)}'

            # print(Fore.LIGHTMAGENTA_EX,'----------Load saved data point .....................................objects:',self.ch_env.objects,Fore.RESET)
            self.ch_env.reload()
            # print(self.ch_env.objects_poses)

            assert len(self.ch_env.d.qpos) == 3 + 4 + 15 + len(self.ch_env.objects) * 7

        else:
            self.loaded_synthesised_data = None

            self.ch_env.drop_new_obj(selected_index=None, stablize=True)

            nn=random.randint(0, self.ch_env.max_obj_per_scene-len(self.ch_env.objects))

            for k in range(nn):
                self.ch_env.drop_new_obj(selected_index=None, stablize=True)

            assert len(self.ch_env.d.qpos) == 3 + 4 + 15 + len(self.ch_env.objects) * 7


        # self.ch_env.print_state()


        '''get scene perception'''
        depth, pc, floor_mask = self.ch_env.get_scene_preception(view=False)

        # obj_normals=estimate_suction_direction(pc[~floor_mask],view=False,radius=0.01, max_nn=10)
        # approach=np.zeros((600*600,3))
        # approach[:,2]-=1
        # approach[~floor_mask]=-obj_normals
        # approach=approach.transpose().reshape(3,600,600)
        # approach=torch.from_numpy(approach).cuda()

        # view_npy_open3d(pc)
        full_objects_pc = self.ch_env.get_obj_point_clouds(view=False)
        full_pointcloud=np.vstack([pc[floor_mask],full_objects_pc])

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
        # pc, _ = self.ch_env.depth_to_pointcloud(depth.cpu().numpy(), self.ch_env.intr, self.ch_env.extr)
        pc = torch.from_numpy(pc).cuda()
        # view_npy_open3d(pc.cpu().numpy())



        # view_npy_open3d(pc.cpu().numpy())
        # return
        # torch.save(depth, 'depth_ch_tmp')
        # floor_mask = torch.from_numpy(floor_mask).cuda()
        # torch.save(floor_mask, 'floor_mask_ch_tmp')
        # exit()


        latent_vector=torch.randn((1,8,depth.shape[0],depth.shape[1]),device='cuda')

        for k in range(self.iter_per_scene):

            with torch.no_grad():
                self.gan.generator.eval()
                gripper_pose, grasp_quality_logits, grasp_collision_logits ,features,_ = self.gan.generator(
                    depth[None, None, ...],~floor_mask.view(1,1,600,600),latent_vector,detach_backbone=True)
                self.gan.generator.train()

                grasp_quality=logits_to_probs(grasp_quality_logits)
                grasp_collision=logits_to_probs2(grasp_collision_logits)

                n=max(self.tou, self.skip_rate.val)# ** 2
                f = (1 - grasp_quality.detach()).clamp(min=self.skip_rate.val**2)**2

                annealing_factor = f#torch.clamp(f,min=n) #if self.skip_rate.val < 0.7 else n + (1 - n) * f
                print(Fore.LIGHTYELLOW_EX,f'mean_annealing_factor= {annealing_factor.mean()},max_annealing_factor= {annealing_factor.max()},min_annealing_factor= {annealing_factor.min()}, tou={self.tou}, skip rate={self.skip_rate.val}',Fore.RESET)

                self.max_G_norm=max(self.G_grad_norm_MR.val* self.tou**2 + (1 - self.tou**2)*5,5)
                self.max_D_norm= max(self.D_grad_norm_MR.val * self.tou**2 + (1 - self.tou**2)*1,1)

                # if self.tou<0.6:
                #     self.batch_size=1
                #     self.learning_rate=1e-5
                # elif self.tou>0.7:
                #     self.batch_size = 1
                #     self.learning_rate=1e-4

                gripper_pose_ref = ch_pose_interpolation(gripper_pose,
                                                         annealing_factor=annealing_factor,tou=max(self.tou,self.skip_rate.val))  # [b,self.n_param,600,600]
                if self.loaded_synthesised_data is not None:
                    '''inject saved poses'''
                    gripper_pose_ref = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000, self.n_param)
                    for t in range(len(self.loaded_synthesised_data.target_indexes)):
                        index=self.loaded_synthesised_data.target_indexes[t]
                        pose=self.loaded_synthesised_data.grasp_parameters[t]
                        # saved_target_point=self.loaded_synthesised_data.grasp_target_points[t]
                        # target_point = pc[index]

                        # print(f'saved_target_point: {saved_target_point}, target_point: {target_point}')
                        # print(f'saved pose: {pose}')

                        pose=torch.tensor(pose).cuda()
                        gripper_pose_ref[index]=pose

                    # gripper_pose_ref=gripper_pose_ref.permute(1,2,0)
                    # print(f'saved_pose : {gripper_pose_ref[0,:,t]} , pose : {pose}')

                    gripper_pose_ref=gripper_pose_ref.reshape(600,600,self.n_param).permute(2, 0, 1).unsqueeze(0)


                model_b_quality_logits=self.gan.generator.get_grasp_quality(
                    depth[None, None, ...], ~floor_mask.view(1, 1, 600, 600),gripper_pose_ref )
                model_b_quality=logits_to_probs(model_b_quality_logits)

                # gripper_pose_ref[0,0:3]=approach
                if i % int(50) == 0 and i != 0 and k == 0:
                    try:
                        self.export_check_points()
                        self.save_statistics()
                    except Exception as e:
                        print(Fore.RED, str(e), Fore.RESET)
                if i % 10 == 0 and k == 0:
                    self.view_result(gripper_pose, floor_mask)

                self.tmp_pose_record = []
                d_pairs, g_pairs ,sampler_samples= self.sample_contrastive_pairs(pc, floor_mask, gripper_pose,
                                                                                  gripper_pose_ref,
                                                                                  self.tou, grasp_quality.detach(),grasp_collision.detach(),
                                                                                  self.superior_A_model_moving_rate,latent_vector,model_b_quality)
                # print(f'len d {len(d_pairs)}, len g {len(g_pairs)}')

                if synthesizie_only: break

            if test_mode:
                if len(d_pairs)>0 and view:
                    self.prepare_voxels( d_pairs, depth, pc, full_pointcloud,view=view)
                return



            self.tou = 1 - self.superior_A_model_moving_rate.val


            gripper_pose = gripper_pose[0].permute(1, 2, 0).reshape(360000, self.n_param)
            gripper_pose_ref_pixel=gripper_pose_ref
            gripper_pose_ref = gripper_pose_ref[0].permute(1, 2, 0).reshape(360000, self.n_param)

            if len(d_pairs) == self.batch_size:
                print()
                print('------------------------------------------------step_Critic--------------------------------------------------------')
                d_cropped_spheres=self.prepare_voxels( d_pairs, depth, pc, full_pointcloud)
                # d_cropped_spheres=None
                self.step_discriminator(d_cropped_spheres,depth,clean_depth, gripper_pose, gripper_pose_ref, d_pairs,floor_mask,grasp_quality,latent_vector=latent_vector)
                self.print_pairs_info(d_pairs, gripper_pose, gripper_pose_ref)
                print()

            # if sampler_samples==batch_size:
            if len(g_pairs) == self.batch_size:
                print()
                print('------------------------------------------------step_Policy_and_action--------------------------------------------------------')
                g_cropped_spheres=self.prepare_voxels( g_pairs, depth, pc, full_pointcloud)
                # g_cropped_spheres=None
                self.step_generator(g_cropped_spheres,depth,clean_depth, floor_mask, pc, gripper_pose_ref_pixel, g_pairs,latent_vector)
                self.print_pairs_info( g_pairs, gripper_pose, gripper_pose_ref)
                print()
            # elif self.skip_rate.val>0.9:
            elif self.skip_rate.val<0.5:
                print()
                print('------------------------------------------------step_Policy--------------------------------------------------------')
                self.step_generator(None,depth,clean_depth, floor_mask, pc, gripper_pose_ref_pixel, g_pairs,latent_vector)
                print()

            if not ((len(d_pairs)==self.batch_size) or (len(g_pairs)==self.batch_size)) and not test_mode:
                # self.superior_A_model_moving_rate.update(0)
                self.tou = 1 - self.superior_A_model_moving_rate.val
                if k==0:

                    self.ch_env.remove_objects(n=2)
                    break
            # else:
                # self.ch_env.update_obj_info(0.9)

            # else:
            #     self.step_generator_without_sampler(depth, floor_mask, pc,latent_vector)
            # continue



    def view_result(self, gripper_poses,floor_mask):
        with torch.no_grad():
            self.ch_env.save_obj_dict()

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

            self.Ave_samples_per_scene.view()
            self.Ave_importance.view()

            self.G_grad_norm_MR.view()
            self.D_grad_norm_MR.view()

            self.bin_collision_statistics.print()
            self.objects_collision_statistics.print()
            self.collision_statistics.print()

            self.grasp_quality_statistics.print()


            # self.taxonomies.view()
            # self.alpha.view()
            # self.beta.view()
            # self.fingers.view()
            # self.transition.view()

            # self.quat_centers.view()
            # self.fingers_centers.view()
            # self.transition_centers.view()

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

        self.data_tracker.save()


        self.bin_collision_statistics.save()
        self.collision_statistics.save()
        self.grasp_quality_statistics.save()

        torch.save(self.sampling_centroid,self.last_pose_center_path)

        # self.taxonomies.save()
        # self.alpha.save()
        # self.beta.save()
        # self.fingers.save()
        # self.transition.save()
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
            if self.skip_rate.val > 0.8:
                self.batch_size = 1
                self.iter_per_scene = 1#5
                self.ch_env.max_obj_per_scene = 1
            elif self.skip_rate.val < 0.4:
                self.batch_size = 2
                self.iter_per_scene = 1
                self.ch_env.max_obj_per_scene = int(7*np.random.rand())
            # cuda_memory_report()
            # self.batch_size = 1

            if args.catch_exceptions:
                try:
                    self.step(i)
                    pi.step(i)
                except Exception as e:
                    print(Fore.RED, str(e), Fore.RESET)
                    traceback.print_exc()
                    torch.cuda.empty_cache()
                    # self.ch_env.update_obj_info(0.1)
                    self.ch_env.remove_objects(n=self.ch_env.max_obj_per_scene)
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
        default=1e-4,
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
