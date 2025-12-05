import os
import numpy as np
from colorama import Fore
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from GraspAgent_2.model.CH_model2 import CH_model_key, CH_D, CH_G
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
        self.n_param=8

        self.last_pose_center_path=CH_model_key+'_pose_center'
        if os.path.exists(self.last_pose_center_path):
            self.sampling_centroid = torch.load(self.last_pose_center_path).cuda()
            if self.sampling_centroid.shape!=self.n_param: self.sampling_centroid = torch.tensor([0, 1, 0, 0, 0.5,0.5,0.5,  0.],
                                                        device='cuda')
        else: self.sampling_centroid = torch.tensor([0, 1, 0, 0, 0.5,0.5,0.5,  0.],
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


        return gan


    def check_collision(self,target_point,target_pose,view=False):
        with torch.no_grad():
            quat, fingers, shifted_point = self.process_pose(target_point, target_pose, view=view)

        return self.ch_env.check_collision(hand_pos=shifted_point,hand_quat=quat,hand_fingers=None,view=False)

    def process_pose(self,target_point, target_pose, view=False):
        target_pose_ = target_pose.clone()
        target_point_ = np.copy(target_point)

        quat = target_pose_[:4].cpu().tolist()

        fingers = torch.clip(target_pose_[4:4+3],0,1).cpu().tolist()

        transition = target_pose_[4+3:4+3+1].cpu().numpy() / 100

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

    def evaluate_grasp(self, target_point, target_pose, view=False,hard_level=0):

        with torch.no_grad():
            quat,fingers,shifted_point=self.process_pose(target_point, target_pose, view=view)

            if view:
                in_scope, grasp_success, contact_with_obj, contact_with_floor = self.ch_env.view_grasp(
                    hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers,
                   view=view,hard_level=hard_level)
            else:
                in_scope, grasp_success, contact_with_obj, contact_with_floor = self.ch_env.check_graspness(
                    hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers,
                    view=view, hard_level=hard_level)

            initial_collision=contact_with_obj or contact_with_floor

            # print('in_scope, grasp_success, contact_with_obj, contact_with_floor :',in_scope, grasp_success, contact_with_obj, contact_with_floor )

            if grasp_success is not None:
                if grasp_success and not contact_with_obj and not contact_with_floor:
                    return True and in_scope,initial_collision

        return False, initial_collision


    def sample_contrastive_pairs(self,pc,  floor_mask, gripper_pose, gripper_pose_ref, grasp_quality ,grasp_collision):

        pairs = []

        grasp_quality=grasp_quality[0,0].reshape(-1)
        obj_collision=grasp_collision[0,0].reshape(-1)
        floor_collision=grasp_collision[0,1].reshape(-1)
        selection_mask = (~floor_mask) & (floor_collision<0.5) & (obj_collision<0.5)


        gripper_pose_PW = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000,self.n_param)
        clipped_gripper_pose_PW=gripper_pose_PW.clone()
        clipped_gripper_pose_PW[:,4:4+3]=torch.clip(clipped_gripper_pose_PW[:,4:4+3],0,1)
        gripper_pose_ref_PW = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000,self.n_param)

        # grasp_quality = grasp_quality.permute(0, 2, 3, 1)[0, :, :, 0][mask]
        # max_ = grasp_quality.max()
        # min_ = grasp_quality.min()
        # grasp_quality = (grasp_quality - min_) / (max_ - min_)
        # def norm_(gamma ,expo_=1.0,min=0.01):
        #     gamma = (gamma - gamma.min()) / (
        #             gamma.max() - gamma.min())
        #     gamma = gamma ** expo_
        #     gamma=torch.clamp(gamma,min)
        #     return gamma

        # gamma_dive = norm_((1.001 - F.cosine_similarity(clipped_gripper_pose_PW,
        #                                                 sampling_centroid[None, :], dim=-1) ) /2 ,1)
        # gamma_dive *= norm_((1.001 - F.cosine_similarity(gripper_pose_ref_PW,
        #                                                 sampling_centroid[None, :], dim=-1) ) /2 ,1)

        # selection_p = compute_sampling_probability(sampling_centroid, gripper_pose_ref_PW, gripper_pose_PW, pc, bin_mask,grasp_quality)
        # selection_p = torch.rand_like(gripper_pose_PW[:, 0])
        # selection_p = gamma_dive**(1/2)
        selection_p=grasp_quality.clone()#*(1-obj_collision.clone())*(1-floor_collision.clone())
        selection_p=torch.clip(selection_p,0.001)**2

        avaliable_iterations = selection_mask.sum()
        if avaliable_iterations<3: return False, None,None,None

        n = int(min(max_n, avaliable_iterations))

        counter = 0

        sampler_samples=0

        t = 0
        while t < 1:
            t += 1

            dist = MaskedCategorical(probs=selection_p, mask=selection_mask)

            target_index = dist.sample().item()

            selection_mask[target_index] *= False
            avaliable_iterations -= 1
            target_point = pc[target_index]

            target_generated_pose = gripper_pose_PW[target_index]
            target_ref_pose = gripper_pose_ref_PW[target_index]

            margin=1.
            # print('ref')
            # ref_success ,ref_initial_collision= self.evaluate_grasp(target_point,target_ref_pose,view=True,shake_intensity=0.002)
            print('gen')
            gen_success,gen_initial_collision = self.evaluate_grasp(target_point, target_generated_pose,view=False,hard_level=0.5)
            print(Fore.LIGHTCYAN_EX,gen_success,gen_initial_collision,Fore.RESET )
            gen_success,gen_initial_collision = self.evaluate_grasp(target_point, target_generated_pose,view=True,hard_level=0.5)
            print(Fore.LIGHTCYAN_EX,gen_success,gen_initial_collision,Fore.RESET )
    def step(self,i):
        self.ch_env.drop_new_obj(stablize=np.random.random()>0.5)

        '''get scene perception'''
        depth, pc, floor_mask = self.ch_env.get_scene_preception(view=False)
        # return

        depth = torch.from_numpy(depth).cuda()  # [600.600]
        floor_mask = torch.from_numpy(floor_mask).cuda()

        latent_mask = (torch.rand_like(depth) > 0.5).float()
        for k in range(iter_per_scene):

            with torch.no_grad():
                gripper_pose, grasp_quality_logits, grasp_collision_logits = self.gan.generator(
                    depth[None, None, ...], ~floor_mask.view(1, 1, 600, 600), latent_mask[None, None, ...],
                    detach_backbone=True)

                grasp_collision=F.sigmoid(grasp_collision_logits)

                grasp_quality = F.sigmoid(grasp_quality_logits)
                f = self.grasp_quality_statistics.accuracy * ((1 - grasp_quality.detach()) ** 2) + (
                            1 - self.grasp_quality_statistics.accuracy)
                annealing_factor = self.tou * f
                print(f'mean_annealing_factor= {annealing_factor.mean()}, tou={self.tou}')

                gripper_pose_ref = ch_pose_interpolation(gripper_pose, self.sampling_centroid,
                                                         annealing_factor=annealing_factor,quat_centers=self.quat_centers.centers,finger_centers=self.fingers_centers.centers)  # [b,self.n_param,600,600]



                self.tmp_pose_record = []
                self.sample_contrastive_pairs(pc, floor_mask, gripper_pose,
                                                              gripper_pose_ref,
                                                                grasp_quality.detach(),grasp_collision )



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
