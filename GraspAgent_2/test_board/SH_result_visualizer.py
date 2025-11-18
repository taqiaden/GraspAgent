import os
import numpy as np
import trimesh
from colorama import Fore
from filelock import FileLock
from torch import nn

from GraspAgent_2.hands_config.sh_config import fingers_range, fingers_min
from GraspAgent_2.model.SH_GAN import SH_G, SH_D, SH_model_key
from GraspAgent_2.sim_dexee.Shadow_hand_env import grasp_env, quat_rotate_vector
import torch.nn.functional as F

from check_points.check_point_conventions import GANWrapper
from lib.IO_utils import custom_print
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data2
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from records.training_satatistics import  MovingRate

from GraspAgent_2.training.sample_random_grasp import  sh_pose_interpolation

lock = FileLock("file.lock")
iter_per_scene = 1

batch_size = 2
max_n = 50

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


class TrainGraspGAN:
    def __init__(self, n_samples=None, epochs=1, learning_rate=5e-5):

        self.n_samples = n_samples
        self.size = n_samples
        self.epochs = epochs
        self.learning_rate = learning_rate

        '''model wrapper'''
        self.gan = self.prepare_model_wrapper()
        # self.ref_generator = self.initialize_ref_generator()

        '''Moving rates'''
        self.moving_collision_rate = None
        self.moving_firmness = None
        self.moving_out_of_scope = None
        self.relative_sampling_timing = None
        self.superior_A_model_moving_rate = None

        self.bin_collision_statistics = None
        self.objects_collision_statistics = None
        self.background_detector_statistics = None
        self.grasp_quality_statistics=None

        '''initialize statistics records'''
        self.gripper_sampler_statistics = None
        self.critic_statistics = None
        self.critic_statistics = None
        self.data_tracker = None

        self.last_pose_center_path=SH_model_key+'_pose_center'

        if os.path.exists(self.last_pose_center_path):
            self.sampling_centroid = torch.load(self.last_pose_center_path).cuda()
        else: self.sampling_centroid = torch.tensor([0, 1, 0, 0, 0, 0, 0, -1., -1., -1., -1., -1., -1., 0.1],
                                                        device='cuda')
        # print(self.sampling_centroid )
        # exit()
        root_dir = os.getcwd()  # current working directory
        self.sh_env = grasp_env(obj_nums_in_scene=1,root =root_dir+ "/GraspAgent_2/sim_dexee/shadow_dexee/")



    def initialize(self, n_samples=None):
        self.n_samples = n_samples


        self.superior_A_model_moving_rate = MovingRate(SH_model_key + '_superior_A_model',
                                                       decay_rate=0.01,
                                                       initial_val=0.)


    def prepare_model_wrapper(self):
        '''load  models'''
        gan = GANWrapper(SH_model_key, SH_G, SH_D)
        gan.ini_models(train=True)

        return gan

    def process_pose(self,target_point, target_pose, view=False):
        target_pose_ = target_pose.clone()
        target_point_ = np.copy(target_point)

        quat = target_pose_[:4].cpu().tolist()

        target_pose_[4:4+9]/=2
        target_pose_[4:4 + 9]+=0.5

        finger_list = []
        range_=fingers_range
        min_=fingers_min
        for k in range(3):
            finger_list.append(min_[0]+(target_pose_[4 + k]*range_[0]))
            finger_list.append(min_[1]+(target_pose_[4 + k + 3]*range_[1]))
            finger_list.append(min_[2]+(target_pose_[4 + k + 6]*range_[2]))
            finger_list.append(target_pose_[
                                   4 + k + 6] * 0)  # this parameter is coupled with the previous so the value here is just to hold position

        fingers = torch.stack(finger_list).cpu().tolist()

        transition = target_pose_[-1].cpu().numpy() / 10

        approach = quat_rotate_vector(np.array(quat), np.array([0, 0, 1]))
        projected_transition = approach * transition

        shifted_point = (target_point_ + projected_transition).tolist()

        if view:
            print(quat)
            print(fingers)
            print(transition)
            print(projected_transition)
            print(shifted_point)

        return quat,fingers,shifted_point

    def evaluate_grasp(self, target_point, target_pose, view=False):

        with torch.no_grad():
            quat,fingers,shifted_point=self.process_pose(target_point, target_pose, view=view)

            in_scope, grasp_success, contact_with_obj, contact_with_floor = self.sh_env.check_graspness(
                hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers,
                view=view)

            if grasp_success is not None:
                if grasp_success and not contact_with_obj and not contact_with_floor:
                    return True, in_scope

        return False, in_scope

    def sample_contrastive_pairs(self,pc,  floor_mask, gripper_pose, gripper_pose_ref ,grasp_quality,grasp_collision):

        selection_mask = (~floor_mask)# & (grasp_collision[0,0].reshape(-1)<0.7)& (grasp_collision[0,1].reshape(-1)<0.7)

        gripper_pose_PW = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000,14)
        gripper_pose_ref_PW = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000,14)
        # selection_p=torch.clamp(grasp_quality[0,0].reshape(-1),0,1)+0.01
        # selection_p = compute_sampling_probability(sampling_centroid, gripper_pose_ref_PW, gripper_pose_PW, pc, bin_mask,grasp_quality)
        selection_p = torch.rand_like(gripper_pose_PW[:, 0])

        avaliable_iterations = selection_mask.sum()
        if avaliable_iterations<4 : return
        for i in range(3):


            dist = MaskedCategorical(probs=selection_p, mask=selection_mask)
            target_index = dist.sample().item()



            # selection_mask[target_index] *= False
            # avaliable_iterations -= 1
            target_point = pc[target_index]

            target_generated_pose = gripper_pose_PW[target_index]
            target_ref_pose = gripper_pose_ref_PW[target_index]


            gen_success,gen_in_scope = self.evaluate_grasp(target_point, target_generated_pose,view=False)
            print('gen_success:',gen_success)


            if gen_success:
                r=self.evaluate_grasp(target_point, target_generated_pose,view=True)
            #     print('gen---->',r)
                # if gen_success!=r[0]:
                #     gen_success, gen_in_scope = self.evaluate_grasp(target_point, target_generated_pose, view=False)
                #     print(gen_success)
                #     exit()


            ref_success,ref_in_scope = self.evaluate_grasp(target_point,target_ref_pose,view=False)
            print('ref_success:',ref_success)
            if ref_success:
                r=self.evaluate_grasp(target_point, target_ref_pose,view=True)
            #     print('ref------>',r)
                # if ref_success!=r[0]:
                #     ref_success, ref_in_scope = self.evaluate_grasp(target_point, target_ref_pose, view=False)
                #     print(ref_success)
                #     exit()

    def begin(self,iterations=1000):

        pi = progress_indicator('Begin new training round: ', max_limit=iterations)
        gripper_pose = None
        for i in range(iterations):
            # self.sh_env.set_new_scene()
            '''sample initial object pose'''
            obj_pose = [np.random.rand()-0.5, np.random.rand()-0.5, 0.]
            # obj_quat = [1, 0, 0, 0]
            obj_quat = torch.randn((4,))
            obj_quat[[1,2]]*=0
            obj_quat = obj_quat / torch.norm(obj_quat)
            obj_quat=obj_quat.tolist()
            # obj_quat = torch.randn(4)
            # obj_quat=F.normalize(obj_quat,dim=0).tolist()


            '''get scene perception'''
            depth,pc,floor_mask=self.sh_env.get_scene_preception(view=False)
            # continue
            # pc[:,0]*=-1

            depth=torch.from_numpy(depth).cuda() #[600.600]
            floor_mask=torch.from_numpy(floor_mask).cuda()


            pi.step(i)

            for k in range(iter_per_scene):

                with torch.no_grad():
                    gripper_pose,grasp_quality,grasp_collision = self.gan.generator(
                        depth[None,None,...], detach_backbone=True)


                    x=self.superior_A_model_moving_rate.val

                    tou=1-x

                    gripper_pose_ref = sh_pose_interpolation(gripper_pose,self.sampling_centroid, annealing_factor=tou) # [b,14,600,600]



                    self.sample_contrastive_pairs(pc,floor_mask, gripper_pose, gripper_pose_ref,grasp_quality,grasp_collision)




def train_N_grasp_GAN(n=1):
    lr = 5e-5
    Train_grasp_GAN = TrainGraspGAN(n_samples=None, learning_rate=lr)
    torch.cuda.empty_cache()
    # torch.autograd.set_detect_anomaly(True)

    for i in range(n):
        # try:
            cuda_memory_report()
            Train_grasp_GAN.initialize(n_samples=None)
            Train_grasp_GAN.begin(iterations=30)
        # except Exception as e:
        #     print(Fore.RED, str(e), Fore.RESET)
        #     del Train_grasp_GAN
        #     torch.cuda.empty_cache()
        #     Train_grasp_GAN = TrainGraspGAN(n_samples=None, learning_rate=lr)

    # del Train_grasp_GAN


if __name__ == "__main__":
    # grasp_quality_statistics = TrainingTracker(name=SH_model_key + '_grasp_quality',
    #                                                 track_label_balance=True)
    train_N_grasp_GAN(n=10000)
