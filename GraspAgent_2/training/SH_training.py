import os
import numpy as np
from colorama import Fore
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from GraspAgent_2.hands_config.sh_config import fingers_range, fingers_min
from GraspAgent_2.model.SH_GAN import SH_G, SH_D, SH_model_key
from GraspAgent_2.sim_dexee.Shadow_hand_env import grasp_env, quat_rotate_vector
from Online_data_audit.data_tracker import gripper_grasp_tracker, DataTracker
from check_points.check_point_conventions import GANWrapper
from lib.IO_utils import custom_print
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data2
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from records.training_satatistics import TrainingTracker, MovingRate
from GraspAgent_2.training.sample_random_grasp import  sh_pose_interpolation
import torch

iter_per_scene = 1

batch_size = 4
freeze_backbone = False
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


def balanced_sampling(values, mask=None, exponent=2.0, balance_indicator=1.0):
    with torch.no_grad():
        max_ = values.max().item()
        min_ = values.min().item()
        range_ = max_ - min_

        if not range_>0.:
            selection_probability=torch.rand_like(values)
        else:
            pivot_point = np.sqrt(np.abs(balance_indicator)) * np.sign(balance_indicator)
            xa=(1-values)* pivot_point
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

        self.update_record_path=SH_model_key+'_update_record'
        if os.path.exists(self.update_record_path):
            self.update_record = torch.load(self.update_record_path).cuda()
        else: self.update_record = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0., 0., 0., 0., 0., 0., 0.],
                                                        device='cuda')
        self.tmp_pose_record=[]

        self.last_pose_center_path=SH_model_key+'_pose_center'
        if os.path.exists(self.last_pose_center_path):
            self.sampling_centroid = torch.load(self.last_pose_center_path).cuda()
        else: self.sampling_centroid = torch.tensor([0, 1, 0, 0, 0, 0, 0, -1., -1., -1., -1., -1., -1., 0.1],
                                                        device='cuda')
        root_dir = os.getcwd()  # current working directory
        self.sh_env = grasp_env(obj_nums_in_scene=1,root =root_dir+ "/GraspAgent_2/sim_dexee/shadow_dexee/",max_obj_per_scene=3)



    def initialize(self, n_samples=None):
        self.n_samples = n_samples

        '''Moving rates'''

        self.gradient_moving_rate = MovingRate(SH_model_key + '_gradient', decay_rate=0.01, initial_val=1000.)

        self.moving_collision_rate = MovingRate(SH_model_key + '_collision', decay_rate=0.01,
                                                initial_val=1.)
        self.relative_sampling_timing = MovingRate(SH_model_key + '_relative_sampling_timing',
                                                   decay_rate=0.01,
                                                    initial_val=1.)
        self.superior_A_model_moving_rate = MovingRate(SH_model_key + '_superior_A_model',
                                                       decay_rate=0.01,
                                                       initial_val=0.)


        '''initialize statistics records'''

        self.bin_collision_statistics = TrainingTracker(name=SH_model_key + '_bin_collision',
                                                        track_label_balance=True)
        self.objects_collision_statistics = TrainingTracker(name=SH_model_key + '_objects_collision',
                                                            track_label_balance=True)




        self.gripper_sampler_statistics = TrainingTracker(name=SH_model_key + '_gripper_sampler',
                                                          track_label_balance=False)

        self.grasp_quality_statistics = TrainingTracker(name=SH_model_key + '_grasp_quality',
                                                        track_label_balance=True)

        self.critic_statistics = TrainingTracker(name=SH_model_key + '_critic',
                                                  track_label_balance=False)

        self.data_tracker = DataTracker(name=gripper_grasp_tracker)



    def prepare_model_wrapper(self):
        '''load  models'''
        gan = GANWrapper(SH_model_key, SH_G, SH_D)
        gan.ini_models(train=True)

        gan.critic_adam_optimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.999)
        # gan.critic_sgd_optimizer(learning_rate=self.learning_rate*5,momentum=0.)
        gan.generator_adam_optimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999)
        # gan.generator_sgd_optimizer(learning_rate=self.learning_rate*10,momentum=0.)

        return gan

    def step_discriminator(self,depth,   gripper_pose, gripper_pose_ref ,pairs ):
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
        score = self.gan.critic(depth[None,None,...], generated_grasps_stack,pairs,detach_backbone=freeze_backbone)

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

            label = ref_scores_[j].squeeze()
            pred_ = gen_scores_[j].squeeze()

            # if ( label-pred_) * k>1:
            #     print(Fore.LIGHTRED_EX)
            #     '''curcclium loss'''
            #     l = (torch.clamp((pred_ - label) * k*-1 +1 , 0.) ** 2)
            # else:
            #     print(Fore.LIGHTYELLOW_EX)
            #     l = (torch.clamp((pred_ - label) * k + 1, 0.) ** 2)

            loss+=(torch.clamp((pred_ - label) * k + 1, 0.)**1)/batch_size
            # loss+=l/batch_size
        # loss=self.RGAN_D_loss(pairs,gen_scores_,ref_scores_)
        loss.backward()

        self.critic_statistics.loss=loss.item()
        self.gan.critic_optimizer.step()

        self.gan.critic.zero_grad()
        self.gan.critic_optimizer.zero_grad()

        print(  Fore.LIGHTYELLOW_EX,f'd_loss={loss.item()}',
              Fore.RESET)

    def get_generator_loss(self, depth, gripper_pose, gripper_pose_ref, pairs):

        gripper_pose = gripper_pose[0].permute(1, 2, 0).reshape(360000, 14)

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
        critic_score = self.gan.critic(depth[None,None,...], generated_grasps_stack,pairs,detach_backbone=True)

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

            target_generated_pose = gripper_pose[target_index].detach()
            target_ref_pose = gripper_pose_ref[target_index].detach()

            label = ref_scores_[j].squeeze()
            pred_ = gen_scores_[j].squeeze()

            if k==1:
                print(Fore.LIGHTCYAN_EX,f'{target_ref_pose.cpu()} {target_generated_pose.cpu().detach()} s=[{label.cpu()},{pred_.cpu()}] ',Fore.RESET)
            else:
                print(Fore.LIGHTGREEN_EX,f'{target_ref_pose.cpu()} {target_generated_pose.cpu().detach()} s=[{label.cpu()},{pred_.cpu()}] ',Fore.RESET)
            print()

            w=1 if k>0 else 0
            loss += ((torch.clamp( label - pred_, 0.)*w) **2)/ batch_size

        return loss


    def step_generator(self,depth,floor_mask,pc,gripper_pose_ref,pairs,tou):
        '''zero grad'''
        self.gan.critic.zero_grad(set_to_none=True)
        self.gan.generator.zero_grad(set_to_none=True)
        # self.gan.critic.eval()

        '''generated grasps'''
        gripper_pose, grasp_quality,  grasp_collision = self.gan.generator(depth[None, None, ...],
                                                                                                detach_backbone=freeze_backbone)

        gripper_pose_PW = gripper_pose[0].permute(1, 2, 0).reshape(360000,14)
        object_collision = grasp_collision[0,0].reshape(-1)
        floor_collision = grasp_collision[0,1].reshape(-1)

        grasp_quality = grasp_quality[0,0].reshape(-1)

        '''loss computation'''
        gripper_collision_loss = torch.tensor(0., device=depth.device)
        gripper_quality_loss_ = torch.tensor(0., device=depth.device)


        if tou<0.5 or True:

            for k in range(batch_size):
                '''gripper-object collision'''
                gripper_target_index = balanced_sampling(object_collision.detach(),
                                                         mask=~floor_mask.detach(),
                                                         exponent=10.0,
                                                         balance_indicator=self.objects_collision_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = object_collision[gripper_target_index].squeeze()
                gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

                collision=self.check_collision(gripper_target_point,gripper_target_pose,view=False)

                label = torch.ones_like(gripper_prediction_) if collision else torch.zeros_like(gripper_prediction_)

                object_collision_loss = bce_loss(gripper_prediction_, label)

                with torch.no_grad():
                    self.objects_collision_statistics.loss = object_collision_loss.item()
                    self.objects_collision_statistics.update_confession_matrix(label.detach(),
                                                                          gripper_prediction_.detach())

                gripper_collision_loss+=object_collision_loss/batch_size


            for k in range(batch_size):
                '''gripper-bin collision'''
                gripper_target_index = balanced_sampling(floor_collision.detach(),
                                                         mask=~floor_mask.detach(),
                                                         exponent=10.0,
                                                         balance_indicator=self.bin_collision_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = floor_collision[gripper_target_index].squeeze()
                gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

                collision = self.check_collision(gripper_target_point, gripper_target_pose, view=False)

                label = torch.ones_like(gripper_prediction_) if collision else torch.zeros_like(gripper_prediction_)

                floor_collision_loss = bce_loss(gripper_prediction_, label)

                with torch.no_grad():
                    self.bin_collision_statistics.loss = floor_collision_loss.item()
                    self.bin_collision_statistics.update_confession_matrix(label.detach(),
                                                                               gripper_prediction_.detach())

                gripper_collision_loss += floor_collision_loss / batch_size

            for k in range(batch_size):
                '''grasp quality'''
                '''gripper-bin collision'''
                gripper_target_index = balanced_sampling(grasp_quality.detach(),
                                                         mask=~floor_mask.detach(),
                                                         exponent=10.0,
                                                         balance_indicator=self.grasp_quality_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = grasp_quality[gripper_target_index].squeeze()
                gripper_target_pose = gripper_pose_PW[gripper_target_index].detach()

                grasp_success,in_scope = self.evaluate_grasp(gripper_target_point, gripper_target_pose, view=False)

                label = torch.ones_like(gripper_prediction_) if grasp_success else torch.zeros_like(gripper_prediction_)

                grasp_quality_loss = bce_loss(gripper_prediction_, label)**2

                with torch.no_grad():
                    self.grasp_quality_statistics.loss = grasp_quality_loss.item()
                    self.grasp_quality_statistics.update_confession_matrix(label.detach(),
                                                                           gripper_prediction_.detach())

                gripper_quality_loss_ += grasp_quality_loss / batch_size

        gripper_sampling_loss = self.get_generator_loss(
            depth,  gripper_pose, gripper_pose_ref,
            pairs)

        assert not torch.isnan(gripper_sampling_loss).any(), f'{gripper_sampling_loss}'

        print(Fore.LIGHTYELLOW_EX,
              f'g_loss={gripper_sampling_loss.item()}',
              Fore.RESET)

        loss = gripper_sampling_loss*30+gripper_collision_loss+gripper_quality_loss_ #+ background_detection_loss * 30 + gripper_collision_loss + 10 * gripper_quality_loss_

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
            gripper_pose, grasp_quality, grasp_collision = self.gan.generator(depth[None, None, ...],
                                                                              detach_backbone=True)
            gripper_pose = gripper_pose[0].permute(1, 2, 0).reshape(360000, 14)

            for j in range(batch_size):
                index=pairs[j][0]
                old_pose=self.tmp_pose_record[j]
                new_pose=gripper_pose[index]
                diff=torch.abs(old_pose-new_pose)*100
                self.update_record=self.update_record*0.99+diff*0.01


    def check_collision(self,target_point,target_pose,view=False):
        with torch.no_grad():
            quat, fingers, shifted_point = self.process_pose(target_point, target_pose, view=view)

        return self.sh_env.check_collision(hand_pos=shifted_point,hand_quat=quat,hand_fingers=fingers,view=False)

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

    def evaluate_grasp(self, target_point, target_pose, view=False,shake_intensity=None):

        with torch.no_grad():
            quat,fingers,shifted_point=self.process_pose(target_point, target_pose, view=view)

            in_scope, grasp_success, contact_with_obj, contact_with_floor = self.sh_env.check_graspness(
                hand_pos=shifted_point, hand_quat=quat, hand_fingers=fingers,
               view=view,shake_intensity=shake_intensity)

            if grasp_success is not None:
                if grasp_success and not contact_with_obj and not contact_with_floor:
                    return True, in_scope

        return False, in_scope


    def sample_contrastive_pairs(self,pc,  floor_mask, gripper_pose, gripper_pose_ref,
                                 sampling_centroid, batch_size, annealing_factor, grasp_quality,
                                 superior_A_model_moving_rate):

        pairs = []

        selection_mask = ~floor_mask
        grasp_quality=grasp_quality[0,0].reshape(-1)
        gripper_pose_PW = gripper_pose.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000,14)
        gripper_pose_ref_PW = gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :].reshape(360000,14)

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

        gamma_dive = norm_((1.001 - F.cosine_similarity(gripper_pose_PW,
                                                        sampling_centroid[None, :], dim=-1) ) /2 ,1)
        gamma_dive *= norm_((1.001 - F.cosine_similarity(gripper_pose_ref_PW,
                                                        sampling_centroid[None, :], dim=-1) ) /2 ,1)

        # selection_p = compute_sampling_probability(sampling_centroid, gripper_pose_ref_PW, gripper_pose_PW, pc, bin_mask,grasp_quality)
        # selection_p = torch.rand_like(gripper_pose_PW[:, 0])
        selection_p = gamma_dive**(1/2)

        avaliable_iterations = selection_mask.sum()
        if avaliable_iterations<3: return False, None,None

        n = int(min(max_n, avaliable_iterations))

        counter = 0

        t = 0
        while t < n:

            dist = MaskedCategorical(probs=selection_p, mask=selection_mask)

            target_index = dist.sample().item()

            selection_mask[target_index] *= False
            avaliable_iterations -= 1
            target_point = pc[target_index]

            target_generated_pose = gripper_pose_PW[target_index]
            target_ref_pose = gripper_pose_ref_PW[target_index]

            ref_success,ref_in_scope = self.evaluate_grasp(target_point,target_ref_pose,view=False)

            if not ref_success or not ref_in_scope:
                q=torch.clamp(grasp_quality[target_index],0,1).item()
                if np.random.rand()>(1-q)*annealing_factor**2:continue

            gen_success,gen_in_scope = self.evaluate_grasp(target_point, target_generated_pose,view=False)

            t += 1
            if ref_success == gen_success:
                if ref_success:
                    print('                                                         ...')
                continue
            if gen_success and not gen_in_scope:continue

            print(f'ref_success={ref_success}, gen_success={gen_success}')
            print(f'ref_in_scope={ref_in_scope}, gen_in_scope={gen_in_scope}')

            k=1 if ref_success and not gen_success else -1
            if k == 1:
                superior_A_model_moving_rate.update(0.)
            elif k == -1:
                superior_A_model_moving_rate.update(1.)

            counter += 1
            t = 0
            hh = (counter / batch_size) ** 2
            n = int(min(hh * max_n + n, avaliable_iterations))

            pairs.append((target_index,  k))

            self.tmp_pose_record.append(target_generated_pose.detach().clone())

            superior_pose = target_ref_pose if k > 0 else target_generated_pose
            if sampling_centroid is None:
                sampling_centroid = superior_pose.detach().clone()
            else:
                sampling_centroid = sampling_centroid * 0.999 + superior_pose.detach().clone() * 0.001

            if counter == batch_size: break


        if counter == batch_size:
            return True, pairs, sampling_centroid
        else:
            return False, pairs, sampling_centroid



    def step(self,i):
        self.sh_env.drop_new_obj()


        '''get scene perception'''
        depth, pc, floor_mask = self.sh_env.get_scene_preception(view=False)

        depth = torch.from_numpy(depth).cuda()  # [600.600]
        floor_mask = torch.from_numpy(floor_mask).cuda()


        for k in range(iter_per_scene):

            with torch.no_grad():
                gripper_pose, grasp_quality, grasp_collision = self.gan.generator(
                    depth[None, None, ...], detach_backbone=True)

                x = self.superior_A_model_moving_rate.val

                tou = 1 - x
                gripper_pose_ref = sh_pose_interpolation(gripper_pose, self.sampling_centroid,
                                                         annealing_factor=tou)  # [b,14,600,600]

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
                counted, pairs, sampling_centroid = self.sample_contrastive_pairs(pc, floor_mask, gripper_pose,
                                                                                  gripper_pose_ref,
                                                                                  self.sampling_centroid, batch_size,
                                                                                  tou, grasp_quality.detach(),
                                                                                  self.superior_A_model_moving_rate)
                if not counted: continue

                self.sampling_centroid = sampling_centroid


            gripper_pose = gripper_pose[0].permute(1, 2, 0).reshape(360000, 14)
            gripper_pose_ref = gripper_pose_ref[0].permute(1, 2, 0).reshape(360000, 14)

            self.step_discriminator(depth, gripper_pose, gripper_pose_ref, pairs)

            self.step_generator(depth, floor_mask, pc, gripper_pose_ref, pairs, tou)
            # continue


    def view_result(self, gripper_poses,floor_mask):
        with torch.no_grad():

            print('Center pos: ',self.sampling_centroid.cpu().numpy())

            print('Parameters update rate: ',self.update_record)

            cuda_memory_report()

            values = gripper_poses[0].permute(1, 2, 0).reshape(360000, 14).detach()  # .cpu().numpy()
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


            self.gradient_moving_rate.view()

    def save_statistics(self):
        self.moving_collision_rate.save()
        self.relative_sampling_timing.save()
        self.superior_A_model_moving_rate.save()

        self.critic_statistics.save()
        self.gripper_sampler_statistics.save()

        self.data_tracker.save()

        self.gradient_moving_rate.save()

        self.bin_collision_statistics.save()
        self.objects_collision_statistics.save()
        self.grasp_quality_statistics.save()

        torch.save(self.sampling_centroid,self.last_pose_center_path)
        torch.save(self.update_record,self.update_record_path)



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
            try:
                self.step(i)
                pi.step(i)
            except Exception as e:
                print(Fore.RED, str(e), Fore.RESET)
                torch.cuda.empty_cache()
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
        Train_grasp_GAN.begin(iterations=30)


if __name__ == "__main__":
    train_N_grasp_GAN(n=10000)
