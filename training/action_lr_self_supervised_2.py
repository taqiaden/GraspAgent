import copy
import os

import numpy as np
import torch
from colorama import Fore
from filelock import FileLock
from torch import nn
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
from training.learning_objectives.gripper_collision import gripper_collision_loss, evaluate_grasps3
from training.learning_objectives.shift_affordnace import shift_affordance_loss
from training.learning_objectives.suction_seal import suction_seal_loss
from visualiztion import view_o3d, view_npy_open3d

detach_backbone=False
generator_exponent = 2.0
discriminator_exponent = 2.0
firmness_exponent=2.0
lock = FileLock("file.lock")

training_buffer = online_data2()
training_buffer.main_modality=training_buffer.depth

bce_loss=nn.BCELoss()

balanced_bce_loss=BalancedBCELoss()
print=custom_print

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)
cache_name='normals'

def suction_sampler_loss(pc,target_normal,file_index):
    file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
    if os.path.exists(file_path):
        labels = load_pickle(file_path)
    else:
        labels = estimate_suction_direction(pc,view=False)  # inference time on local computer = 1.3 s        if file_index is not None:
        file_path = cache_dir + cache_name + '/' + file_index + '.pkl'
        save_pickle(file_path, labels)

    # labels = estimate_suction_direction(pc, view=False)  # inference time on local computer = 1.3 s
    labels = torch.from_numpy(labels).to('cuda')

    return ((1 - cos(target_normal, labels.squeeze())) ** 2).mean()


def model_dependent_sampling(pc,model_predictions,model_max_score,model_score_range,mask=None,maximum_iterations=10000,probability_exponent=2.0,balance_indicator=1.0,random_sampling_probability=0.003):
    for i in range(maximum_iterations):
        if mask is None:
            target_index = np.random.randint(0, pc.shape[0])
        else:
            idx_nonzero,=np.nonzero(mask)
            target_index=np.random.choice(idx_nonzero)
        if np.random.random() <  random_sampling_probability:break
        prediction_ = model_predictions[target_index]
        pivot_point=np.sqrt(np.abs(balance_indicator))*np.sign(balance_indicator)
        xa=((model_max_score - prediction_).item() / model_score_range) * pivot_point
        selection_probability = ((1-pivot_point)/2 + xa+0.5*(1-abs(pivot_point)))
        selection_probability=selection_probability**probability_exponent
        if np.random.random() < selection_probability: break
    else:
        return np.random.randint(0, pc.shape[0])
    return target_index

def critic_loss(c_,s_,f_,prediction_,label_):
    if  s_[0] > 0:
        if c_[1] + s_[1] == 0:
            return (torch.clamp(prediction_ - label_ + 1., 0.)**discriminator_exponent), True

        elif s_[1] == 0:
            return (torch.clamp(prediction_ - label_ , 0.)** discriminator_exponent) , True
        else:
            return 0.0, False

    if c_[1] + s_[1] == 0:
        return (torch.clamp(prediction_ - label_ + 1., 0.)**discriminator_exponent), True

    else:
        '''improve firmness'''
        # print(f'f____{sum(c_)} , {sum(s_) }')
        if f_[1] > f_[0]:

            return (torch.clamp(prediction_ - label_ , 0.)**firmness_exponent) , True
        elif f_[0] >= f_[1]:
            # return  (  label_ -prediction_+1.)**2, True
            return 0.0*(torch.clamp(label_ - prediction_, 0.)**firmness_exponent), True
        else:
            return 0.0, False

class TrainActionNet:
    def __init__(self,n_samples=None,epochs=1,learning_rate=5e-5):
        self.n_samples=n_samples
        self.size=n_samples
        self.epochs=epochs
        self.learning_rate=learning_rate

        '''model wrapper'''
        self.gan=self.prepare_model_wrapper()
        self.ref_generator=copy.deepcopy(self.gan.generator)
        self.ref_generator.train()
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

        '''optimizers'''
        '''different learning rate for the backbone'''
        backbone_params=[]
        other_params=[]
        for name,param in gan.generator.named_parameters():
            if 'back_bone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        params_group=[
            {'params':backbone_params,'lr':self.learning_rate/6},
            {'params': other_params, 'lr': self.learning_rate }
        ]

        # gan.critic_sgd_optimizer(learning_rate=self.learning_rate*10)
        # gan.critic_rmsprop_optimizer(learning_rate=self.learning_rate)
        gan.critic_adam_optimizer(learning_rate=self.learning_rate*10,beta1=0.9)
        gan.generator_adam_optimizer(param_group=params_group,beta1=0.9)
        # gan.generator_sgd_optimizer(learning_rate=self.learning_rate)
        return gan

    def simulate_elevation_variations(self,original_depth,seed,max_elevation=0.2,exponent=2.0):
        with torch.no_grad():
            _, _, _, _, _, background_class_3, _ = self.gan.generator(
                original_depth.clone(),seed=seed, alpha=0.0, dist_width_sigmoid=False)

            '''Elevation-based Augmentation'''
            objects_mask = background_class_3 <= 0.5
            shift_entities_mask = objects_mask & (original_depth > 0.0001)
            new_depth = original_depth.clone().detach()
            new_depth[shift_entities_mask] -= max_elevation * (np.random.rand()**exponent) * camera.scale

            return new_depth


    def begin(self):

        pi = progress_indicator('Begin new training round: ', max_limit=len(self.data_loader))
        gripper_pose=None
        for i, batch in enumerate(self.data_loader, 0):
            # print(i)

            depth,file_ids= batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pi.step(i)
            max_n = 300
            m = 16 if self.relative_sampling_timing.val<0.5 and self.moving_out_of_scope.val<0.1 and self.moving_collision_rate.val<0.5 else 2

            # detach_beta, detach_dist, detach_width = False, False, False
            # detach_beta=True if np.random.rand()<0.3 else False
            # detach_dist=True if np.random.rand()<0.3 else False
            # detach_width=True if np.random.rand()<0.3 else False
            # if detach_beta and detach_dist and detach_width: detach_beta,detach_dist,detach_width=False,False,False

            '''Elevation-based augmentation'''
            seed=np.random.randint(0,5000)
            if np.random.rand()>0.5: depth=self.simulate_elevation_variations(depth,seed)

            '''generate grasps'''
            with torch.no_grad():
                gripper_pose,suction_direction,griper_collision_classifier_2,_,_,background_class_2,_ = self.gan.generator(depth.clone(),seed=seed,alpha=0.0,dist_width_sigmoid=False)
                early_sampled_poses=gripper_pose.detach().clone()
                # r_k=(max(self.moving_collision_rate.val , self.moving_out_of_scope.val,0.001) )
                r_k = (max( self.moving_out_of_scope.val,self.relative_sampling_timing.val, 0.001))**0.5
                # r_k=self.relative_sampling_timing.val**0.5
                gripper_pose_ref,_,_,_,_,_,_ = self.ref_generator(depth.clone(),seed=seed,alpha=0.0,randomization_factor=r_k,dist_width_sigmoid=False)
                # if detach_beta:gripper_pose_ref[:,3:5]=gripper_pose[:,3:5].detach().clone()
                # if detach_dist:gripper_pose_ref[:,-2]=gripper_pose[:,-2].detach().clone()
                # if detach_width:gripper_pose_ref[:,-1]=gripper_pose[:,-1].detach().clone()

                '''get parameters'''
                pc, mask = depth_to_point_clouds(depth[0, 0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)

            '''zero grad'''
            self.gan.critic.zero_grad()
            self.gan.generator.zero_grad()

            '''self supervised critic learning'''
            loss = torch.tensor([0.], device=gripper_pose.device)
            with torch.no_grad():
                generated_grasps_cat = torch.cat([gripper_pose, gripper_pose_ref], dim=0)
                depth_cat = depth.repeat(2, 1, 1, 1)
            critic_score = self.gan.critic(depth_cat, generated_grasps_cat)
            counter = 0
            tracked_indexes=[]
            background_class_predictions = background_class_2.permute(0, 2, 3, 1)[0, :, :, 0][mask]
            objects_mask = background_class_predictions <= 0.5
            collide_with_objects_p=griper_collision_classifier_2[0, 0][mask].detach()
            collide_with_bins_p=griper_collision_classifier_2[0, 1][mask].detach()

            # pivot=self.moving_collision_rate.val
            # selection_p=1.-torch.abs(1.0-torch.sqrt(collide_with_objects_p*collide_with_bins_p)-pivot)
            selection_p=torch.rand_like(collide_with_objects_p)

            gripper_pose2=gripper_pose.permute(0, 2, 3, 1)[0, :, :, :][mask]
            ref_gripper_pose2=gripper_pose_ref.permute(0, 2, 3, 1)[0, :, :, :][mask]
            gen_scores_=critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]
            ref_scores_=critic_score.permute(0, 2, 3, 1)[1, :, :, 0][mask]

            '''reachability mask'''
            # def get_reachability_scope(poses_7):
            #     pc_tensor = torch.from_numpy(pc).to('cuda').float()
            #     distance = poses_7[:, -2:-1] * distance_scope
            #     gripper_approach=poses_7[:,:3]
            #     transition = pc_tensor + distance * gripper_approach
            #     with torch.no_grad():
            #         gripper_scope = self.gripper_arm_reachability_net(
            #             torch.cat([transition, gripper_approach], dim=-1)).squeeze()
            #     return gripper_scope
            # reachability_mask=(get_reachability_scope(ref_gripper_pose2)>0.5)

            # print(f'reachablity instances are {reachability_mask.sum()} out of {reachability_mask.shape}')

            selection_mask=objects_mask.clone() if self.relative_sampling_timing.val<0.5 else torch.full_like(objects_mask,True,dtype=torch.bool)
            n=int(min(max_n,selection_mask.sum()))
            speculated_generator_loss=0.
            for t in range(n):
                # idx_nonzero, = np.nonzero(selection_mask)
                dist=MaskedCategorical(probs=selection_p,mask=selection_mask)

                target_index=dist.sample()
                selection_mask[target_index]=False

                # target_index = np.random.choice(idx_nonzero)
                target_point=pc[target_index]

                target_generated_pose=gripper_pose2[target_index]
                target_ref_pose=ref_gripper_pose2[target_index]
                label_=ref_scores_[target_index]
                prediction_=gen_scores_[target_index]
                c_,s_,f_=evaluate_grasps3(target_point, target_generated_pose, target_ref_pose, pc, visualize=False)
                # print(c_,'  ',s_)

                if c_[0]==0 or c_[1]==0 or (collide_with_bins_p[target_index]<=0.5 and collide_with_objects_p[target_index]<=0.5):
                    self.moving_collision_rate.update(int(c_[0]>0))
                if sum(c_)==0 and sum(s_)==0:
                    self.moving_firmness.update(int(f_[0] > f_[1]))
                self.moving_out_of_scope.update(int(s_[0]>0))

                l,counted=critic_loss(c_, s_, f_, prediction_, label_)
                if counted:
                    counter+=1
                    loss+=l/m
                    avoid_collision=(c_[0]>0. or s_[0]>0.)
                    speculated_generator_loss+=avoid_collision*torch.clamp(label_.detach() - prediction_.detach(), 0.)
                    tracked_indexes.append((target_index,avoid_collision,f_[1]>f_[0]))
                if counter==m:
                    self.relative_sampling_timing.update((t+1)/n)
                    break
            l_c=loss.item()
            self.critic_statistics.loss=l_c
            if counter== m:
                loss.backward()
                self.gan.critic_optimizer.step()
                self.gan.critic_optimizer.zero_grad()
            else:
                print('pass, counter/m=',counter,'/',m)
                continue

            # if speculated_generator_loss<1e-4:
            #     print('pass generator training',f' critic loss: {loss.item()}')
            #     continue # train the discriminator

            '''zero grad'''
            self.gan.critic.zero_grad()
            self.gan.generator.zero_grad()
            
            '''generated grasps'''
            gripper_pose, suction_direction, griper_collision_classifier, suction_quality_classifier, shift_appealing_classifier,background_class,depth_features = self.gan.generator(
                depth.clone(),seed=seed,alpha=0.0,detach_backbone=detach_backbone,dist_width_sigmoid=False)

            assert (gripper_pose.detach()-early_sampled_poses).sum()==0, f'{(gripper_pose.detach()-early_sampled_poses).sum()==0}'

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
            gripper_sampling_loss = gripper_pose.mean()*0.0

            non_zero_background_loss_counter=0

            '''gripper sampler loss'''
            with torch.no_grad():
                ref_critic_score = self.gan.critic(depth.clone(), gripper_pose_ref)
                assert not torch.isnan(ref_critic_score).any(), f'{ref_critic_score}'
                ref_scores_ = ref_critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]

            generated_critic_score = self.gan.critic(depth.clone(), gripper_pose, detach_backbone=True)
            pred_scores_= generated_critic_score.permute(0, 2, 3, 1)[0, :, :, 0][mask]

            for j in range(m):
                target_index = tracked_indexes[j][0]
                avoid_collision=tracked_indexes[j][1]
                improve_firmness=tracked_indexes[j][2]
                label=ref_scores_[target_index]
                pred_=pred_scores_[target_index]
                gripper_sampling_loss +=  avoid_collision*(torch.clamp(label - pred_, 0.)**generator_exponent) / m

                # gripper_sampling_loss-=weight*pred_/m
            self.gripper_sampler_statistics.loss=gripper_sampling_loss.item()

            suction_sampling_loss += suction_sampler_loss(pc, suction_direction.permute(0, 2, 3, 1)[0][mask],file_index=file_ids[0])

            gripper_poses=gripper_pose[0].permute(1,2,0)[mask].detach()#.cpu().numpy()
            suction_head_predictions=suction_quality_classifier[0, 0][mask]
            gripper_head_predictions=griper_collision_classifier[0, :].permute(1,2,0)[mask]
            shift_head_predictions = shift_appealing_classifier[0, 0][mask]
            background_class_predictions = background_class.permute(0,2, 3, 1)[0, :, :, 0][mask]

            '''limits'''
            with torch.no_grad():
                normals = suction_direction[0].permute(1, 2, 0)[mask].detach().cpu().numpy()
                objects_mask = background_class_predictions.detach().cpu().numpy() <= 0.5
                gripper_head_max_score = torch.max(griper_collision_classifier[:,i%2]).item()
                gripper_head_score_range = (gripper_head_max_score - torch.min(griper_collision_classifier[:,i%2])).item()
                suction_head_max_score = torch.max(suction_quality_classifier).item()
                suction_head_score_range = (suction_head_max_score - torch.min(suction_quality_classifier)).item()
                shift_head_max_score = torch.max(shift_appealing_classifier).item()
                shift_head_score_range = (shift_head_max_score - torch.min(shift_appealing_classifier)).item()

            '''background detection head'''
            try:
                bin_mask = bin_planes_detection(pc, sides_threshold = 0.005,floor_threshold=0.0015, view=False, file_index=file_ids[0],cache_name='bin_planes2')
            except Exception as error_message:
                print(file_ids[0])
                print(error_message)
                bin_mask=None

            if bin_mask is None:
                print(Fore.RED,f'Failed to generate label for background segmentation, file id ={file_ids[0]}',Fore.RESET)
            else:
                label = torch.from_numpy(bin_mask).to(background_class_predictions.device).float()
                # background_loss += balanced_bce_loss(background_class_predictions,label,positive_weight=2.0,negative_weight=1)
                background_loss+=bce_loss(background_class_predictions,label)
                self.background_detector_statistics.update_confession_matrix(label,background_class_predictions.detach())
                non_zero_background_loss_counter+=1

            for k in range(m):
                '''gripper collision head'''
                sta=self.objects_collision_statistics if i%2==0 else self.bin_collision_statistics
                gripper_target_index=model_dependent_sampling(pc, gripper_head_predictions[:,i%2], gripper_head_max_score, gripper_head_score_range,objects_mask,probability_exponent=10,balance_indicator=sta.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = gripper_head_predictions[gripper_target_index]
                gripper_target_pose = gripper_poses[gripper_target_index]
                gripper_loss+=gripper_collision_loss(gripper_target_pose, gripper_target_point, pc,objects_mask, gripper_prediction_,self.objects_collision_statistics ,self.bin_collision_statistics)/m

            for k in range(m):
                '''suction seal head'''
                suction_target_index=model_dependent_sampling(pc, suction_head_predictions, suction_head_max_score, suction_head_score_range,objects_mask,probability_exponent=10,balance_indicator=self.suction_head_statistics.label_balance_indicator)
                suction_prediction_ = suction_head_predictions[suction_target_index]
                suction_loss+=suction_seal_loss(pc,normals,suction_target_index,suction_prediction_,self.suction_head_statistics,objects_mask)/m

            for k in range(m):
                '''shift affordance head'''
                shift_target_index = model_dependent_sampling(pc, shift_head_predictions, shift_head_max_score,shift_head_score_range,probability_exponent=10,balance_indicator=self.shift_head_statistics.label_balance_indicator)
                shift_target_point = pc[shift_target_index]
                shift_prediction_=shift_head_predictions[shift_target_index]
                shift_loss+=shift_affordance_loss(pc,shift_target_point,objects_mask,self.shift_head_statistics,shift_prediction_)/m

            if non_zero_background_loss_counter>0: background_loss/non_zero_background_loss_counter

            print(f'c_loss={truncate(l_c)}, g_loss={(gripper_sampling_loss.item())},  ratios c:{self.moving_collision_rate.val}, s:{self.moving_out_of_scope.val}')

            loss=suction_loss*0.1+gripper_loss*0.5+shift_loss*0.3+gripper_sampling_loss*5.0+suction_sampling_loss+background_loss*10.0
            loss.backward()
            self.gan.generator_optimizer.step()
            self.gan.generator_optimizer.zero_grad()

            with torch.no_grad():
                self.gripper_sampler_statistics.loss = gripper_sampling_loss.item()
                self.suction_sampler_statistics.loss = suction_sampling_loss.item()
                self.suction_head_statistics.loss = suction_loss.item()
                self.shift_head_statistics.loss = shift_loss.item()
                self.background_detector_statistics.loss=background_loss.item()

            if i%25==0 and i!=0:
                self.view_result(gripper_pose)
                self.export_check_points()
                self.save_statistics()
                self.ref_generator = copy.deepcopy(self.gan.generator)
                self.ref_generator.train()

        pi.end()

        self.view_result(gripper_pose)

        self.export_check_points()
        self.clear()

    def view_result(self,gripper_pose):
        with torch.no_grad():
            self.suction_sampler_statistics.print()
            self.suction_head_statistics.print()
            self.bin_collision_statistics.print()
            self.objects_collision_statistics.print()

            self.shift_head_statistics.print()
            self.background_detector_statistics.print()
            self.gripper_sampler_statistics.print()
            self.critic_statistics.print()

            values = gripper_pose.permute(1, 0, 2, 3).flatten(1).detach()

            print(f'gripper_pose sample = {values[:,10000].cpu()}')
            print(f'gripper_pose std = {torch.std(values, dim=-1).cpu()}')
            print(f'gripper_pose mean = {torch.mean(values, dim=-1).cpu()}')
            print(f'gripper_pose max = {torch.max(values, dim=-1)[0].cpu()}')
            print(f'gripper_pose min = {torch.min(values, dim=-1)[0].cpu()}')

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
    lr = 5e-4
    train_action_net = TrainActionNet( n_samples=None, learning_rate=lr)
    train_action_net.initialize(n_samples=100)
    train_action_net.begin()
    for i in range(1000):
        try:
            cuda_memory_report()
            train_action_net.initialize(n_samples=None)
            train_action_net.begin()
        except Exception as error_message:
            torch.cuda.empty_cache()
            print(Exception,error_message)
