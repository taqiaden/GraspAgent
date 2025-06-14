import copy
import os
import time
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from Configurations.config import workers
from Online_data_audit.data_tracker2 import DataTracker2
from check_points.check_point_conventions import ModelWrapper, GANWrapper
from dataloaders.policy_dl import SeizePolicyDataset, ClearPolicyDataset, DemonstrationsDataset
from lib.IO_utils import load_pickle, save_pickle
from lib.Multible_planes_detection.plane_detecttion import cache_dir, bin_planes_detection
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data2,demonstrations_data
from lib.depth_map import depth_to_point_clouds, transform_to_camera_frame
from lib.image_utils import view_image
from lib.loss.D_loss import binary_smooth_l1, binary_l1
from lib.math_utils import seeds
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from lib.sklearn_clustering import dbscan_clustering
from models.action_net import action_module_key, ActionNet
from models.policy_net import policy_module_key, PolicyNet
from records.training_satatistics import TrainingTracker, MovingRate
from registration import camera
from training.learning_objectives.shift_affordnace import shift_labeling
from training.learning_objectives.suction_seal import l1_smooth_loss
from training.ppo_memory import PPOMemory
import random
from lib.report_utils import wait_indicator as wi
from visualiztion import view_npy_open3d

buffer_file='buffer.pkl'
action_data_tracker_path=r'online_data_dict.pkl'
cache_name='clustering'

online_data2=online_data2()
demonstrations_data=demonstrations_data()


# bce_loss= torch.nn.BCELoss()

def policy_loss(new_policy_probs,old_policy_probs,advantages,epsilon=0.2):
    ratio = new_policy_probs / old_policy_probs
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    objective = torch.min(ratio * advantages, clipped_ratio * advantages)
    return objective

class TrainPolicyNet:
    def __init__(self,learning_rate=5e-5):

        self.policy_clip_margin=0.2
        self.action_net = None
        self.learning_rate=learning_rate
        self.model_wrapper=ModelWrapper(model=PolicyNet(), module_key=policy_module_key)
        self.quality_dataloader=None

        '''initialize statistics records'''
        self.gripper_quality_net_statistics = TrainingTracker(name=policy_module_key + '_gripper_quality',
                                                              track_label_balance=True,min_decay=0.01)
        self.suction_quality_net_statistics = TrainingTracker(name=policy_module_key + '_suction_quality',
                                                              track_label_balance=True,min_decay=0.01)

        self.demonstrations_statistics=TrainingTracker(name=policy_module_key + '_demonstrations',
                                                              track_label_balance=False,min_decay=0.01)

        self.buffer = online_data2.load_pickle(buffer_file) if online_data2.file_exist(buffer_file) else PPOMemory()

        self.data_tracker = DataTracker2(name=action_data_tracker_path, list_size=10)
        self.last_tracker_size=len(self.data_tracker)

        self.buffer_time_stamp=None
        self.data_tracker_time_stamp=None

        self.gripper_sampling_rate=MovingRate('gripper_sampling_rate',min_decay=0.01)

        ''''statistics tracker'''
        self.ini_policy_moving_loss=MovingRate(policy_module_key+'_ini_policy_moving_loss',min_decay=0.01)
        self.ini_value_moving_loss=MovingRate(policy_module_key+'_ini_value_moving_loss',min_decay=0.01)

    def initialize_model(self):
        self.model_wrapper.ini_model(train=True)

        '''different learning rate for the backbone'''
        backbone_params=[]
        other_params=[]
        for name,param in self.model_wrapper.model.named_parameters():
            if 'back_bone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        params_group=[
            {'params':backbone_params,'lr':self.learning_rate/10},
            {'params': other_params, 'lr': self.learning_rate }
        ]

        self.model_wrapper.ini_adam_optimizer(params_group=params_group,learning_rate=self.learning_rate)

    # @property
    # def training_trigger(self):
    #     return len(self.data_tracker)>self.last_tracker_size

    def synchronize_buffer(self):
        new_buffer=False
        new_data_tracker=False
        new_buffer_time_stamp=online_data2.file_time_stamp(buffer_file)
        if self.buffer_time_stamp is None or new_buffer_time_stamp!=self.buffer_time_stamp:
            self.buffer = online_data2.load_pickle(buffer_file) if online_data2.file_exist(buffer_file) else PPOMemory()
            self.buffer_time_stamp=new_buffer_time_stamp
            new_buffer=True

        new_data_tracker_time_stamp=online_data2.file_time_stamp(action_data_tracker_path)
        if self.data_tracker_time_stamp is None or self.data_tracker_time_stamp!=new_data_tracker_time_stamp:
            self.data_tracker = DataTracker2(name=action_data_tracker_path, list_size=10)
            self.data_tracker_time_stamp=new_data_tracker_time_stamp
            new_data_tracker=True

        return new_buffer,new_data_tracker

    def experience_sampling(self,replay_size):
        suction_size=int(self.gripper_sampling_rate.val*replay_size)
        gripper_size=replay_size-suction_size
        gripper_ids=self.data_tracker.gripper_grasp_sampling(gripper_size,self.gripper_quality_net_statistics.label_balance_indicator)
        suction_ids=self.data_tracker.suction_grasp_sampling(suction_size,self.suction_quality_net_statistics.label_balance_indicator)

        return gripper_ids+suction_ids

    def demonstrations_buffer_sampling(self,batch_size,n_batches):
        all_ids=demonstrations_data.get_indexes()
        sampled_size=int(batch_size*n_batches)
        sampled_ids=random.sample(all_ids,sampled_size)
        return sampled_ids


    def mixed_buffer_sampling(self,batch_size,n_batches,online_ratio=0.5):
        total_size=int(batch_size*n_batches)
        online_size=int(total_size*online_ratio)
        available_buffer_size=len(self.buffer.non_episodic_file_ids)
        online_size=min(available_buffer_size,online_size)
        replay_size=total_size-online_size
        print(f'Sample {online_size} from online buffer and {replay_size} from experience.')

        '''sample from old experience'''
        replay_ids=self.experience_sampling(replay_size)

        '''sample from online pool'''
        if available_buffer_size==0:
            print('No file is found in the recent buffer')
            return replay_ids
        else:
            indexes=np.random.choice(np.arange(available_buffer_size,dtype=np.int64),online_size).tolist()
            online_ids=[self.buffer.non_episodic_file_ids[i] for i in indexes]

        return online_ids+replay_ids

    def get_seize_policy_dataloader(self,file_ids,batch_size):
        dataset = SeizePolicyDataset(data_pool=online_data2, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                                       shuffle=True)
        return  data_loader
    def get_demonstrations_data_loader(self,file_ids,batch_size):
        dataset = DemonstrationsDataset(data_pool=demonstrations_data, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                                  shuffle=True)
        return data_loader
    def get_clear_policy_dataloader(self,file_ids,batch_size,buffer):
        dataset = ClearPolicyDataset(data_pool=online_data2, policy_buffer=buffer,file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                                  shuffle=True)
        return data_loader

    def initialize_action_net(self):
        actions_net = ModelWrapper(model=ActionNet(), module_key=action_module_key)
        actions_net.ini_model(train=False)
        self.action_net=actions_net.model

    def get_random_target_mask(self,pc,mask,background_class,file_id):
        pixel_mask=background_class<=0.5
        pixel_mask.permute(1, 2, 0)[ :, :, 0][~mask]*=False

        # view_image(pixel_mask[ 0].cpu().numpy().astype(np.float64))
        background_class_predictions = background_class.permute(1, 2, 0)[ :, :, 0][mask]
        objects_mask = background_class_predictions <= 0.5

        object_points = pc[objects_mask.cpu().numpy()]

        cluster_file_path = cache_dir + cache_name + '/' + str(file_id.item()) + '.pkl'
        if os.path.exists(cluster_file_path):
            clusters_label = load_pickle(cluster_file_path)
            if clusters_label.shape[0] != object_points.shape[0]:
                clusters_label = dbscan_clustering(object_points, view=False)
                save_pickle(cluster_file_path, clusters_label)
        else:
            clusters_label = dbscan_clustering(object_points, view=False)
            save_pickle(cluster_file_path, clusters_label)

        sets_ = set(clusters_label)
        sets_ = [i for i in sets_ if i >= 0]

        if len(sets_)==0:
            # view_npy_open3d(object_points)
            # view_image(pixel_mask[0].cpu().numpy().astype(np.float64))
            return pixel_mask
        counts_ = [(clusters_label == x).sum() for x in sets_]
        picked_cluster_label = sets_[np.argmax(np.array(counts_))]
        # picked_cluster_label=np.random.randint(0,len(sets_))
        cluster_mask = clusters_label == picked_cluster_label
        cluster_mask = torch.from_numpy(cluster_mask).cuda()

        '''target_mask[j, 0][mask][target_mask_][~cluster_mask]*=0'''
        temp_1 = pixel_mask[ 0][mask]
        temp_2 = temp_1[objects_mask]
        temp_2[~cluster_mask] *= False
        temp_1[objects_mask] = temp_2
        pixel_mask[0][mask] = temp_1

        return pixel_mask

    def policy_init_loss(self,q_value,pcs,masks,target_masks,action_probs,sample_size=2):
        value_loss = torch.tensor(0., device=q_value.device)
        for j in range(q_value.shape[0]):
            mask = masks[j]
            pc = pcs[j]
            target_mask_predictions = target_masks.permute(0, 2, 3, 1)[j, :, :, 0][mask]
            target_mask_ = target_mask_predictions > 0.5

            gripper_grasp_value = q_value[j, 0][mask]
            suction_grasp_value = q_value[j, 1][mask]
            gripper_shift_value = q_value[j, 2][mask]
            suction_shift_value = q_value[j, 3][mask]


            target_object_points = pc[target_mask_.detach().cpu().numpy()]

            # view_npy_open3d(pc=target_object_points)


            '''gripper grasp sampling'''
            sampling_p = torch.rand_like(gripper_grasp_value, device='cpu')
            indexes_list=[]
            dist_list=[]
            for k in range(sample_size):
                dist = Categorical(probs=sampling_p)
                shared_index = dist.sample()
                dist.probs[shared_index] = 0

                '''labeling'''
                target_point = pc[shared_index]
                min_dist_ = np.min(np.linalg.norm(target_object_points - target_point[np.newaxis, :], axis=-1))
                max_ref = 0.15
                if min_dist_ < max_ref:
                    label = (1 - (min_dist_ / max_ref)) **0.3
                else:
                    label = 0.

                indexes_list.append(shared_index)
                dist_list.append(label)

                # is_object = objects_mask[shared_index]
                # is_bin = ~is_object
                # shift_weight=shift_appealing_j[shared_index].item()
                # print(gripper_grasp_value[shared_index],label,is_object)
                '''higher value for objects closer to the target'''
                value_loss += (gripper_grasp_value[shared_index] - label  * 0.5) ** 2
                value_loss += (suction_grasp_value[shared_index] - label  * 0.5) ** 2
                value_loss += (gripper_shift_value[shared_index] - label ) ** 2
                value_loss += (suction_shift_value[shared_index] - label ) ** 2


        '''adapt policy net to value net'''
        policy_label = q_value.detach().clone()
        policy_label = policy_label.reshape(policy_label.shape[0], -1)
        policy_label = (policy_label - policy_label.min()) / (policy_label.max() - policy_label.min())
        policy_label = F.softmax(policy_label, dim=-1)
        policy_pred = action_probs.reshape(policy_label.shape[0], -1)

        policy_loss = ((policy_pred - policy_label) ** 2).mean()

        value_loss=value_loss / sample_size

        self.ini_policy_moving_loss.update(policy_loss.item())
        self.ini_value_moving_loss.update(value_loss.item())

        loss = value_loss + policy_loss

        return loss

    def generate_random_target_mask(self,depth,file_ids,target_masks,pcs,masks):
        '''action space'''
        if self.action_net is None: self.initialize_action_net()

        with torch.no_grad():
            _, _, _, _, shift_appealing_classifier \
                , background_class = self.action_net(depth.clone())

            for j in range(depth.shape[0]):
                pc=pcs[j]
                # print(pc.shape)
                mask=masks[j]

                '''pick random cluster to be the target'''
                # During inference this is replaced by object specific grasp segmentation using Grounded dino sam 2.0
                background_class_predictions = background_class.permute(0, 2, 3, 1)[j, :, :, 0][mask]
                objects_mask = background_class_predictions <= 0.5

                target_masks[j] = self.get_random_target_mask(pc, mask, background_class[j], file_id=file_ids[j])

                # view_image(target_masks[j,0].cpu().numpy().astype(np.float64))

        return target_masks,objects_mask,shift_appealing_classifier

    def get_point_clouds(self,depth):

        pcs = []
        masks = []
        for j in range(depth.shape[0]):
            pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
            pc = transform_to_camera_frame(pc, reverse=True)

            pcs.append(pc)
            masks.append(mask)

        return pcs,masks

    def quality_loss(self,griper_grasp_quality_score,suction_grasp_quality_score,gripper_score,suction_score,used_gripper,used_suction,gripper_pixel_index,suction_pixel_index):
        loss = torch.tensor(0., device=griper_grasp_quality_score.device) * griper_grasp_quality_score.mean()
        for j in range(griper_grasp_quality_score.shape[0]):

            weight=1.0
            if used_gripper[j]:
                label = gripper_score[j]
                if label == -1: continue
                g_pix_A = gripper_pixel_index[j, 0]
                g_pix_B = gripper_pixel_index[j, 1]
                prediction = griper_grasp_quality_score[j, 0, g_pix_A, g_pix_B]
                l = binary_smooth_l1(prediction, label)

                self.gripper_sampling_rate.update(1)

                self.gripper_quality_net_statistics.loss = l.item()
                self.gripper_quality_net_statistics.update_confession_matrix(label, prediction)
                loss += l
                if used_suction[j]:weight=2.0

            if used_suction[j]:
                label = suction_score[j]
                if label == -1: continue
                s_pix_A = suction_pixel_index[j, 0]
                s_pix_B = suction_pixel_index[j, 1]
                prediction = suction_grasp_quality_score[j, 0, s_pix_A, s_pix_B]
                l = binary_smooth_l1(prediction, label)

                self.gripper_sampling_rate.update(0)

                self.suction_quality_net_statistics.loss = l.item()
                self.suction_quality_net_statistics.update_confession_matrix(label, prediction)

                loss += l*weight


        return loss
    def analytical_bin_mask(self, pc, file_ids):
        try:
            bin_mask = bin_planes_detection(pc, sides_threshold=0.005, floor_threshold=0.0015, view=False,
                                            file_index=file_ids[0], cache_name='bin_planes2')
        except Exception as error_message:
            print(file_ids[0])
            print(error_message)
            bin_mask = None
        return bin_mask
    def simulate_elevation_variations(self, original_depth,objects_mask, max_elevation=0.2, exponent=2.0):
        '''Elevation-based Augmentation'''
        shift_entities_mask = objects_mask & (original_depth > 0.0001)
        new_depth = original_depth.clone().detach()
        new_depth[shift_entities_mask] -= max_elevation * (np.random.rand() ** exponent) * camera.scale

        return new_depth
    def first_phase_training(self,max_size=100,batch_size=1):

        '''dataloaders'''
        # seize_policy_ids = self.mixed_buffer_sampling(batch_size=batch_size, n_batches=max_size)
        seize_policy_ids = self.experience_sampling(int(batch_size*max_size))
        seize_policy_data_loader=self.get_seize_policy_dataloader(seize_policy_ids,batch_size)

        '''demonstrations'''
        demonstrations_ids=self.demonstrations_buffer_sampling(batch_size=batch_size, n_batches=max_size)
        demonstrations_data_loader=self.get_demonstrations_data_loader(demonstrations_ids,batch_size)

        pi = progress_indicator('Begin new training round: ', max_limit=len(seize_policy_data_loader))
        if self.action_net is None: self.initialize_action_net()

        for i,(seize_policy_batch,demonstrations_batch) in enumerate(zip(seize_policy_data_loader,demonstrations_data_loader),start=0):

            '''learn from demonstrations'''
            rgb, depth,labels,file_ids=demonstrations_batch
            rgb = rgb.cuda().float().permute(0, 3, 1, 2)
            depth = depth.cuda().float()

            '''zero grad'''
            self.model_wrapper.model.zero_grad()

            b = rgb.shape[0]
            pcs, masks = self.get_point_clouds(depth)

            '''Elevation-based augmentation'''
            for k in range(depth.shape[0]):
                '''background detection head'''
                bin_mask = self.analytical_bin_mask(pcs[k], file_ids[k])
                if bin_mask is None: continue
                objects_mask_numpy = bin_mask <= 0.5
                objects_mask = torch.from_numpy(objects_mask_numpy).cuda()
                objects_mask_pixel_form = torch.ones_like(depth)
                objects_mask_pixel_form[0, 0][masks[k]] = objects_mask_pixel_form[0, 0][masks[k]] * objects_mask
                objects_mask_pixel_form = objects_mask_pixel_form > 0.5
                if np.random.rand() > 0.7:
                    depth[i:i+1] = self.simulate_elevation_variations(depth[i:i+1], objects_mask_pixel_form, exponent=5.0)
                    pcs[k], masks[k] = depth_to_point_clouds(depth[0, 0].cpu().numpy(), camera)
                    pcs[k] = transform_to_camera_frame(pcs[k], reverse=True)

            with torch.no_grad():
                gripper_pose,normal_direction, _, _, shift_appealing_classifier \
                    , background_class = self.action_net(depth.clone())


            target_masks=background_class<0.5
            griper_grasp_quality_score, suction_grasp_quality_score, \
                q_value, action_probs, _ = \
                self.model_wrapper.model(rgb, depth.clone(), gripper_pose, normal_direction, target_masks.float(), target_masks)



            demonstration_loss=torch.tensor(0.,device=rgb.device)
            for j in range(b):
                if labels[j,0] != -1:
                    pass
                elif labels[j,1] != -1:
                    '''No grasp points'''
                    target_predictions=griper_grasp_quality_score[j,0][masks[j]]
                    ground_truth=torch.zeros_like(target_predictions)
                    demonstration_loss+=(binary_l1(target_predictions, ground_truth)**2.).mean()
                elif labels[j,2] != -1:
                    '''No suction points'''
                    target_predictions=suction_grasp_quality_score[j,0][masks[j]]
                    ground_truth=torch.zeros_like(target_predictions)
                    demonstration_loss+=(binary_l1(target_predictions, ground_truth)**2.).mean()
                elif labels[j,3] != -1:
                    '''Priority to grasp'''
                    objects_mask=target_masks[j,0][masks[j]]
                    target_gripper_predictions=griper_grasp_quality_score[j,0][masks[j]]
                    target_suction_predictions=suction_grasp_quality_score[j,0][masks[j]]
                    demonstration_loss+=(torch.clamp(target_suction_predictions[objects_mask]-target_gripper_predictions[objects_mask],0.)).mean()
                elif labels[j,4] != -1:
                    '''Priority to suction'''
                    objects_mask=target_masks[j,0][masks[j]]
                    target_gripper_predictions=griper_grasp_quality_score[j,0][masks[j]]
                    target_suction_predictions=suction_grasp_quality_score[j,0][masks[j]]
                    demonstration_loss+=(torch.clamp(target_gripper_predictions[objects_mask]-target_suction_predictions[objects_mask],0.)).mean()
                else:
                    assert False,f'{labels}'

            self.demonstrations_statistics.loss=demonstration_loss.item()
            # print('---',labels)
            # print('....',objects_mask[objects_mask].shape)
            demonstration_loss.backward()

            self.model_wrapper.optimizer.step()

            '''learn from robot actions'''
            rgb, depth,target_masks, pose_7, gripper_pixel_index, \
                suction_pixel_index, gripper_score, \
                suction_score, normal, used_gripper, used_suction,file_ids = seize_policy_batch

            rgb = rgb.cuda().float().permute(0, 3, 1, 2)
            target_masks = target_masks.cuda().float()
            depth=depth.cuda().float()
            pose_7 = pose_7.cuda().float()
            gripper_score = gripper_score.cuda().float()
            suction_score = suction_score.cuda().float()
            normal = normal.cuda().float()

            b = rgb.shape[0]
            w = rgb.shape[2]
            h = rgb.shape[3]

            pcs,masks=self.get_point_clouds(depth)

            '''random target for policy initialization'''
            altered_target_masks, objects_mask,shift_appealing_classifier = self.generate_random_target_mask(depth, file_ids, target_masks.clone(), pcs, masks)

            '''zero grad'''
            self.model_wrapper.model.zero_grad()

            '''process pose'''
            pose_7_stack = torch.zeros((b, 7, w, h), device=rgb.device)
            normal_stack = torch.zeros((b, 3, w, h), device=rgb.device)

            for j in range(b):
                g_pix_A = gripper_pixel_index[j, 0]
                g_pix_B = gripper_pixel_index[j, 1]
                s_pix_A = suction_pixel_index[j, 0]
                s_pix_B = suction_pixel_index[j, 1]
                pose_7_stack[j, :, g_pix_A, g_pix_B] = pose_7[j]
                normal_stack[j, :, s_pix_A, s_pix_B] = normal[j]

            # print(shift_appealing_classifier.shape)
            # print(altered_target_masks.shape)
            # view_image(rgb[0].cpu().numpy().astype(np.float64).transpose(1,2,0),hide_axis=True)

            shift_mask_=(shift_appealing_classifier>0.5)
            for j in range(b):
                shift_mask_[j,0][~masks[j]] *= False
            griper_grasp_quality_score, suction_grasp_quality_score, \
                 q_value, action_probs,_ = \
                self.model_wrapper.model(rgb, depth.clone(), pose_7_stack, normal_stack, altered_target_masks,shift_mask_)

            # view_image(target_masks[0, 0].cpu().numpy().astype(np.float64),hide_axis=True)
            # view_image(altered_target_masks[0, 0].cpu().numpy().astype(np.float64),hide_axis=True)
            # # target_value=q_value[0, 0]
            # # view_image(target_value.detach().cpu().numpy().astype(np.float64))
            # target_value=action_probs[0, 0]
            # view_image(target_value.detach().cpu().numpy().astype(np.float64),hide_axis=True)
            # # shift_appealing_mask=shift_appealing_classifier[0,0]>0.5
            # # shift_appealing_mask[~masks[0]]=False

            '''accumulate loss'''
            quality_loss=self.quality_loss(griper_grasp_quality_score,suction_grasp_quality_score,
                                    gripper_score,suction_score,used_gripper,used_suction,
                                    gripper_pixel_index,suction_pixel_index)

            '''policy initialization loss'''
            # policy_loss = self.policy_init_loss(q_value, pcs, masks, altered_target_masks, action_probs,  sample_size=10)

            loss =  quality_loss

            assert not torch.isnan(loss).any(), f'{loss}'
            # print(policy_loss.item())
            # print(quality_loss.item(),policy_loss.item())

            loss.backward()

            self.model_wrapper.optimizer.step()

            pi.step(i)
        pi.end()

    def forward_seize_policy_loss(self,seize_policy_batch):
        rgb, depth, target_masks, pose_7, gripper_pixel_index, \
            suction_pixel_index, gripper_score, \
            suction_score, normal, used_gripper, used_suction, file_ids = seize_policy_batch

        rgb = rgb.cuda().float().permute(0, 3, 1, 2)
        target_masks = target_masks.cuda().float()
        depth = depth.cuda().float()
        pose_7 = pose_7.cuda().float()
        gripper_score = gripper_score.cuda().float()
        suction_score = suction_score.cuda().float()
        normal = normal.cuda().float()

        b = rgb.shape[0]
        w = rgb.shape[2]
        h = rgb.shape[3]

        '''zero grad'''
        self.model_wrapper.model.zero_grad()

        '''process pose'''
        pose_7_stack = torch.zeros((b, 7, w, h), device=rgb.device)
        normal_stack = torch.zeros((b, 3, w, h), device=rgb.device)

        for j in range(b):
            g_pix_A = gripper_pixel_index[j, 0]
            g_pix_B = gripper_pixel_index[j, 1]
            s_pix_A = suction_pixel_index[j, 0]
            s_pix_B = suction_pixel_index[j, 1]
            pose_7_stack[j, :, g_pix_A, g_pix_B] = pose_7[j]
            normal_stack[j, :, s_pix_A, s_pix_B] = normal[j]

        griper_grasp_quality_score, suction_grasp_quality_score, \
            q_value, action_probs, _ = \
            self.model_wrapper.model(rgb, depth.clone(), pose_7_stack, normal_stack, target_masks)

        '''accumulate loss'''
        quality_loss = self.quality_loss(griper_grasp_quality_score, suction_grasp_quality_score,
                                         gripper_score, suction_score, used_gripper, used_suction,
                                         gripper_pixel_index, suction_pixel_index)

        return quality_loss

    def forward_clear_policy_loss(self,clear_policy_batch):

        (rgb, depth, target_masks,values,advantages,
         action_indexes,point_indexes,probs,rewards,
         end_of_episodes)=clear_policy_batch

        rgb = rgb.cuda().float().permute(0, 3, 1, 2)
        target_masks = target_masks.cuda().float()
        depth = depth.cuda().float()

        pcs, masks=self.get_point_clouds(depth)

        b = rgb.shape[0]
        w = rgb.shape[2]
        h = rgb.shape[3]

        '''process pose'''
        pose_7_stack = torch.zeros((b, 7, w, h), device=rgb.device)
        normal_stack = torch.zeros((b, 3, w, h), device=rgb.device)

        griper_grasp_quality_score, suction_grasp_quality_score, \
            q_value, action_probs, _ = \
            self.model_wrapper.model(rgb, depth.clone(), pose_7_stack, normal_stack, target_masks)

        '''accumulate critic actor loss'''
        actor_loss=torch.tensor(0.,device=q_value.device)
        critic_loss=torch.tensor(0.,device=q_value.device)
        for j in range(b):
            mask=masks[j]
            action_index=action_indexes[j]
            point_index=point_indexes[j]

            q_value_j = q_value[j].permute(1, 2, 0)[mask]
            action_probs_j = action_probs[j].permute(1, 2, 0)[mask]

            old_probs=probs[j]
            new_probs=action_probs_j[point_index,action_index]
            new_probs=torch.log(new_probs)

            prob_ratio = new_probs.exp() / old_probs.exp()
            weighted_probs = advantages[j] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip_margin,
                                                 1 + self.policy_clip_margin) * advantages[j]

            actor_loss += -torch.min(weighted_probs, weighted_clipped_probs)

            critic_value=q_value_j[point_index,action_index]
            returns=advantages[j]+values[j]

            critic_loss += (returns - critic_value) ** 2

        total_loss = actor_loss + 0.5 * critic_loss

        return total_loss

    def second_phase_training(self,batch_size=1,n_epochs = 4):
        '''dataloaders'''
        clear_policy_ids=self.buffer.generate_batches(batch_size)
        seize_policy_ids = self.mixed_buffer_sampling(batch_size=batch_size, n_batches=len(clear_policy_ids))

        seize_policy_data_loader=self.get_seize_policy_dataloader(seize_policy_ids,batch_size)
        clear_policy_data_loader=self.get_clear_policy_dataloader(clear_policy_ids,batch_size,self.buffer)

        assert len(seize_policy_data_loader)==len(clear_policy_data_loader)

        pi = progress_indicator('Begin new training round: ', max_limit=len(seize_policy_data_loader)*n_epochs)
        counter=0
        for e in range(n_epochs):
            for seize_policy_batch, clear_policy_batch in zip(seize_policy_data_loader, clear_policy_data_loader):
                counter+=1
                self.model_wrapper.model.zero_grad()

                seize_policy_loss=self.forward_seize_policy_loss(seize_policy_batch)
                seize_policy_loss.backward()

                clear_policy_loss=self.forward_clear_policy_loss(clear_policy_batch)
                clear_policy_loss.backward()

                self.model_wrapper.optimizer.step()
                self.model_wrapper.optimizer.zero_grad()

                pi.step(counter)
        pi.end()

    def view_result(self):
        with torch.no_grad():
            self.gripper_quality_net_statistics.print()
            self.suction_quality_net_statistics.print()
            self.demonstrations_statistics.print()

            self.gripper_sampling_rate.view()

            self.ini_policy_moving_loss.view()
            self.ini_value_moving_loss.view()

    def save_statistics(self):
        self.gripper_quality_net_statistics.save()
        self.suction_quality_net_statistics.save()
        self.demonstrations_statistics.save()

        self.gripper_sampling_rate.save()

        self.ini_policy_moving_loss.save()
        self.ini_value_moving_loss.save()

    def export_check_points(self):
        try:
            self.model_wrapper.export_model()
            self.model_wrapper.export_optimizer()
        except Exception as e:
            print(str(e))

    def clear(self):
        self.gripper_quality_net_statistics.clear()
        self.suction_quality_net_statistics.clear()
        self.demonstrations_statistics.clear()

if __name__ == "__main__":
    # seeds(0)
    lr = 1e-4
    train_action_net = TrainPolicyNet(  learning_rate=lr)
    train_action_net.initialize_model()
    train_action_net.synchronize_buffer()

    wait = wi('Begin synchronized trianing')


    # counter=0

    while True:
        # try:
            new_buffer,new_data_tracker=train_action_net.synchronize_buffer()
    
            '''test code'''
            train_action_net.first_phase_training(max_size=10)
            # train_action_net.second_phase_training()
    
            train_action_net.export_check_points()
            train_action_net.save_statistics()
            train_action_net.view_result()
            continue
    
            '''online learning'''
            if new_data_tracker:
                cuda_memory_report()
    
                train_action_net.first_phase_training(max_size=10)
                train_action_net.export_check_points()
                train_action_net.save_statistics()
                train_action_net.view_result()
            else:
                wait.step(0.5)
        # except Exception as e:
        #     print(str(e))
            # counter+=1