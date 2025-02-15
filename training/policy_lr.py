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
from dataloaders.policy_dl import GraspQualityDataset
from lib.IO_utils import load_pickle, save_pickle
from lib.Multible_planes_detection.plane_detecttion import cache_dir
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data2
from lib.depth_map import depth_to_point_clouds, transform_to_camera_frame
from lib.image_utils import view_image
from lib.loss.D_loss import binary_smooth_l1
from lib.math_utils import seeds
from lib.report_utils import progress_indicator
from lib.rl.masked_categorical import MaskedCategorical
from lib.sklearn_clustering import dbscan_clustering
from models.action_net import action_module_key2, ActionNet
from models.policy_net import policy_module_key, PolicyNet
from records.training_satatistics import TrainingTracker, MovingRate
from registration import camera
from training.learning_objectives.suction_seal import l1_smooth_loss
from training.ppo_memory import PPOMemory
import random
from lib.report_utils import wait_indicator as wi
from visualiztion import view_npy_open3d

buffer_file='buffer.pkl'
action_data_tracker_path=r'online_data_dict.pkl'
cache_name='clustering'


online_data2=online_data2()

# bce_loss= torch.nn.BCELoss()

def policy_loss(new_policy_probs,old_policy_probs,advantages,epsilon=0.2):
    ratio = new_policy_probs / old_policy_probs
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    objective = torch.min(ratio * advantages, clipped_ratio * advantages)
    return objective

class TrainPolicyNet:
    def __init__(self,learning_rate=5e-5):

        self.action_net = None
        self.learning_rate=learning_rate
        self.model_wrapper=ModelWrapper(model=PolicyNet(), module_key=policy_module_key)

        self.quality_dataloader=None

        '''initialize statistics records'''
        self.gripper_quality_net_statistics = TrainingTracker(name=policy_module_key + '_gripper_quality',
                                                              track_label_balance=True)
        self.suction_quality_net_statistics = TrainingTracker(name=policy_module_key + '_suction_quality',
                                                              track_label_balance=True)

        self.buffer = online_data2.load_pickle(buffer_file) if online_data2.file_exist(buffer_file) else PPOMemory()
        self.data_tracker = DataTracker2(name=action_data_tracker_path, list_size=10)
        self.last_tracker_size=len(self.data_tracker)

        self.buffer_time_stamp=None
        self.data_tracker_time_stamp=None

        self.gripper_sampling_rate=MovingRate('gripper_sampling_rate')

        ''''statistics tracker'''
        self.ini_policy_moving_loss=MovingRate(policy_module_key+'_ini_policy_moving_loss',min_decay=0.01)
        self.representation_transfer_moving_loss=MovingRate(policy_module_key+'_representation_transfer_moving_loss',min_decay=0.01)
        self.ini_value_moving_loss=MovingRate(policy_module_key+'_ini_value_moving_loss',min_decay=0.01)


    def initialize_model(self):
        self.model_wrapper.ini_model(train=True)
        self.model_wrapper.ini_adam_optimizer(learning_rate=self.learning_rate)

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

    def init_quality_data_loader(self,file_ids,batch_size):
        dataset = GraspQualityDataset(data_pool=online_data2, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                                       shuffle=True)
        return  data_loader

    def initialize_action_net(self):
        action_net = GANWrapper(action_module_key2, ActionNet)
        action_net.ini_generator(train=False)
        self.action_net=action_net.generator

    def policy_initialization(self,max_size=10,batch_size=1):
        ids = self.experience_sampling(max_size)
        data_loader = self.init_quality_data_loader(ids, batch_size)
        pi = progress_indicator('Begin new training round: ', max_limit=len(data_loader))
        # assert size==len(file_ids)
        cloned_model=copy.deepcopy(self.model_wrapper.model)
        for i, batch in enumerate(data_loader, 0):

            rgb, depth, target_masks, pose_7, gripper_pixel_index, \
                suction_pixel_index, gripper_score, \
                suction_score, normal, used_gripper, used_suction,file_ids = batch

            rgb = rgb.cuda().float().permute(0, 3, 1, 2)
            target_masks = target_masks.cuda().float()
            depth = depth.cuda().float()

            b = rgb.shape[0]
            m = 100

            '''action space'''
            if self.action_net is None: self.initialize_action_net()
            with torch.no_grad():
                gripper_pose, suction_direction, _, _, _ \
                    , background_class, action_depth_features = self.action_net(depth.clone(), clip=True)

                pcs=[]
                masks=[]
                for j in range(b):
                    pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
                    pc = transform_to_camera_frame(pc, reverse=True)

                    pcs.append(pc)
                    masks.append(mask)
                    target_masks[j,0][~mask]*=0

                    '''pick random cluster to be the target'''
                    # During inference this is replaced by object specific grasp segmentation using Grounded dino sam 2.0
                    target_mask_predictions = target_masks.permute(0, 2, 3, 1)[j, :, :, 0][mask]
                    target_mask_ = target_mask_predictions > 0.5
                    object_points = pc[target_mask_.cpu().numpy()]

                    cluster_file_path = cache_dir+cache_name+'/' + str(file_ids[j].item()) + '.pkl'
                    if os.path.exists(cluster_file_path):
                        clusters_label=load_pickle(cluster_file_path)
                        if clusters_label.shape[0]!=object_points.shape[0]:
                            clusters_label = dbscan_clustering(object_points, view=False)
                            save_pickle(cluster_file_path, clusters_label)
                    else:
                        clusters_label=dbscan_clustering(object_points,view=False)
                        save_pickle(cluster_file_path, clusters_label)

                    sets_=set(clusters_label)
                    sets_=[i for i in sets_ if i>=0]
                    counts_=[(clusters_label==x).sum() for x in sets_ ]
                    picked_cluster_label=sets_[np.argmax(np.array(counts_))]

                    # max_cluster = max(sets_)
                    # picked_cluster_label=np.random.randint(0,max_cluster)
                    cluster_mask=clusters_label==picked_cluster_label
                    cluster_mask=torch.from_numpy(cluster_mask).cuda()

                    '''target_mask[j, 0][mask][target_mask_][~cluster_mask]*=0'''
                    temp_1 = target_masks[j, 0][mask]
                    temp_2 = temp_1[target_mask_]
                    temp_2[~cluster_mask] *= 0
                    temp_1[target_mask_] = temp_2
                    target_masks[j, 0][mask] = temp_1

                    # view_image(target_mask[j,0].cpu().numpy().astype(np.float64))
            # continue
            '''freeze other task output'''
            with torch.no_grad():
                ref_griper_grasp_score, ref_suction_grasp_score, \
                     _, _,_ = \
                    cloned_model(rgb, depth.clone(), gripper_pose, suction_direction, target_masks)

            '''zero grad'''
            self.model_wrapper.model.zero_grad()

            griper_grasp_score, suction_grasp_score, \
                 q_value, action_probs,policy_depth_features = \
                self.model_wrapper.model(rgb, depth.clone(), gripper_pose, suction_direction, target_masks)


            assert torch.isnan(q_value).any()==False,f'{q_value}'
            assert torch.isnan(action_probs).any()==False,f'{action_probs}'

            # view_image(target_masks[0, 0].cpu().numpy().astype(np.float64))
            # target_value=q_value[0, 0]
            # view_image(target_value.detach().cpu().numpy().astype(np.float64))
            # target_value=q_value[0, 2]
            # view_image(target_value.detach().cpu().numpy().astype(np.float64))
            # continue

            loss=torch.tensor([0.],device=q_value.device)
            value_loss=torch.tensor([0.],device=q_value.device)
            for j in range(b):
                mask=masks[j]
                pc=pcs[j]
                target_mask_predictions = target_masks.permute(0, 2, 3, 1)[j, :, :, 0][mask]
                target_mask_ = target_mask_predictions > 0.5
                background_class_predictions = background_class.permute(0, 2, 3, 1)[j, :, :, 0][mask]
                objects_mask = background_class_predictions <= 0.5
                gripper_grasp_value=q_value[j,0][mask]
                suction_grasp_value=q_value[j,1][mask]
                gripper_shift_value=q_value[j,2][mask]
                suction_shift_value=q_value[j,3][mask]

                target_object_points=pc[target_mask_.detach().cpu().numpy()]
                # view_npy_open3d(pc=object_points)

                def sample_(sampling_p):
                    # dist = MaskedCategorical(probs=sampling_p, mask=objects_mask)
                    dist = Categorical(probs=sampling_p)
                    index = dist.sample()
                    dist.probs[index]=0
                    target_point = pc[index]
                    min_dist_ = np.min(np.linalg.norm(target_object_points - target_point[np.newaxis, :], axis=-1))
                    max_ref=0.7
                    if min_dist_ < max_ref:
                        label=(1-(min_dist_/max_ref) )**2
                    else:
                        label=0.
                    return index, label

                '''gripper grasp sampling'''
                sampling_p=torch.rand_like(gripper_grasp_value,device='cpu')
                for k in range(m):
                    shared_index,label=sample_(sampling_p)
                    is_object=objects_mask[shared_index]
                    is_bin=~is_object
                    # shift_weight=shift_appealing_j[shared_index].item()

                    '''higher value for objects closer to the target'''
                    value_loss+=(gripper_grasp_value[shared_index]-label*is_object*0.5)**2
                    value_loss+=(suction_grasp_value[shared_index]-label*is_object*0.5)**2
                    value_loss+=(gripper_shift_value[shared_index]-label*(1+is_bin)*0.5)**2
                    value_loss+=(suction_shift_value[shared_index]-label*(1+is_bin)*0.5)**2

                    # print('Gripper grasp initialization result: ',((pred[0]>=pred[1])==(label[0]>=label[1])).item())
            '''adapt policy net to value net'''
            policy_label=q_value.detach().clone()
            policy_label = policy_label.reshape(policy_label.shape[0], -1)
            policy_label=(policy_label-policy_label.min())/(policy_label.max()-policy_label.min())
            policy_label = F.softmax(policy_label, dim=-1)
            policy_pred=action_probs.reshape(policy_label.shape[0], -1)

            policy_loss=((policy_pred-policy_label)**2).mean()
            representation_transfer_loss=((policy_depth_features-action_depth_features)**2).mean()
            distillation_loss=((griper_grasp_score-ref_griper_grasp_score)**2).mean()
            distillation_loss+=((suction_grasp_score-ref_suction_grasp_score)**2).mean()

            loss+=value_loss/m
            loss+=policy_loss
            loss+=representation_transfer_loss*0.01
            loss+=distillation_loss*1000

            self.ini_policy_moving_loss.update(policy_loss.item())
            self.ini_value_moving_loss.update(value_loss.item())
            self.representation_transfer_moving_loss.update(representation_transfer_loss.item())

            loss.backward()
            self.model_wrapper.optimizer.step()
            self.model_wrapper.optimizer.zero_grad()
            self.model_wrapper.model.zero_grad()

            pi.step(i)
        pi.end()

    def step_quality_training(self,max_size=100,batch_size=1):
        ids = self.mixed_buffer_sampling(batch_size=batch_size, n_batches=max_size)
        data_loader=self.init_quality_data_loader(ids,batch_size)
        pi = progress_indicator('Begin new training round: ', max_limit=len(data_loader))
        # assert size==len(file_ids)
        cloned_model=copy.deepcopy(self.model_wrapper.model)
        for i, batch in enumerate(data_loader, 0):

            rgb, depth,mask, pose_7, gripper_pixel_index, \
                suction_pixel_index, gripper_score, \
                suction_score, normal, used_gripper, used_suction,file_ids = batch

            rgb = rgb.cuda().float().permute(0, 3, 1, 2)
            mask = mask.cuda().float()
            depth=depth.cuda().float()
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

            griper_grasp_score, suction_grasp_score, \
                 q_value, action_probs,_ = \
                self.model_wrapper.model(rgb, depth.clone(), pose_7_stack, normal_stack, mask)

            with torch.no_grad():
                _, _, ref_q_value, ref_action_probs ,_= cloned_model(rgb, depth.clone(), pose_7_stack, normal_stack, mask)

            '''accumulate loss'''
            loss = torch.tensor(0., device=rgb.device)*griper_grasp_score.mean()
            for j in range(b):
                if used_gripper[j]:
                    label = gripper_score[j]
                    if label==-1:continue
                    g_pix_A = gripper_pixel_index[j, 0]
                    g_pix_B = gripper_pixel_index[j, 1]
                    prediction = griper_grasp_score[j, 0, g_pix_A, g_pix_B]
                    l=binary_smooth_l1(prediction, label)

                    self.gripper_sampling_rate.update(1)

                    self.gripper_quality_net_statistics.loss=l.item()
                    self.gripper_quality_net_statistics.update_confession_matrix(label,prediction)
                    loss += l

                if used_suction[j]:
                    label = suction_score[j]
                    if label==-1:continue
                    s_pix_A = suction_pixel_index[j, 0]
                    s_pix_B = suction_pixel_index[j, 1]
                    prediction = suction_grasp_score[j, 0, s_pix_A, s_pix_B]
                    l=binary_smooth_l1(prediction, label)

                    self.gripper_sampling_rate.update(0)

                    self.suction_quality_net_statistics.loss=l.item()
                    self.suction_quality_net_statistics.update_confession_matrix(label,prediction)

                    loss += l

            # distillation_loss=((q_value-ref_q_value)**2).mean()+((action_probs-ref_action_probs)**2).mean()
            # loss+=distillation_loss*1000


            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model_wrapper.model.parameters(),2.0)
            self.model_wrapper.optimizer.step()

            pi.step(i)
        pi.end()

    def view_result(self):
        with torch.no_grad():
            self.gripper_quality_net_statistics.print()
            self.suction_quality_net_statistics.print()

            self.gripper_sampling_rate.view()

            self.ini_policy_moving_loss.view()
            self.ini_value_moving_loss.view()
            self.representation_transfer_moving_loss.view()


    def save_statistics(self):
        self.gripper_quality_net_statistics.save()
        self.suction_quality_net_statistics.save()

        self.gripper_sampling_rate.save()

        self.ini_policy_moving_loss.save()
        self.ini_value_moving_loss.save()
        self.representation_transfer_moving_loss.save()

    def export_check_points(self):
        self.model_wrapper.export_model()
        self.model_wrapper.export_optimizer()

    def clear(self):
        self.gripper_quality_net_statistics.clear()
        self.suction_quality_net_statistics.clear()

if __name__ == "__main__":
    # seeds(0)
    lr = 1e-4
    train_action_net = TrainPolicyNet(  learning_rate=lr)
    train_action_net.initialize_model()
    train_action_net.synchronize_buffer()

    wait = wi('Begin synchronized trianing')

    # counter=0

    while True:
        new_buffer,new_data_tracker=train_action_net.synchronize_buffer()

        '''test code'''
        train_action_net.step_quality_training(max_size=10)
        # train_action_net.policy_initialization(max_size=10)
        train_action_net.export_check_points()
        train_action_net.save_statistics()
        train_action_net.view_result()
        continue

        '''online learning'''
        if new_data_tracker:
            cuda_memory_report()

            train_action_net.policy_initialization(max_size=10)
            train_action_net.step_quality_training(max_size=10)
            train_action_net.export_check_points()
            train_action_net.save_statistics()
            train_action_net.view_result()
        else:
            wait.step(0.5)

        # counter+=1