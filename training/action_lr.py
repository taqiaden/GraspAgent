import numpy as np
import torch
from colorama import Fore
from filelock import FileLock
from torch import nn
from Configurations.config import workers
from Online_data_audit.data_tracker import sample_positive_buffer, gripper_grasp_tracker, DataTracker
from analytical_suction_sampler import estimate_suction_direction
from check_points.check_point_conventions import GANWrapper
from dataloaders.action_dl import ActionDataset
from lib.IO_utils import custom_print
from lib.Multible_planes_detection.plane_detecttion import bin_planes_detection
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from lib.loss.balanced_bce_loss import BalancedBCELoss
from lib.report_utils import progress_indicator
from models.action_net import ActionNet, Critic, action_module_key
from records.training_satatistics import TrainingTracker, MovingRate, truncate
from registration import camera
from training.learning_objectives.gripper_collision import gripper_collision_loss, evaluate_grasps
from training.learning_objectives.shift_affordnace import shift_affordance_loss
from training.learning_objectives.suction_seal import suction_seal_loss

detach_backbone=False

lock = FileLock("file.lock")

training_buffer = online_data()
training_buffer.main_modality=training_buffer.depth

bce_loss=nn.BCELoss()

balanced_bce_loss=BalancedBCELoss()
print=custom_print

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)

def suction_sampler_loss(pc,target_normal):

    labels = estimate_suction_direction(pc, view=False)  # inference time on local computer = 1.3 s
    labels = torch.from_numpy(labels).to('cuda')

    return ((1 - cos(target_normal, labels.squeeze())) ** 2).mean()

def gripper_sampler_loss(pixel_index,j,generated_critic_score,critic_score_label,enable_instance_training,firmness_state):
    pix_A = pixel_index[j, 0]
    pix_B = pixel_index[j, 1]

    prediction_score = generated_critic_score[j, 0, pix_A, pix_B]
    loss = torch.clamp(critic_score_label - prediction_score,0)**2

    return loss

def model_dependent_sampling(pc,model_predictions,model_max_score,model_score_range,objects_mask=None,maximum_iterations=10000,probability_exponent=2.0,balance_indicator=1.0,random_sampling_probability=0.003):
    for i in range(maximum_iterations):
        if objects_mask is None:
            target_index = np.random.randint(0, pc.shape[0])
        else:
            idx_nonzero,=np.nonzero(objects_mask)
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

def step_critic_training(gan, generated_grasps, batch_size, pixel_index, label_generated_grasps, depth,
                         collision_state_list, out_of_scope_list, firmness_state_list, alpha, beta, firmness_weight):
    '''concatenation'''
    with torch.no_grad():
        generated_grasps_cat = torch.cat([generated_grasps, label_generated_grasps], dim=0)
        depth_cat = depth.repeat(2, 1, 1, 1)

    '''get predictions'''
    critic_score = gan.critic(depth_cat, generated_grasps_cat)

    '''accumulate loss'''
    collision_loss = 0.
    firmness_loss = 0.
    curriculum_loss=0.
    # c_loss=0.0
    critic_score_labels=[]
    for j in range(batch_size):
        pix_A = pixel_index[j, 0]
        pix_B = pixel_index[j, 1]
        prediction_ = critic_score[j, 0, pix_A, pix_B]
        label_ = critic_score[j + batch_size, 0, pix_A, pix_B]
        critic_score_labels.append(label_.detach().clone())
        # print(prediction_, '---',label_)

        bad_state_grasp = collision_state_list[j] or out_of_scope_list[j]
        firmness_state = firmness_state_list[j]

        collision_loss += (torch.clamp(prediction_ - label_ +1., 0.) * bad_state_grasp)**2
        # collision_loss += ((prediction_ - label_ +1.) * bad_state_grasp)**2

        # p_=int(np.random.rand()<(beta**2))

        firmness_loss += (torch.clamp((prediction_ - label_), 0.) * (1 - bad_state_grasp) * (1 - firmness_state) )**2
        firmness_loss += (torch.clamp((label_ - prediction_), 0.) * (1 - bad_state_grasp) * firmness_state )**2

        # firmness_loss += ((prediction_ - label_+1) * (1 - bad_state_grasp) * (1 - firmness_state) *p_)**2
        # firmness_loss += ((label_ - prediction_+1)* (1 - bad_state_grasp) * firmness_state *p_)**2

    c_loss = ( collision_loss + firmness_loss )  #* 10*(alpha*beta) # + curriculum_loss

    '''optimizer step'''
    c_loss.backward()
    gan.critic_optimizer.step()
    gan.critic_optimizer.zero_grad()
    return c_loss.item(), critic_score_labels

class TrainActionNet:
    def __init__(self,batch_size=1,n_samples=None,epochs=1,learning_rate=5e-5):

        self.batch_size=batch_size
        self.n_samples=n_samples
        self.size=n_samples
        self.epochs=epochs
        self.learning_rate=learning_rate

        '''model wrapper'''
        self.gan=self.prepare_model_wrapper()
        self.data_loader=None

        '''Moving rates'''
        self.moving_collision_rate=None
        self.moving_firmness=None
        self.moving_out_of_scope=None

        '''initialize statistics records'''
        self.suction_head_statistics = None
        self.bin_collision_statistics = None
        self.objects_collision_statistics=None
        self.shift_head_statistics = None
        self.gripper_sampler_statistics = None
        self.suction_sampler_statistics = None
        self.critic_statistics = None
        self.background_detector_statistics = None

        self.data_tracker = None
        self.swiped_samples=None

    def initialize(self,n_samples=None):
        self.n_samples=n_samples

        self.prepare_data_loader()

        '''Moving rates'''
        self.moving_collision_rate=MovingRate('collision',decay_rate=0.0001)
        self.moving_firmness=MovingRate('firmness',decay_rate=0.0001)
        self.moving_out_of_scope=MovingRate('out_of_scope',decay_rate=0.0001)
        '''initialize statistics records'''
        self.suction_head_statistics = TrainingTracker(name=action_module_key+'_suction_head', iterations_per_epoch=len(self.data_loader), track_label_balance=True)
        self.bin_collision_statistics = TrainingTracker(name=action_module_key+ '_bin_collision',
                                                        iterations_per_epoch=len(self.data_loader),
                                                        track_label_balance=True)
        self.objects_collision_statistics = TrainingTracker(name=action_module_key+ '_objects_collision',
                                                            iterations_per_epoch=len(self.data_loader),
                                                            track_label_balance=True)
        self.shift_head_statistics = TrainingTracker(name=action_module_key+'_shift_head', iterations_per_epoch=len(self.data_loader), track_label_balance=True)
        self.gripper_sampler_statistics = TrainingTracker(name=action_module_key+'_gripper_sampler', iterations_per_epoch=len(self.data_loader), track_label_balance=False)
        self.suction_sampler_statistics = TrainingTracker(name=action_module_key+'_suction_sampler', iterations_per_epoch=len(self.data_loader), track_label_balance=False)
        self.critic_statistics = TrainingTracker(name=action_module_key+'_critic', iterations_per_epoch=len(self.data_loader), track_label_balance=False)
        self.background_detector_statistics = TrainingTracker(name=action_module_key+'_background_detector', iterations_per_epoch=len(self.data_loader), track_label_balance=False)

        self.data_tracker = DataTracker(name=gripper_grasp_tracker)
        self.swiped_samples=0

    def prepare_data_loader(self):
        file_ids = sample_positive_buffer(size=self.n_samples, dict_name=gripper_grasp_tracker,
                                          disregard_collision_samples=True,sample_with_probability=False)
        print(Fore.CYAN, f'Buffer size = {len(file_ids)}',Fore.RESET)
        dataset = ActionDataset(data_pool=training_buffer, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=workers,
                                                       shuffle=True)
        self.size=len(dataset)
        self.data_loader= data_loader

    def prepare_model_wrapper(self):
        '''load  models'''
        gan = GANWrapper(action_module_key, ActionNet, Critic)
        gan.ini_models(train=True)

        '''optimizers'''
        gan.critic_sgd_optimizer(learning_rate=self.learning_rate*100)
        # gan.critic_rmsprop_optimizer(learning_rate=self.learning_rate)
        # gan.critic_adam_optimizer(learning_rate=self.learning_rate*10,beta1=0.5)

        gan.generator_adam_optimizer(learning_rate=self.learning_rate,beta1=0.9)

        return gan

    def begin(self):
        collision_times = 0.
        out_of_scope_times = 0.
        good_firmness_times = 0.

        pi = progress_indicator('Begin new training round: ', max_limit=len(self.data_loader))
        gripper_pose=None
        for i, batch in enumerate(self.data_loader, 0):
            depth,pose_7,pixel_index,file_ids= batch
            depth = depth.cuda().float()  # [b,1,480.712]
            pose_7 = pose_7.cuda().float()
            b = depth.shape[0]

            '''generate grasps'''
            with torch.no_grad():
                gripper_pose,suction_direction,_,_,_,_,_ = self.gan.generator(depth.clone(),alpha=0.0)

                '''process gripper label'''
                label_generated_grasps = gripper_pose.clone()
                for j in range(b):
                    pix_A = pixel_index[j, 0]
                    pix_B = pixel_index[j, 1]
                    label_generated_grasps[j, :, pix_A, pix_B] = pose_7[j]
                '''Evaluate generated grasps'''
                collision_state_list, firmness_state_list, out_of_scope_list = evaluate_grasps(b, pixel_index, depth,
                                                                                               gripper_pose, pose_7,visualize=False)

                '''update metrics'''
                collision_times += sum(collision_state_list)
                out_of_scope_times += sum(out_of_scope_list)
                good_firmness_times += sum(firmness_state_list)
                enable_instance_training = [c or o or not f for c, o,f in zip(collision_state_list, out_of_scope_list,firmness_state_list)]

                for k in range(len(collision_state_list)):
                    d_=int(collision_state_list[k] or out_of_scope_list[k])
                    self.data_tracker.update_value(file_id=file_ids[k],list_index=3,data=d_,decay_rate=0.1)
                    self.moving_collision_rate.update(collision_state_list[k])
                    self.moving_firmness.update(firmness_state_list[k])
                    self.moving_out_of_scope.update(out_of_scope_list[k])

            '''zero grad'''
            self.gan.critic.zero_grad()
            self.gan.generator.zero_grad()

            '''train critic'''
            alpha=self.moving_collision_rate.val+self.moving_out_of_scope.val
            beta = 1-self.moving_firmness.val
            firmness_weight =np.tanh(11*beta-9)*0.5+0.5

            l_c ,critic_score_labels= step_critic_training(self.gan, gripper_pose, b, pixel_index,
                                           label_generated_grasps, depth,
                                           collision_state_list, out_of_scope_list, firmness_state_list,alpha=alpha,beta=beta,firmness_weight=firmness_weight)


            self.critic_statistics.loss=l_c/b

            '''zero grad'''
            self.gan.critic.zero_grad()
            self.gan.generator.zero_grad()
            self.swiped_samples+=b

            pcs=[]
            masks=[]
            # spatial_masks=[]
            for j in range(b):
                '''get parameters'''
                pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)
                pcs.append(pc)
                masks.append(mask)

            '''generated grasps'''
            gripper_pose, suction_direction, griper_collision_classifier, suction_quality_classifier, shift_affordance_classifier,background_class,depth_features = self.gan.generator(
                depth.clone(),alpha=0.0,detach_backbone=detach_backbone)

            '''loss computation'''
            suction_loss=suction_quality_classifier.mean()*0.0
            gripper_loss=griper_collision_classifier.mean()*0.0
            shift_loss=shift_affordance_classifier.mean()*0.0
            background_loss=background_class.mean()*0.0
            decay_loss=torch.tensor(0.0,device=gripper_pose.device)
            suction_sampling_loss = suction_direction.mean()*0.0
            gripper_sampling_loss = gripper_pose.mean()*0.0

            non_zero_background_loss_counter=0

            effective_batch_size=b if detach_backbone else sum(enable_instance_training)
            decay_= lambda scores:torch.clamp(scores,0).mean()*0.1

            for j in range(b):
                if enable_instance_training[j]==0 and not detach_backbone:continue
                pc=pcs[j]
                mask=masks[j]

                generated_critic_score = self.gan.critic(depth.clone(), gripper_pose, detach_backbone=True)


                gripper_sampling_loss += gripper_sampler_loss(pixel_index, j,
                                                              generated_critic_score, critic_score_labels[j],enable_instance_training[j],firmness_state_list[j]) #* int(counted_samples[j]==1)
                gripper_sampling_loss=gripper_sampling_loss/effective_batch_size

                suction_sampling_loss += suction_sampler_loss(pc, suction_direction.permute(0, 2, 3, 1)[j][mask])
                suction_sampling_loss=suction_sampling_loss/effective_batch_size

                gripper_poses=gripper_pose[j].permute(1,2,0)[mask].detach()#.cpu().numpy()
                suction_head_predictions=suction_quality_classifier[j, 0][mask]
                gripper_head_predictions=griper_collision_classifier[j, :].permute(1,2,0)[mask]
                shift_head_predictions = shift_affordance_classifier[j, 0][mask]
                background_class_predictions = background_class.permute(0,2, 3, 1)[j, :, :, 0][mask]

                # decay_loss += (decay_(gripper_head_predictions) + decay_(suction_head_predictions) + decay_(
                #     shift_head_predictions))/effective_batch_size

                '''limits'''
                with torch.no_grad():
                    normals = suction_direction[j].permute(1, 2, 0)[mask].detach().cpu().numpy()
                    objects_mask = background_class_predictions.detach().cpu().numpy() <= 0.5
                    gripper_head_max_score = torch.max(griper_collision_classifier[:, i % 2]).item()
                    gripper_head_score_range = (
                                gripper_head_max_score - torch.min(griper_collision_classifier[:, i % 2])).item()
                    suction_head_max_score = torch.max(suction_quality_classifier).item()
                    suction_head_score_range = (suction_head_max_score - torch.min(suction_quality_classifier)).item()
                    shift_head_max_score = torch.max(shift_affordance_classifier).item()
                    shift_head_score_range = (shift_head_max_score - torch.min(shift_affordance_classifier)).item()


                '''background detection head'''
                try:
                    bin_mask = bin_planes_detection(pc, sides_threshold = 0.0035,floor_threshold=0.002, view=False, file_index=file_ids[j])
                except Exception as error_message:
                    print(file_ids[j])
                    print(error_message)
                    bin_mask=None

                if bin_mask is not None:
                    label = torch.from_numpy(bin_mask).to(background_class_predictions.device).float()
                    background_loss += balanced_bce_loss(background_class_predictions,label,positive_weight=1.5,negative_weight=1)
                    self.background_detector_statistics.update_confession_matrix(label,background_class_predictions.detach())
                    non_zero_background_loss_counter+=1

                n_g=10
                for k in range(n_g):
                    '''gripper collision head'''
                    sta = self.objects_collision_statistics if i % 2 == 0 else self.bin_collision_statistics
                    gripper_target_index = model_dependent_sampling(pc, gripper_head_predictions[:, i % 2],
                                                                    gripper_head_max_score, gripper_head_score_range,
                                                                    objects_mask, probability_exponent=10,
                                                                    balance_indicator=sta.label_balance_indicator)
                    gripper_target_point = pc[gripper_target_index]
                    gripper_prediction_ = gripper_head_predictions[gripper_target_index]
                    gripper_target_pose = gripper_poses[gripper_target_index]
                    gripper_loss += gripper_collision_loss(gripper_target_pose, gripper_target_point, pc, objects_mask,
                                                           gripper_prediction_, self.objects_collision_statistics,
                                                           self.bin_collision_statistics) / (n_g*effective_batch_size)

                n_sh=10
                for k in range(n_sh):
                    '''suction seal head'''
                    suction_target_index=model_dependent_sampling(pc, suction_head_predictions, suction_head_max_score, suction_head_score_range,objects_mask,probability_exponent=10,balance_indicator=self.suction_head_statistics.label_balance_indicator)
                    suction_prediction_ = suction_head_predictions[suction_target_index]
                    suction_loss+=suction_seal_loss(pc,normals,suction_target_index,suction_prediction_,self.suction_head_statistics,objects_mask)/(n_sh*effective_batch_size)

                n_s=10
                for k in range(n_s):
                    '''shift affordance head'''
                    shift_target_index = model_dependent_sampling(pc, shift_head_predictions, shift_head_max_score,shift_head_score_range,probability_exponent=10,balance_indicator=self.shift_head_statistics.label_balance_indicator)
                    shift_target_point = pc[shift_target_index]
                    shift_prediction_=shift_head_predictions[shift_target_index]
                    shift_loss+=shift_affordance_loss(pc,shift_target_point,objects_mask,self.shift_head_statistics,shift_prediction_,normals,shift_target_index)/(n_s*effective_batch_size)

            if non_zero_background_loss_counter>0: background_loss/non_zero_background_loss_counter

            if i%5==0:print(f'c_loss={truncate(l_c)}, g_loss={truncate(gripper_sampling_loss.item())} alpha = {truncate(alpha)}, beta = {truncate(beta)}, firmness weight = {truncate(firmness_weight)}')

            loss=suction_loss*0.1+gripper_loss*0.5+shift_loss*0.3+decay_loss*0.1+gripper_sampling_loss*2.0+suction_sampling_loss+background_loss*3.0
            loss.backward()
            self.gan.generator_optimizer.step()
            self.gan.generator_optimizer.zero_grad()

            with torch.no_grad():
                self.gripper_sampler_statistics.loss = gripper_sampling_loss.item()
                self.suction_sampler_statistics.loss = suction_sampling_loss.item()
                self.suction_head_statistics.loss = suction_loss.item()
                self.shift_head_statistics.loss = shift_loss.item()
                self.background_detector_statistics.loss=background_loss.item()

            if i%50==0 and i!=0:
                self.view_result(gripper_pose,collision_times,good_firmness_times,out_of_scope_times)
                self.export_check_points()
                self.save_statistics()

            pi.step(i)
        pi.end()

        self.view_result(gripper_pose,collision_times,good_firmness_times,out_of_scope_times)

        self.export_check_points()
        self.clear()

    def view_result(self,gripper_pose,collision_times,good_firmness_times,out_of_scope_times):
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
            std = torch.std(values, dim=-1)

            print(f'gripper_pose std = {std.detach().cpu()}')
            print(f'Collision ratio = {collision_times / self.swiped_samples}')
            print(f'firm grasp ratio = {good_firmness_times / self.swiped_samples}')
            print(f'out of scope ratio = {out_of_scope_times / self.swiped_samples}')

            self.moving_collision_rate.view()
            self.moving_firmness.view()
            self.moving_out_of_scope.view()

    def save_statistics(self):
        self.moving_collision_rate.save()
        self.moving_firmness.save()
        self.moving_out_of_scope.save()
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
    lr = 1e-6
    train_action_net = TrainActionNet(batch_size=2, n_samples=None, learning_rate=lr)
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

        # lr=max(lr/1.1,1e-6)