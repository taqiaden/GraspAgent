import numpy as np
import torch
from colorama import Fore
from filelock import FileLock
from torch import nn
from torch.utils import data
from Configurations.config import workers
from Online_data_audit.data_tracker import sample_positive_buffer, gripper_grasp_tracker
from check_points.check_point_conventions import GANWrapper
from dataloaders.action_dl import ActionDataset
from interpolate_bin import alpha
from lib.IO_utils import custom_print
from lib.Multible_planes_detection.plane_detecttion import bin_planes_detection
from lib.cuda_utils import cuda_memory_report
from lib.dataset_utils import online_data
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from lib.loss.balanced_bce_loss import BalancedBCELoss
from lib.report_utils import progress_indicator
from models.action_net import ActionNet, Critic, action_module_key, random_approach_tensor
from records.training_satatistics import TrainingTracker, MovingRate
from registration import camera
from training.learning_objectives.grasp_sampling_evalutor import gripper_sampler_loss
from training.learning_objectives.gripper_collision import gripper_collision_loss, evaluate_grasps
from training.learning_objectives.shift_affordnace import shift_affordance_loss
from training.learning_objectives.suction_sampling_evaluator import suction_sampler_loss
from training.learning_objectives.suction_seal import suction_seal_loss

lock = FileLock("file.lock")
instances_per_sample=1

training_buffer = online_data()
training_buffer.main_modality=training_buffer.depth

bce_loss=nn.BCELoss()

balanced_bce_loss=BalancedBCELoss()
print=custom_print

def model_dependent_sampling(pc,model_predictions,model_max_score,model_score_range,objects_mask=None,maximum_iterations=1000,probability_exponent=2.0,balance_indicator=1.0,random_sampling_probability=0.003):
    for i in range(maximum_iterations):
        target_index = np.random.randint(0, pc.shape[0])
        if np.random.random() <  random_sampling_probability:break
        prediction_ = model_predictions[target_index]
        if objects_mask is not None:
            if objects_mask[target_index] == 0:
                continue
        pivot_point=np.sqrt(np.abs(balance_indicator))*np.sign(balance_indicator)
        xa=((model_max_score - prediction_).item() / model_score_range) * pivot_point
        selection_probability = ((1-pivot_point)/2 + xa+0.5*(1-abs(pivot_point)))
        selection_probability=selection_probability**probability_exponent
        if np.random.random() < selection_probability: break
    else:
        return np.random.randint(0, pc.shape[0])
    return target_index

class TrainActionNet:
    def __init__(self,batch_size=1,n_samples=None,epochs=1,learning_rate=5e-5):

        self.batch_size=batch_size
        self.n_samples=n_samples
        self.size=n_samples
        self.epochs=epochs
        self.learning_rate=learning_rate

        '''model wrapper'''
        self.gan=self.prepare_model_wrapper()

        self.moving_collision_rate=MovingRate('collision',decay_rate=0.001)
        self.moving_firmness=MovingRate('firmness',decay_rate=0.001)
        self.moving_out_of_scope=MovingRate('out_of_scope',decay_rate=0.001)

        self.data_loader=self.prepare_data_loader()

        '''initialize statistics records'''
        self.suction_head_statistics = TrainingTracker(name=action_module_key+'_suction_head', iterations_per_epoch=len(self.data_loader), samples_size=self.size,track_label_balance=True)
        self.gripper_head_statistics = TrainingTracker(name=action_module_key+'_gripper_head', iterations_per_epoch=len(self.data_loader), samples_size=self.size,track_label_balance=True)
        self.shift_head_statistics = TrainingTracker(name=action_module_key+'_shift_head', iterations_per_epoch=len(self.data_loader), samples_size=self.size,track_label_balance=True)
        self.gripper_sampler_statistics = TrainingTracker(name=action_module_key+'_gripper_sampler', iterations_per_epoch=len(self.data_loader), samples_size=self.size,track_label_balance=True)
        self.suction_sampler_statistics = TrainingTracker(name=action_module_key+'_suction_sampler', iterations_per_epoch=len(self.data_loader), samples_size=self.size,track_label_balance=True)
        self.critic_statistics = TrainingTracker(name=action_module_key+'_critic', iterations_per_epoch=len(self.data_loader), samples_size=self.size,track_label_balance=True)
        self.background_detector_statistics = TrainingTracker(name=action_module_key+'_background_detector', iterations_per_epoch=len(self.data_loader), samples_size=self.size,track_label_balance=False)

        self.swiped_samples=0

    def prepare_data_loader(self):
        file_ids = sample_positive_buffer(size=self.n_samples, dict_name=gripper_grasp_tracker,
                                          disregard_collision_samples=True)
        print(Fore.CYAN, f'Buffer size = {len(file_ids)}')
        dataset = ActionDataset(data_pool=training_buffer, file_ids=file_ids)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=workers,
                                                       shuffle=True)
        self.size=len(dataset)
        return data_loader

    def train_critic(self,gan, generated_grasps, batch_size, pixel_index, label_generated_grasps, depth,
                     collision_state_list, out_of_scope_list, firmness_state_list):

        '''concatenation'''
        with torch.no_grad():
            generated_grasps_cat = torch.cat([generated_grasps, label_generated_grasps], dim=0)
            depth_cat = depth.repeat(2, 1, 1, 1)

        '''get predictions'''
        critic_score = gan.critic(depth_cat, generated_grasps_cat)

        '''accumulate loss'''
        # curriculum_loss = 0.
        collision_loss = 0.
        firmness_loss = 0.
        for j in range(batch_size):
            pix_A = pixel_index[j, 0]
            pix_B = pixel_index[j, 1]
            prediction_ = critic_score[j, 0, pix_A, pix_B]
            label_ = critic_score[j + batch_size, 0, pix_A, pix_B]
            # print(Fore.YELLOW,f'prediction score = {prediction_.item()}, label score = {label_.item()}',Fore.RESET)

            collision_state_ = collision_state_list[j]
            out_of_scope = out_of_scope_list[j]

            bad_state_grasp = collision_state_ or out_of_scope
            # w = 10 if out_of_scope else 1
            firmness_state = firmness_state_list[j]
            # curriculum_loss += (torch.clamp(label_ - prediction_ - m1, 0))**loss_power
            collision_loss += (torch.clamp(prediction_ - label_ + 1, 0) * bad_state_grasp)  # *w
            # generated_dist = generated_grasps[j, -2, pix_A, pix_B]
            # activate_firmness_loss=1 if generated_dist<0.2 else 0.0
            # firmness_loss += (torch.clamp((prediction_ - label_) * (1 - 2 * firmness_state), 0) * (1 - bad_state_grasp))#*activate_firmness_loss
            firmness_loss += torch.clamp((prediction_ - label_), 0) * (1 - bad_state_grasp) * (1 - firmness_state)

            # print(f'col_l = {collision_loss}, firmness loss = {firmness_loss}')

        C_loss = collision_loss + firmness_loss

        # print(Fore.GREEN, 'C_loss=', C_loss.item(), Fore.RESET)

        '''optimizer step'''
        C_loss.backward()
        gan.critic_optimizer.step()
        gan.critic_optimizer.zero_grad()

        return C_loss.item()

    def train_generator(self,gan, depth, batch_size, pixel_index, collision_state_list,
                        out_of_scope_list,firmness_state_list
                        , gripper_pose, suction_direction, pcs, masks):

        '''Critic score of generated grasps'''
        generated_critic_score = gan.critic(depth.clone(), gripper_pose)

        '''accumulate loss'''
        gripper_sampling_loss = torch.tensor(0.0, device=gripper_pose.device)
        suction_sampling_loss = torch.tensor(0.0, device=gripper_pose.device)

        for j in range(batch_size):
            pc = pcs[j]
            mask = masks[j]
            gripper_loss = gripper_sampler_loss(pixel_index, j, collision_state_list[j], out_of_scope_list[j],firmness_state_list[j],  generated_critic_score)
            suction_loss = suction_sampler_loss(pc, suction_direction.permute(0, 2, 3, 1)[j][mask])
            gripper_sampling_loss += gripper_loss
            suction_sampling_loss += suction_loss  # *balance_weight

        return gripper_sampling_loss, suction_sampling_loss

    def prepare_model_wrapper(self):
        '''load  models'''
        gan = GANWrapper(action_module_key, ActionNet, Critic)
        gan.ini_models(train=True)

        '''optimizers'''
        gan.critic_sgd_optimizer(learning_rate=self.learning_rate*10)
        gan.generator_adam_optimizer(learning_rate=self.learning_rate)

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
                size = depth.shape[0] * depth.shape[1] * depth.shape[2] * depth.shape[3]
                random_approach = random_approach_tensor(size)
                gripper_pose,suction_direction,_,_,_,_,_ = self.gan.generator(depth.clone(),alpha=0.5,random_tensor=random_approach)
                '''process gripper label'''
                label_generated_grasps = gripper_pose.clone()
                for j in range(b):
                    pix_A = pixel_index[j, 0]
                    pix_B = pixel_index[j, 1]
                    label_generated_grasps[j, :, pix_A, pix_B] = pose_7[j]
                '''Evaluate generated grasps'''
                collision_state_list, firmness_state_list, out_of_scope_list = evaluate_grasps(b, pixel_index, depth,
                                                                                               gripper_pose, pose_7)
            collision_times += sum(collision_state_list)
            out_of_scope_times += sum(out_of_scope_list)
            good_firmness_times += sum(firmness_state_list)

            for k in range(len(collision_state_list)):
                self.moving_collision_rate.update(collision_state_list[k])
                self.moving_firmness.update(firmness_state_list[k])
                self.moving_out_of_scope.update(out_of_scope_list[k])

            '''zero grad'''
            self.gan.critic.zero_grad()
            self.gan.generator.zero_grad()

            '''train critic'''
            self.critic_statistics.running_loss += self.train_critic(self.gan, gripper_pose, b, pixel_index,
                                           label_generated_grasps, depth,
                                           collision_state_list, out_of_scope_list, firmness_state_list)

            '''zero grad'''
            self.gan.critic.zero_grad()
            self.gan.generator.zero_grad()

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
            gripper_pose, suction_direction, griper_collision_classifier, suction_quality_classifier, shift_affordance_classifier,background_class,_ = self.gan.generator(
                depth.clone(),alpha=0.5,random_tensor=random_approach)

            '''train generator'''
            gripper_sampling_loss,suction_sampling_loss = self.train_generator(self.gan, depth, b,pixel_index,
                                              collision_state_list, out_of_scope_list,firmness_state_list,gripper_pose,
                                                                          suction_direction,pcs,masks)
            self.gripper_sampler_statistics.running_loss+=gripper_sampling_loss.item()
            self.suction_sampler_statistics.running_loss+=suction_sampling_loss.item()

            '''loss computation'''
            suction_loss=torch.tensor(0.0,device=gripper_pose.device)
            gripper_loss=torch.tensor(0.0,device=gripper_pose.device)
            shift_loss=torch.tensor(0.0,device=gripper_pose.device)
            background_loss=torch.tensor(0.0,device=gripper_pose.device)
            non_zero_background_loss_counter=0
            for j in range(b):
                pc=pcs[j]
                mask=masks[j]

                gripper_poses=gripper_pose[j].permute(1,2,0)[mask].detach()#.cpu().numpy()
                suction_head_predictions=suction_quality_classifier[j, 0][mask]
                gripper_head_predictions=griper_collision_classifier[j, 0][mask]
                shift_head_predictions = shift_affordance_classifier[j, 0][mask]
                background_class_predictions = background_class.permute(0,2, 3, 1)[j, :, :, 0][mask]

                '''limits'''
                with torch.no_grad():
                    normals = suction_direction[j].permute(1, 2, 0)[mask].detach().cpu().numpy()
                    objects_mask = background_class_predictions.detach().cpu().numpy() <= 0.5
                    gripper_head_max_score = torch.max(griper_collision_classifier).item()
                    gripper_head_score_range = (gripper_head_max_score - torch.min(griper_collision_classifier)).item()
                    suction_head_max_score = torch.max(suction_quality_classifier).item()
                    suction_head_score_range = (suction_head_max_score - torch.min(suction_quality_classifier)).item()
                    shift_head_max_score = torch.max(shift_affordance_classifier).item()
                    shift_head_score_range = (shift_head_max_score - torch.min(shift_affordance_classifier)).item()

                '''background detection head'''
                bin_mask = bin_planes_detection(pc, sides_threshold = 0.0035,floor_threshold=0.002, view=False, file_index=file_ids[j])
                if bin_mask is not None:
                    label = torch.from_numpy(bin_mask).to(background_class_predictions.device).float()
                    background_loss += balanced_bce_loss(background_class_predictions,label,positive_weight=1.5,negative_weight=1)

                    self.background_detector_statistics.update_confession_matrix(label,background_class_predictions.detach())

                    non_zero_background_loss_counter+=1

                for k in range(instances_per_sample):
                    '''gripper collision head'''
                    gripper_target_index=model_dependent_sampling(pc, gripper_head_predictions, gripper_head_max_score, gripper_head_score_range,objects_mask,probability_exponent=10,balance_indicator=self.gripper_head_statistics.label_balance_indicator)
                    gripper_target_point = pc[gripper_target_index]
                    gripper_prediction_ = gripper_head_predictions[gripper_target_index]
                    gripper_target_pose = gripper_poses[gripper_target_index]
                    gripper_loss+=gripper_collision_loss(gripper_target_pose, gripper_target_point, pc, gripper_prediction_,self.gripper_head_statistics)

                    '''suction seal head'''
                    suction_target_index=model_dependent_sampling(pc, suction_head_predictions, suction_head_max_score, suction_head_score_range,objects_mask,probability_exponent=10,balance_indicator=self.suction_head_statistics.label_balance_indicator)
                    suction_prediction_ = suction_head_predictions[suction_target_index]
                    suction_loss+=suction_seal_loss(pc,normals,suction_target_index,suction_prediction_,self.suction_head_statistics,objects_mask)

                    '''shift affordance head'''
                    shift_target_index = model_dependent_sampling(pc, shift_head_predictions, shift_head_max_score,shift_head_score_range,probability_exponent=10,balance_indicator=self.shift_head_statistics.label_balance_indicator)
                    shift_target_point = pc[shift_target_index]
                    shift_prediction_=shift_head_predictions[shift_target_index]
                    shift_loss+=shift_affordance_loss(pc,shift_target_point,objects_mask,self.shift_head_statistics,shift_prediction_,normals,shift_target_index)

            decay_= lambda scores:torch.clamp(scores-torch.zeros_like(scores),0).mean()
            # reversed_decay_= lambda scores:torch.clamp(torch.ones_like(scores)-scores,0).mean()

            decay_loss=decay_(griper_collision_classifier)+decay_(suction_quality_classifier)+decay_(shift_affordance_classifier)
            decay_loss*=0.3
            # reversed_decay_loss=reversed_decay_(background_class)*0.3

            if non_zero_background_loss_counter>0: background_loss/non_zero_background_loss_counter

            loss=suction_loss+gripper_loss+shift_loss+gripper_sampling_loss*10+suction_sampling_loss*50+decay_loss+background_loss*30
            loss.backward()
            self.gan.generator_optimizer.step()
            self.gan.generator_optimizer.zero_grad()

            with torch.no_grad():
                self.suction_head_statistics.running_loss += suction_loss.item()
                self.gripper_head_statistics.running_loss += gripper_loss.item()
                self.shift_head_statistics.running_loss += shift_loss.item()
                self.background_detector_statistics.running_loss+=background_loss.item()

            self.swiped_samples+=b
            if i%100==0 and i!=0:
                self.view_result(gripper_pose,collision_times,good_firmness_times,out_of_scope_times)
                self.export_check_points()

            pi.step(i)
        pi.end()

        self.view_result(gripper_pose,collision_times,good_firmness_times,out_of_scope_times)

        self.export_check_points()

    def view_result(self,gripper_pose,collision_times,good_firmness_times,out_of_scope_times):
        with torch.no_grad():
            self.suction_head_statistics.print(self.swiped_samples)
            self.gripper_head_statistics.print(self.swiped_samples)
            self.shift_head_statistics.print(self.swiped_samples)
            self.gripper_sampler_statistics.print(self.swiped_samples)
            self.suction_sampler_statistics.print(self.swiped_samples)
            self.critic_statistics.print(self.swiped_samples)
            self.background_detector_statistics.print(self.swiped_samples)
            values = gripper_pose.permute(1, 0, 2, 3).flatten(1).detach()
            std = torch.std(values, dim=-1)

            print(f'gripper_pose std = {std.detach()}')
            print(f'Collision ratio = {collision_times / self.swiped_samples}')
            print(f'firm grasp ratio = {good_firmness_times / self.swiped_samples}')
            print(f'out of scope ratio = {out_of_scope_times / self.swiped_samples}')

            self.moving_collision_rate.view()
            self.moving_firmness.view()
            self.moving_out_of_scope.view()

    def export_check_points(self):
        self.gan.export_models()
        self.gan.export_optimizers()

        self.moving_collision_rate.save()
        self.moving_firmness.save()
        self.moving_out_of_scope.save()

        self.suction_head_statistics.save()
        self.gripper_head_statistics.save()
        self.shift_head_statistics.save()
        self.critic_statistics.save()
        # self.background_detector_statistics.save()

if __name__ == "__main__":

    for i in range(1000):
        #cuda_memory_report()

        lr=1e-5
        # train_action_net = TrainActionNet(batch_size=2, n_samples=None, learning_rate=lr)
        # train_action_net.begin()
        try:
            cuda_memory_report()
            train_action_net=TrainActionNet(batch_size=2, n_samples=None, learning_rate=lr)
            train_action_net.begin()
        except Exception as error_message:
            torch.cuda.empty_cache()
            print(error_message)