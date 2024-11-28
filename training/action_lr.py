import torch
from colorama import Fore
from torch.utils import data
from Configurations.config import workers, theta_cos_scope
from Online_data_audit.data_tracker import sample_positive_buffer,  gripper_grasp_tracker
from check_points.check_point_conventions import GANWrapper
from dataloaders.action_dl import ActionDataset
from interpolate_bin import estimate_object_mask
from lib.IO_utils import   custom_print
from lib.collision_unit import grasp_collision_detection
from lib.dataset_utils import online_data
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds, pixel_to_point
from models.action_net import ActionNet, Critic
from pose_object import pose_7_to_transformation
from records.training_satatistics import TrainingTracker
from registration import camera
from lib.report_utils import  progress_indicator
from filelock import FileLock

from training.learning_objectives.grasp_sampling_evalutor import gripper_sampler_loss
from training.learning_objectives.gripper_collision import gripper_collision_loss
from training.learning_objectives.shift_affordnace import  shift_affordance_loss
from training.learning_objectives.suction_sampling_evaluator import suction_sampler_loss
from training.learning_objectives.suction_seal import suction_seal_loss
import numpy as np
import torch.nn.functional as F

lock = FileLock("file.lock")
instances_per_sample=1

module_key='action_net'
training_buffer = online_data()
training_buffer.main_modality=training_buffer.depth

print=custom_print

def model_dependent_sampling(pc,model_predictions,model_max_score,model_score_range,spatial_mask=None,maximum_iterations=1000,probability_exponent=2.0,balance_indicator=1.0):
    for i in range(maximum_iterations):
        target_index = np.random.randint(0, pc.shape[0])
        prediction_ = model_predictions[target_index]
        if spatial_mask is not None:
            if spatial_mask[target_index] == 0: continue
        xa=((model_max_score - prediction_).item() / model_score_range) * balance_indicator
        selection_probability = ((1-balance_indicator)/2 + xa+0.5*(1-abs(balance_indicator)))
        selection_probability=selection_probability**probability_exponent
        if np.random.random() < selection_probability: break
    else:
        return np.random.randint(0, pc.shape[0])
    return target_index

def evaluate_grasps(batch_size,pixel_index,depth,generated_grasps,pose_7):
    '''Evaluate generated grasps'''
    collision_state_list = []
    firmness_state_list = []
    out_of_scope_list = []
    for j in range(batch_size):
        pix_A = pixel_index[j, 0]
        pix_B = pixel_index[j, 1]
        depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()

        '''get pose parameters'''
        target_point = pixel_to_point(pixel_index[j].cpu().numpy(), depth_value, camera)
        target_point = transform_to_camera_frame(target_point[None, :], reverse=True)[0]

        target_pose=generated_grasps[j, :, pix_A, pix_B]
        T_d, width, distance=pose_7_to_transformation(target_pose, target_point)
        # if j == 0: print(f'Example _pose = {generated_grasps[j, :, pix_A, pix_B]}')

        '''get point clouds'''
        pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
        pc = transform_to_camera_frame(pc, reverse=True)

        '''check collision'''
        collision_intensity = grasp_collision_detection(T_d,width, pc, visualize=False )
        collision_state_=collision_intensity > 0
        collision_state_list.append(collision_state_)

        '''check parameters are within scope'''
        ref_approach = torch.tensor([0.0, 0.0, 1.0],device=target_pose.device)  # vertical direction

        approach_cos = F.cosine_similarity(target_pose[0:3], ref_approach, dim=0)
        in_scope = 1.0 > generated_grasps[j, -2, pix_A, pix_B] > 0.0 and 1.0 > generated_grasps[
            j, -1, pix_A, pix_B] > 0.0 and approach_cos > theta_cos_scope
        out_of_scope_list.append(not in_scope)

        '''check firmness'''
        label_dist = pose_7[j, -2]
        generated_dist = generated_grasps[j, -2, pix_A, pix_B]
        firmness_=1 if generated_dist.item() > label_dist.item() and not collision_state_ and in_scope else 0
        firmness_state_list.append(firmness_)

    return collision_state_list,firmness_state_list,out_of_scope_list

def train_critic(gan,generated_grasps,batch_size,pixel_index,label_generated_grasps,depth,
                 collision_state_list,out_of_scope_list,firmness_state_list):
    '''zero grad'''
    gan.critic.zero_grad()
    gan.generator.zero_grad()

    '''concatenation'''
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
        firmness_state = firmness_state_list[j]
        # curriculum_loss += (torch.clamp(label_ - prediction_ - m1, 0))**loss_power
        collision_loss += (torch.clamp(prediction_ - label_ + 1, 0) * bad_state_grasp)
        generated_dist = generated_grasps[j, -2, pix_A, pix_B]
        #activate_firmness_loss=1 if generated_dist<0.2 else 0.0
        firmness_loss += (torch.clamp((prediction_ - label_) * (1 - 2 * firmness_state), 0) * (1 - bad_state_grasp))#*activate_firmness_loss
        # firmness_loss += torch.clamp((prediction_ - label_) , 0) * (1 - bad_state_grasp)

        # print(f'col_l = {collision_loss}, firmness loss = {firmness_loss}')

    C_loss = collision_loss**2+firmness_loss**2

    # print(Fore.GREEN, 'C_loss=', C_loss.item(), Fore.RESET)

    '''optimizer step'''
    C_loss.backward()
    gan.critic_optimizer.step()
    gan.critic_optimizer.zero_grad()

    return C_loss.item()

def train_generator(gan,depth,label_generated_grasps,batch_size,pixel_index,collision_state_list,out_of_scope_list,gripper_pose,suction_direction):
    '''critic score of reference label '''
    with torch.no_grad():
        label_critic_score = gan.critic(depth.clone(), label_generated_grasps)

    '''Critic score of generated grasps'''
    generated_critic_score = gan.critic(depth.clone(), gripper_pose)

    '''accumulate loss'''
    gripper_sampling_loss = torch.tensor(0.0,device=gripper_pose.device)
    suction_sampling_loss = torch.tensor(0.0,device=gripper_pose.device)

    for j in range(batch_size):
        gripper_loss=gripper_sampler_loss(pixel_index,j,collision_state_list,out_of_scope_list,label_critic_score,generated_critic_score)
        suction_loss=suction_sampler_loss(depth, j, suction_direction.permute(0, 2, 3, 1))
        balance_weight=1 if suction_loss<=gripper_loss else gripper_loss/suction_loss
        gripper_sampling_loss+=gripper_loss
        suction_sampling_loss+=suction_loss*balance_weight

    return gripper_sampling_loss,suction_sampling_loss

def train(batch_size=1,n_samples=None,epochs=1,learning_rate=5e-5):
    '''load  models'''
    gan=GANWrapper(module_key,ActionNet,Critic)
    gan.ini_models(train=True)

    '''optimizers'''
    gan.critic_sgd_optimizer(learning_rate=learning_rate)
    gan.generator_adam_optimizer(learning_rate=learning_rate)

    '''dataloader'''
    file_ids=sample_positive_buffer(size=n_samples, dict_name=gripper_grasp_tracker)
    print(Fore.CYAN, f'Buffer size = {len(file_ids)}')
    dataset = ActionDataset(data_pool=training_buffer,file_ids=file_ids)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

    suction_head_statistics = TrainingTracker(name=module_key+'_suction_head', iterations_per_epoch=len(data_loader), samples_size=len(dataset),track_label_balance=True)
    gripper_head_statistics = TrainingTracker(name=module_key+'_gripper_head', iterations_per_epoch=len(data_loader), samples_size=len(dataset),track_label_balance=True)
    shift_head_statistics = TrainingTracker(name=module_key+'_shift_head', iterations_per_epoch=len(data_loader), samples_size=len(dataset),track_label_balance=True)
    gripper_sampler_statistics = TrainingTracker(name=module_key+'_gripper_sampler', iterations_per_epoch=len(data_loader), samples_size=len(dataset),track_label_balance=True)
    suction_sampler_statistics = TrainingTracker(name=module_key+'_suction_sampler', iterations_per_epoch=len(data_loader), samples_size=len(dataset),track_label_balance=True)
    critic_statistics = TrainingTracker(name=module_key+'_critic', iterations_per_epoch=len(data_loader), samples_size=len(dataset),track_label_balance=True)

    collision_times = 0.
    out_of_scope_times = 0.
    good_firmness_times = 0.

    pi = progress_indicator('Begin new training round: ', max_limit=len(data_loader))

    for i, batch in enumerate(data_loader, 0):
        depth,pose_7,pixel_index= batch
        depth = depth.cuda().float()  # [b,1,480.712]
        pose_7 = pose_7.cuda().float()
        b = depth.shape[0]

        '''generate grasps'''
        with torch.no_grad():
            gripper_pose,suction_direction,_,_,_ = gan.generator(depth.clone())

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

        '''train critic'''
        critic_statistics.running_loss += train_critic(gan, gripper_pose, b, pixel_index,
                                       label_generated_grasps, depth,
                                       collision_state_list, out_of_scope_list, firmness_state_list)

        '''zero grad'''
        gan.critic.zero_grad()
        gan.generator.zero_grad()

        '''generated grasps'''
        gripper_pose, suction_direction, griper_collision_classifier, suction_quality_classifier, shift_affordance_classifier = gan.generator(
            depth.clone())

        '''train generator'''
        gripper_sampling_loss,suction_sampling_loss = train_generator(gan, depth, label_generated_grasps, b,
                                          pixel_index,
                                          collision_state_list, out_of_scope_list,gripper_pose,suction_direction)
        gripper_sampler_statistics.running_loss+=gripper_sampling_loss.item()
        suction_sampler_statistics.running_loss+=suction_sampling_loss.item()

        '''loss computation'''
        suction_loss=torch.tensor(0.0,device=gripper_pose.device)
        gripper_loss=torch.tensor(0.0,device=gripper_pose.device)
        shift_loss=torch.tensor(0.0,device=gripper_pose.device)
        for j in range(b):
            '''get parameters'''
            pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
            pc = transform_to_camera_frame(pc, reverse=True)
            normals = suction_direction[j].permute(1,2,0)[mask].detach().cpu().numpy()
            gripper_poses=gripper_pose[j].permute(1,2,0)[mask]#.cpu().numpy()
            spatial_mask = estimate_object_mask(pc)
            suction_head_predictions=suction_quality_classifier[j, 0][mask]
            gripper_head_predictions=griper_collision_classifier[j, 0][mask]
            shift_head_predictions = shift_affordance_classifier[j, 0][mask]

            '''limits'''
            gripper_head_max_score = torch.max(griper_collision_classifier).item()
            gripper_head_score_range = (gripper_head_max_score - torch.min(griper_collision_classifier)).item()
            suction_head_max_score = torch.max(suction_quality_classifier).item()
            suction_head_score_range = (suction_head_max_score - torch.min(suction_quality_classifier)).item()
            shift_head_max_score = torch.max(shift_affordance_classifier).item()
            shift_head_score_range = (shift_head_max_score - torch.min(shift_affordance_classifier)).item()

            '''view suction scores'''
            for k in range(instances_per_sample):
                '''gripper head'''
                gripper_target_index=model_dependent_sampling(pc, gripper_head_predictions, gripper_head_max_score, gripper_head_score_range,spatial_mask,probability_exponent=10,balance_indicator=gripper_head_statistics.label_balance_indicator)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = gripper_head_predictions[gripper_target_index]
                gripper_target_pose = gripper_poses[gripper_target_index]
                gripper_loss+=gripper_collision_loss(gripper_target_pose, gripper_target_point, pc, gripper_prediction_,gripper_head_statistics)

                '''suction head'''
                suction_target_index=model_dependent_sampling(pc, suction_head_predictions, suction_head_max_score, suction_head_score_range,spatial_mask,probability_exponent=10,balance_indicator=suction_head_statistics.label_balance_indicator)
                suction_target_point = pc[suction_target_index]
                suction_prediction_ = suction_head_predictions[suction_target_index]
                suction_loss+=suction_seal_loss(suction_target_point,pc,normals,suction_target_index,suction_prediction_,suction_head_statistics)

                '''shift head'''
                shift_target_index = model_dependent_sampling(pc, shift_head_predictions, shift_head_max_score,shift_head_score_range,probability_exponent=10,balance_indicator=shift_head_statistics.label_balance_indicator)
                shift_target_point = pc[shift_target_index]
                shift_prediction_=shift_head_predictions[shift_target_index]
                shift_loss+=shift_affordance_loss(pc,shift_target_point,spatial_mask,shift_head_statistics,shift_prediction_)

        loss=suction_loss+gripper_loss+shift_loss+gripper_sampling_loss+suction_sampling_loss
        loss.backward()
        gan.generator_optimizer.step()

        suction_head_statistics.running_loss += suction_loss.item()
        gripper_head_statistics.running_loss += gripper_loss.item()
        shift_head_statistics.running_loss += shift_loss.item()

        pi.step(i)
    pi.end()

    gan.export_models()
    gan.export_optimizers()

    suction_head_statistics.print()
    gripper_head_statistics.print()
    shift_head_statistics.print()
    gripper_sampler_statistics.print()
    suction_sampler_statistics.print()
    size = len(file_ids)
    print(f'Collision ratio = {collision_times / size}')
    print(f'out of scope ratio = {out_of_scope_times / size}')
    print(f'firm grasp ratio = {good_firmness_times / size}')

    suction_head_statistics.save()
    gripper_head_statistics.save()
    shift_head_statistics.save()

if __name__ == "__main__":
    for i in range(10000):
        train(batch_size=1,n_samples=100,learning_rate=1e-5)