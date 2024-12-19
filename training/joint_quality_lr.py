import numpy as np
import torch
from colorama import Fore
from filelock import FileLock
from torch.utils import data

from Configurations.config import workers
from Online_data_audit.data_tracker import sample_positive_buffer, suction_grasp_tracker
from check_points.check_point_conventions import GANWrapper, ModelWrapper
from dataloaders.joint_quality_dl import JointQualityDataset
from interpolate_bin import estimate_object_mask
from lib.IO_utils import custom_print
from lib.dataset_utils import online_data
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from lib.report_utils import progress_indicator
from models.joint_grasp_sampler import GraspSampler
from models.joint_quality_networks import JointQualityNet
from records.training_satatistics import TrainingTracker
from registration import camera
from training.joint_grasp_sampler_tr import module_key as grasp_sampler_key
from training.learning_objectives.gripper_collision import gripper_collision_loss
from training.learning_objectives.shift_affordnace import shift_affordance_loss
from training.learning_objectives.suction_seal import suction_seal_loss

lock = FileLock("file.lock")
instances_per_sample=2

module_key='joint_quality_networks'
training_buffer = online_data()
training_buffer.main_modality=training_buffer.depth

print=custom_print
max_lr=0.01
min_lr=1*1e-6

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

def train(batch_size=1,n_samples=None,epochs=1,learning_rate=5e-5):
    '''load  models'''
    quality_net=ModelWrapper(model=JointQualityNet(),module_key=module_key)
    quality_net.ini_model(train=True)

    grasp_sampler =GANWrapper(module_key=grasp_sampler_key,generator=GraspSampler)
    grasp_sampler.ini_generator(train=False)

    '''optimizers'''
    quality_net.ini_adam_optimizer(learning_rate=learning_rate)

    '''dataloader'''
    file_ids=sample_positive_buffer(size=n_samples, dict_name=suction_grasp_tracker)
    print(Fore.CYAN, f'Buffer size = {len(file_ids)}')
    dataset = JointQualityDataset(data_pool=training_buffer,file_ids=file_ids)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

    suction_head_statistics = TrainingTracker(name=module_key+'_suction_head', iterations_per_epoch=len(data_loader), samples_size=len(dataset),track_label_balance=True)
    gripper_head_statistics = TrainingTracker(name=module_key+'_gripper_head', iterations_per_epoch=len(data_loader), samples_size=len(dataset),track_label_balance=True)
    shift_head_statistics = TrainingTracker(name=module_key+'_shift_head', iterations_per_epoch=len(data_loader), samples_size=len(dataset),track_label_balance=True)

    for epoch in range(epochs):

        pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(data_loader))

        for i, batch in enumerate(data_loader, 0):

            depth, normals,scores,pixel_index= batch
            depth = depth.cuda().float()  # [b,1,480.712]
            label_normals = normals.cuda().float()
            scores = scores.cuda().float()
            b = depth.shape[0]

            quality_net.model.zero_grad()
            quality_net.optimizer.zero_grad()

            '''generate grasps'''
            with torch.no_grad():
                generated_grasps, generated_normals= grasp_sampler.generator(depth.clone())

            gripper_score,suction_score,shift_score=quality_net.model(depth.clone(), generated_grasps.clone(),generated_normals.clone())

            '''loss computation'''
            suction_loss=torch.tensor(0.0,device=gripper_score.device)
            gripper_loss=torch.tensor(0.0,device=gripper_score.device)
            shift_loss=torch.tensor(0.0,device=gripper_score.device)
            for j in range(b):
                '''get parameters'''
                pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
                pc = transform_to_camera_frame(pc, reverse=True)
                normals = generated_normals[j].permute(1,2,0)[mask].cpu().numpy()
                gripper_poses=generated_grasps[j].permute(1,2,0)[mask]#.cpu().numpy()
                spatial_mask = estimate_object_mask(pc)
                suction_head_predictions=suction_score[j, 0][mask]
                gripper_head_predictions=gripper_score[j, 0][mask]
                shift_head_predictions = shift_score[j, 0][mask]

                '''limits'''
                gripper_head_max_score = torch.max(gripper_score).item()
                gripper_head_score_range = (gripper_head_max_score - torch.min(gripper_score)).item()
                suction_head_max_score = torch.max(suction_score).item()
                suction_head_score_range = (suction_head_max_score - torch.min(suction_score)).item()
                shift_head_max_score = torch.max(shift_score).item()
                shift_head_score_range = (shift_head_max_score - torch.min(shift_score)).item()

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

            loss=suction_loss+gripper_loss+shift_loss
            loss.backward()
            quality_net.optimizer.step()

            suction_head_statistics.running_loss += suction_loss.item()
            gripper_head_statistics.running_loss += gripper_loss.item()
            shift_head_statistics.running_loss += shift_loss.item()

            pi.step(i)
        pi.end()

        quality_net.export_model()
        quality_net.export_optimizer()

    suction_head_statistics.print()
    gripper_head_statistics.print()
    shift_head_statistics.print()


    suction_head_statistics.save()
    gripper_head_statistics.save()
    shift_head_statistics.save()

if __name__ == "__main__":
    for i in range(10000):
        train(batch_size=1,n_samples=300,epochs=1,learning_rate=1e-5)