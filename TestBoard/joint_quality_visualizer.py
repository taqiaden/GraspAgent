import numpy as np
import torch
from colorama import Fore
from filelock import FileLock
from torch import nn
from torch.utils import data

from Configurations.config import workers
from Online_data_audit.data_tracker import suction_grasp_tracker, sample_random_buffer
from check_points.check_point_conventions import GANWrapper, ModelWrapper
from dataloaders.joint_quality_dl import JointQualityDataset
from interpolate_bin import estimate_object_mask
from lib.IO_utils import custom_print
from lib.dataset_utils import online_data
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from models.joint_grasp_sampler import GraspSampler
from models.joint_quality_networks import JointQualityNet
from records.training_satatistics import TrainingTracker
from registration import camera
from training.joint_grasp_sampler_tr import module_key as grasp_sampler_key
from training.joint_quality_lr import model_dependent_sampling
from training.learning_objectives.gripper_collision import gripper_collision_loss
from training.learning_objectives.shift_affordnace import shift_affordance_loss
from training.learning_objectives.suction_seal import suction_seal_loss
from visualiztion import view_suction_zone, view_score2

lock = FileLock("file.lock")
instances_per_sample=1

module_key='joint_quality_networks'
training_buffer = online_data()
training_buffer.main_modality=training_buffer.depth

print=custom_print
max_lr=0.01
min_lr=1*1e-6

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)

def view_scores(pc,scores,threshold=0.5):
    view_scores = scores.detach().cpu().numpy()
    view_scores[view_scores < threshold] *= 0.0
    view_score2(pc, view_scores)

def view_suction_area(pc,dist_mask,target_point,direction,spatial_mask):
    colors = np.zeros_like(pc)
    colors[spatial_mask, 0] += 1.
    colors[dist_mask, 1] += 1.
    view_suction_zone(target_point,direction, pc, colors)

def loop():
    '''load  models'''
    quality_net=ModelWrapper(model=JointQualityNet(),module_key=module_key)
    quality_net.ini_model(train=False)


    grasp_sampler =GANWrapper(module_key=grasp_sampler_key,generator=GraspSampler)
    grasp_sampler.ini_generator(train=False)

    '''dataloader'''
    file_ids=sample_random_buffer(size=None, dict_name=suction_grasp_tracker)
    print(Fore.CYAN, f'Buffer size = {len(file_ids)}')
    dataset = JointQualityDataset(data_pool=training_buffer,file_ids=file_ids)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers, shuffle=True)

    suction_head_statistics = TrainingTracker(name=module_key+'_suction_head', iterations_per_epoch=len(data_loader), samples_size=len(dataset),track_label_balance=True)
    gripper_head_statistics = TrainingTracker(name=module_key+'_gripper_head', iterations_per_epoch=len(data_loader), samples_size=len(dataset),track_label_balance=True)
    shift_head_statistics = TrainingTracker(name=module_key+'_shift_head', iterations_per_epoch=len(data_loader), samples_size=len(dataset),track_label_balance=True)


    for i, batch in enumerate(data_loader, 0):

        depth, normals,scores,pixel_index= batch
        depth = depth.cuda().float()  # [b,1,480.712]
        b = depth.shape[0]

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
                '''view scores'''
                view_scores(pc,gripper_head_predictions,threshold=0.5)
                view_scores(pc,suction_head_predictions,threshold=0.5)
                view_scores(pc,shift_head_predictions,threshold=0.5)

                '''gripper head'''
                gripper_target_index=model_dependent_sampling(pc, gripper_head_predictions, gripper_head_max_score, gripper_head_score_range,spatial_mask)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = gripper_head_predictions[gripper_target_index]
                gripper_target_pose = gripper_poses[gripper_target_index]
                gripper_loss+=gripper_collision_loss(gripper_target_pose, gripper_target_point, pc, gripper_prediction_,gripper_head_statistics,visualize=True)

                '''suction head'''
                suction_target_index=model_dependent_sampling(pc, suction_head_predictions, suction_head_max_score, suction_head_score_range,spatial_mask)
                suction_target_point = pc[suction_target_index]
                suction_prediction_ = suction_head_predictions[suction_target_index]
                suction_loss+=suction_seal_loss(suction_target_point,pc,normals,suction_target_index,suction_prediction_,suction_head_statistics)

                '''shift head'''
                shift_target_index = model_dependent_sampling(pc, shift_head_predictions, shift_head_max_score,shift_head_score_range,probability_exponent=1)
                shift_target_point = pc[shift_target_index]
                shift_prediction_=shift_head_predictions[shift_target_index]
                shift_loss+=shift_affordance_loss(pc,shift_target_point,spatial_mask,shift_head_statistics,shift_prediction_)

        suction_head_statistics.running_loss += suction_loss.item()
        gripper_head_statistics.running_loss += gripper_loss.item()
        shift_head_statistics.running_loss += shift_loss.item()

    suction_head_statistics.print()
    gripper_head_statistics.print()
    shift_head_statistics.print()

if __name__ == "__main__":
    for i in range(10000):
        loop()