import torch
from colorama import Fore
from torch.utils import data
from Configurations.config import workers
from Online_data_audit.data_tracker import sample_positive_buffer,  gripper_grasp_tracker
from check_points.check_point_conventions import GANWrapper
from dataloaders.action_dl import ActionDataset
from interpolate_bin import estimate_object_mask
from lib.IO_utils import   custom_print
from lib.dataset_utils import online_data
from lib.depth_map import transform_to_camera_frame, depth_to_point_clouds
from models.action_net import ActionNet, Critic
from records.training_satatistics import TrainingTracker
from registration import camera
from lib.report_utils import  progress_indicator
from filelock import FileLock
from training.action_lr import model_dependent_sampling
from training.learning_objectives.gripper_collision import gripper_collision_loss, evaluate_grasps
from training.learning_objectives.shift_affordnace import  shift_affordance_loss
from training.learning_objectives.suction_seal import suction_seal_loss
from visualiztion import view_score2, view_npy_open3d

lock = FileLock("file.lock")
instances_per_sample=1

module_key='action_net'
training_buffer = online_data()
training_buffer.main_modality=training_buffer.depth

print=custom_print

def view_scores(pc,scores,threshold=0.5):
    view_scores = scores.detach().cpu().numpy()
    view_scores[view_scores < threshold] *= 0.0
    view_score2(pc, view_scores)

def loop():
    '''load  models'''
    gan=GANWrapper(module_key,ActionNet,Critic)
    gan.ini_models(train=False)

    '''dataloader'''
    file_ids=sample_positive_buffer(size=None, dict_name=gripper_grasp_tracker)
    print(Fore.CYAN, f'Buffer size = {len(file_ids)}')
    dataset = ActionDataset(data_pool=training_buffer,file_ids=file_ids)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=workers, shuffle=True)

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
            gripper_pose, suction_direction, griper_collision_classifier, suction_quality_classifier, shift_affordance_classifier = gan.generator(
                depth.clone())

        '''Evaluate generated grasps'''
        collision_state_list, firmness_state_list, out_of_scope_list = evaluate_grasps(b, pixel_index, depth,
                                                                                       gripper_pose, pose_7,visualize=True)
        collision_times += sum(collision_state_list)
        out_of_scope_times += sum(out_of_scope_list)
        good_firmness_times += sum(firmness_state_list)

        for j in range(b):
            '''get parameters'''
            pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
            pc = transform_to_camera_frame(pc, reverse=True)
            normals = suction_direction[j].permute(1,2,0)[mask].detach().cpu().numpy()
            gripper_poses=gripper_pose[j].permute(1,2,0)[mask]#.cpu().numpy()
            spatial_mask = estimate_object_mask(pc,custom_margin=0.01)
            suction_head_predictions=suction_quality_classifier[j, 0][mask]
            gripper_head_predictions=griper_collision_classifier[j, 0][mask]
            shift_head_predictions = shift_affordance_classifier[j, 0][mask]

            view_npy_open3d(pc,normals=normals)
            view_scores(pc, gripper_head_predictions, threshold=0.5)
            view_scores(pc, suction_head_predictions, threshold=0.5)
            view_scores(pc, shift_head_predictions, threshold=0.5)

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
                gripper_target_index=model_dependent_sampling(pc, gripper_head_predictions, gripper_head_max_score, gripper_head_score_range,spatial_mask,probability_exponent=10,balance_indicator=-1)
                gripper_target_point = pc[gripper_target_index]
                gripper_prediction_ = gripper_head_predictions[gripper_target_index]
                gripper_target_pose = gripper_poses[gripper_target_index]
                gripper_collision_loss(gripper_target_pose, gripper_target_point, pc, gripper_prediction_,gripper_head_statistics)

                '''suction head'''
                suction_target_index=model_dependent_sampling(pc, suction_head_predictions, suction_head_max_score, suction_head_score_range,spatial_mask,probability_exponent=10,balance_indicator=-1)
                suction_prediction_ = suction_head_predictions[suction_target_index]
                suction_seal_loss(pc,normals,suction_target_index,suction_prediction_,suction_head_statistics,spatial_mask,visualize=True)

                '''shift head'''
                shift_target_index = model_dependent_sampling(pc, shift_head_predictions, shift_head_max_score,shift_head_score_range,probability_exponent=10,balance_indicator=-1)
                shift_target_point = pc[shift_target_index]
                shift_prediction_=shift_head_predictions[shift_target_index]
                shift_affordance_loss(pc,shift_target_point,spatial_mask,shift_head_statistics,shift_prediction_,normals,shift_target_index,visualize=True)

        pi.step(i)
    pi.end()

    suction_head_statistics.print()
    gripper_head_statistics.print()
    shift_head_statistics.print()
    gripper_sampler_statistics.print()
    suction_sampler_statistics.print()
    size = len(file_ids)
    print(f'Collision ratio = {collision_times / size}')
    print(f'out of scope ratio = {out_of_scope_times / size}')
    print(f'firm grasp ratio = {good_firmness_times / size}')

if __name__ == "__main__":
    for i in range(10000):
        loop()