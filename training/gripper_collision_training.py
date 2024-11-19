import numpy as np
import torch
from colorama import Fore
from torch import nn
from Online_data_audit.data_tracker import sample_positive_buffer, gripper_grasp_tracker
from check_points.check_point_conventions import ModelWrapper
from dataloaders.gripper_collision_dl import GripperCollisionDataset
from lib.IO_utils import custom_print
from lib.collision_unit import grasp_collision_detection
from lib.dataset_utils import online_data
from lib.depth_map import depth_to_point_clouds, pixel_to_point
from lib.loss.D_loss import binary_l1
from lib.models_utils import initialize_model
from lib.optimizer import exponential_decay_lr_
from lib.report_utils import progress_indicator
from models.Grasp_GAN import gripper_sampler_net, gripper_sampler_path
from models.gripper_collision_net import gripper_collision_net_path, GripperCollisionNet
from pose_object import pose_7_to_transformation
from records.training_satatistics import TrainingTracker
from registration import transform_to_camera_frame, camera
from visualiztion import vis_scene

instances_per_sample=16
module_key = 'gripper_collision'
training_buffer = online_data()

training_buffer.main_modality=training_buffer.depth
print=custom_print
BATCH_SIZE=1
learning_rate=1*1e-6
EPOCHS = 1
weight_decay = 0.000001
workers=2

max_lr=0.1
min_lr=1*1e-6

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)

def collision_loss(pc,target_pose, target_point,prediction_):
    T_d, width, distance = pose_7_to_transformation(target_pose, target_point)

    '''check collision'''
    collision_intensity = grasp_collision_detection(T_d, width, pc, visualize=False)

    '''target prediction and label score'''
    label = torch.zeros_like(prediction_) if collision_intensity > 0 else torch.ones_like(prediction_)

    '''update confession matrix'''
    statistics.update_confession_matrix(label, prediction_)

    '''instance loss'''
    loss_ = binary_l1(prediction_, label) ** 2
    return loss_

def cumulative_collision_loss(depth,collision_scores,generated_grasps,statistics):
    loss = 0
    for j in range(BATCH_SIZE):
        # print(j)
        '''get point clouds'''
        pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
        pc = transform_to_camera_frame(pc, reverse=True)
        for k in range(instances_per_sample):

            '''pick random pixels'''
            while True:
                pix_A = np.random.randint(40, 400)
                pix_B = np.random.randint(40, 632)

                selection_probability = (1 - collision_scores[j, 0, pix_A, pix_B]).item()
                selection_probability=min(selection_probability,0.95)

                if mask[pix_A, pix_B] == 1 and np.random.rand() > selection_probability: break
            depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()

            '''get pose parameters'''
            target_point = pixel_to_point(np.array([pix_A, pix_B]), depth_value, camera)
            target_point = transform_to_camera_frame(target_point[None, :], reverse=True)[0]
            target_pose = generated_grasps[j, :, pix_A, pix_B]

            '''target prediction and label score'''
            prediction_ = collision_scores[j, 0, pix_A, pix_B]


            '''instance loss'''
            loss_ = collision_loss(pc,target_pose, target_point,prediction_)

            if loss_ == 0.0:
                statistics.labels_with_zero_loss += 1
            loss += loss_
    return loss

def train_(file_ids,learning_rate):
    '''dataloader'''
    dataset = GripperCollisionDataset(data_pool=training_buffer,file_ids=file_ids)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers, shuffle=True)

    '''model'''
    gripper_collision=ModelWrapper(model=GripperCollisionNet,module_key=module_key)
    gripper_collision.ini_model()
    generator = initialize_model(gripper_sampler_net, gripper_sampler_path).eval()

    '''optimizer'''
    print(Fore.CYAN,f'Learning rate = {learning_rate}',Fore.RESET)
    gripper_collision.ini_sgd_optimizer(learning_rate=learning_rate)

    for epoch in range(EPOCHS):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(data_loader))
        statistics=TrainingTracker(name='',iterations_per_epoch=len(data_loader),samples_size=len(dataset))

        for i, batch in enumerate(data_loader, 0):
            depth= batch
            depth=depth.cuda().float()

            '''generate grasps'''
            with torch.no_grad():
                generated_grasps = generator(depth.clone())

            '''get predictions'''
            gripper_collision.model.zero_grad()
            collision_scores=gripper_collision.model(depth.clone(),generated_grasps.clone())

            '''compute loss'''
            loss=cumulative_collision_loss(depth,collision_scores,generated_grasps,statistics)/(BATCH_SIZE*instances_per_sample)

            '''optimize'''
            loss.backward()
            gripper_collision.optimizer.step()

            statistics.running_loss += loss.item()
            pi.step(i)

        pi.end()
        statistics.print()

        '''export check points'''
        gripper_collision.export_model()
        gripper_collision.export_optimizer()

        return statistics



if __name__ == "__main__":
    while True:
        # training_data.clear()
        file_ids = sample_positive_buffer(size=100, dict_name=gripper_grasp_tracker)
        statistics=train_(file_ids,learning_rate)

        '''update learning rate'''
        performance_indicator=statistics.confession_matrix.TP/statistics.confession_matrix.total_classification()
        learning_rate=exponential_decay_lr_(performance_indicator, max_lr, min_lr)
        # training_data.clear()


