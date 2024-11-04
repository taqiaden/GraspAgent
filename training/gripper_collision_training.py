import numpy as np
import torch
from torch import nn
from check_points.check_point_conventions import ModelWrapper
from dataloaders.gripper_collision_dl import GripperCollisionDataset, load_training_buffer
from lib.IO_utils import custom_print
from lib.collision_unit import grasp_collision_detection
from lib.dataset_utils import training_data
from lib.depth_map import depth_to_point_clouds, pixel_to_point
from lib.loss.D_loss import binary_l1
from lib.models_utils import initialize_model
from lib.report_utils import progress_indicator
from models.Grasp_GAN import gripper_sampler_net, gripper_sampler_path
from models.gripper_collision_net import gripper_collision_net_path, GripperCollisionNet
from pose_object import pose_7_to_transformation
from records.training_satatistics import TrainingTracker
from registration import transform_to_camera_frame, camera
from visualiztion import vis_scene

instances_per_sample=16
gripper_collision_optimizer_path = r'gripper_collision_optimizer'
training_data=training_data()
training_data.main_modality=training_data.depth
print=custom_print
BATCH_SIZE=2
learning_rate=1*1e-6
EPOCHS = 3
weight_decay = 0.000001
workers=2

cos=nn.CosineSimilarity(dim=-1,eps=1e-6)

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

                selection_probability = (1 - collision_scores[j, 0, pix_A, pix_B])

                if mask[pix_A, pix_B] == 1 and np.random.rand() > selection_probability: break
            depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()

            '''get pose parameters'''
            target_point = pixel_to_point(np.array([pix_A, pix_B]), depth_value, camera)
            target_point = transform_to_camera_frame(target_point[None, :], reverse=True)[0]
            target_pose = generated_grasps[j, :, pix_A, pix_B]
            T_d, width, distance = pose_7_to_transformation(target_pose, target_point)

            '''check collision'''
            # if collision_scores[j,0,pix_A,pix_B]>0.5:
            #     vis_scene(T_d, width, npy=pc)
            collision_intensity = grasp_collision_detection(T_d, width, pc, visualize=False)
            # print(f'collision result= {collision_intensity}')

            '''target prediction and label score'''
            prediction_ = collision_scores[j, 0, pix_A, pix_B]
            label = torch.zeros_like(prediction_) if collision_intensity > 0 else torch.ones_like(prediction_)

            '''update confession matrix'''
            statistics.update_confession_matrix(label, prediction_)

            '''instance loss'''
            loss_ = binary_l1(prediction_, label) ** 2
            if loss_ == 0.0:
                statistics.labels_with_zero_loss += 1
            loss += loss_
    return loss

def train_():
    '''dataloader'''
    dataset = GripperCollisionDataset(data_pool=training_data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers, shuffle=True)

    '''model'''
    gripper_collision=ModelWrapper(model=GripperCollisionNet,model_name=gripper_collision_net_path,optimizer_name=gripper_collision_optimizer_path)
    gripper_collision.ini_model()
    generator = initialize_model(gripper_sampler_net, gripper_sampler_path).eval()

    '''optimizer'''
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

if __name__ == "__main__":
    while True:
        training_data.clear()
        if len(training_data) == 0:
            load_training_buffer(size=100)
        train_()
        training_data.clear()


