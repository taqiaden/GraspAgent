import numpy as np
import torch
from colorama import Fore
from torch import nn

from Configurations.ENV_boundaries import bin_center
from Configurations.config import shift_length, shift_elevation_threshold, shift_contact_margin
from Online_data_audit.data_tracker import sample_positive_buffer, gripper_grasp_tracker
from check_points.check_point_conventions import ModelWrapper
from dataloaders.shift_net_dl import ShiftNetDataset
from interpolate_bin import estimate_object_mask
from lib.IO_utils import custom_print
from lib.dataset_utils import online_data
from lib.depth_map import depth_to_point_clouds, pixel_to_point
from lib.loss.D_loss import binary_l1
from lib.optimizer import exponential_decay_lr_
from lib.report_utils import progress_indicator
from models.shift_net import ShiftNet
from records.training_satatistics import TrainingTracker, MovingMetrics
from registration import transform_to_camera_frame, camera
from visualiztion import vis_scene, view_npy_open3d, view_shift_pose, view_score, view_score2

instances_per_sample=1
module_key = r'shift_net'
training_buffer = online_data()

training_buffer.main_modality=training_buffer.depth
print=custom_print
BATCH_SIZE=1
learning_rate=5*1e-4
EPOCHS = 1
weight_decay = 0.000001
workers=2

max_lr=0.01
min_lr=5*1e-5

mes_loss=nn.MSELoss()


cos=nn.CosineSimilarity(dim=-1,eps=1e-6)

def view_shift(pc,spatial_mask,shift_mask,start_point, end_point):
    colors = np.zeros_like(pc)
    colors[spatial_mask, 0] += 1.
    colors[shift_mask, 1] += 1.
    view_shift_pose(start_point, end_point, pc, pc_colors=colors)
    spatial_mask[spatial_mask == 0] = 1

def view_scores(pc,scores):
    view_scores = scores.detach().cpu().numpy()
    view_scores[view_scores < 0.5] *= 0.0
    view_score2(pc, view_scores)

def get_shift_parameteres(depth,pix_A, pix_B,j):
    '''check affected entities'''
    depth_value = depth[j, 0, pix_A, pix_B].cpu().numpy()
    start_point = pixel_to_point(np.array([pix_A, pix_B]), depth_value, camera)
    start_point = transform_to_camera_frame(start_point[None, :], reverse=True)[0]

    '''get start and end points'''
    target = np.copy(bin_center)
    target[2] = start_point[2]
    end_point = start_point + ((target - start_point) * shift_length) / np.linalg.norm(target - start_point)
    shifted_start_point=start_point + ((target - start_point) * shift_contact_margin) / np.linalg.norm(target - start_point)

    direction = end_point - start_point

    return direction,start_point,end_point,shifted_start_point

def get_shift_mask(pc,direction,shifted_start_point,end_point,spatial_mask):
    '''signed distance'''
    d = direction / np.linalg.norm(direction)
    s = np.dot(shifted_start_point - pc, d)
    t = np.dot(pc - end_point, d)

    '''distance to the line segment'''
    flatten_pc = np.copy(pc)
    flatten_pc[:, 2] = shifted_start_point[2]
    distances = np.cross(end_point - shifted_start_point, shifted_start_point - flatten_pc) / np.linalg.norm(
        end_point - shifted_start_point)
    distances = np.linalg.norm(distances, axis=1)

    '''isolate the bin'''
    shift_mask = (s < 0.0) & (t < 0.0) & (distances < shift_contact_margin) & (pc[:, 2] > shifted_start_point[2] + shift_elevation_threshold) & spatial_mask
    return shift_mask

def estimate_shift_score(pc,shift_mask,shift_scores,mask,shifted_start_point,j,lambda1):
    '''get collided points'''
    direct_collision_points = pc[shift_mask]
    if direct_collision_points.shape[0] > 0:
        '''get score of collided points'''
        collision_score = shift_scores[j, 0][mask][shift_mask]
        collision_score = torch.clip(collision_score, 0, 1.0)

        '''distance weighting'''
        dist_weight = (shift_length - np.linalg.norm(shifted_start_point[np.newaxis] - direct_collision_points,
                                                     axis=-1)) / shift_length
        assert np.all(dist_weight < 1),f'{dist_weight}'

        dist_weight = dist_weight ** 2
        dist_weight=torch.from_numpy(dist_weight).to(collision_score.device)

        '''estimate shift score'''
        shift_label_score = (dist_weight * collision_score).sum() / dist_weight.sum()
        shift_label_score = shift_label_score * (1 - lambda1) + lambda1
        # print(shift_label_score.item())
        return shift_label_score.float()
    else:
        return torch.tensor(0,device=shift_scores.device).float()

def cumulative_shift_loss(depth,shift_scores,statistics,moving_rates):
    loss = 0
    for j in range(BATCH_SIZE):
        '''get point clouds'''
        pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
        pc = transform_to_camera_frame(pc, reverse=True)

        spatial_mask = estimate_object_mask(pc)

        '''view scores'''
        # view_scores(pc, shift_scores[j, 0][mask])

        for k in range(instances_per_sample):

            '''pick random pixels'''
            while True:
                pix_A = np.random.randint(0, 480)
                pix_B = np.random.randint(0, 712)

                # selection_probability = abs(0.5 - shift_scores[j, 0, pix_A, pix_B]).item()
                # selection_probability=min(selection_probability,0.75)

                if mask[pix_A, pix_B] == 1: break

            direction,start_point,end_point,shifted_start_point=get_shift_parameteres(depth, pix_A, pix_B, j)
            shift_mask=get_shift_mask(pc, direction, shifted_start_point, end_point,spatial_mask)

            lambda1 = max(moving_rates.tnr - moving_rates.tpr, 0)**0.5
            label=estimate_shift_score(pc, shift_mask, shift_scores, mask, shifted_start_point, j,lambda1)

            '''view shift action'''
            # view_shift(pc,spatial_mask,shift_mask,start_point, end_point)

            '''target prediction and label score'''
            prediction_ = shift_scores[j, 0, pix_A, pix_B]
            # label = torch.ones_like(prediction_) if np.any(shift_mask==True)  else torch.zeros_like(prediction_)

            '''update confession matrix'''
            statistics.update_confession_matrix(label, prediction_)

            '''instance loss'''

            loss_ = mes_loss(prediction_, label)
            if (label.item()<=0.0) and (prediction_<=0.): loss_*=0

            if label>0.0: loss_*=(1+lambda1)
            decayed_loss=binary_l1(shift_scores[j, 0][mask],torch.zeros_like(shift_scores[j, 0][mask])).mean()
            moving_rates.update(label, prediction_)
            # moving_rates.view()
            if loss_ == 0.0:
                statistics.labels_with_zero_loss += 1
            loss += loss_+decayed_loss*0.1
    return loss

def train_(file_ids,learning_rate):
    moving_rates = MovingMetrics(module_key,decay_rate=0.001)

    '''dataloader'''
    dataset = ShiftNetDataset(data_pool=training_buffer,file_ids=file_ids)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers, shuffle=True)

    '''model'''
    model_wrapper=ModelWrapper(model=ShiftNet,module_key=module_key)
    model_wrapper.ini_model()

    '''optimizer'''
    print(Fore.CYAN,f'Learning rate = {learning_rate}',Fore.RESET)
    model_wrapper.ini_sgd_optimizer(learning_rate=learning_rate)
    statistics = TrainingTracker(name='', iterations_per_epoch=len(data_loader), samples_size=len(dataset))

    for epoch in range(EPOCHS):
        pi = progress_indicator('EPOCH {}: '.format(epoch + 1), max_limit=len(data_loader))

        for i, batch in enumerate(data_loader, 0):
            depth= batch
            depth=depth.cuda().float()

            '''get predictions'''
            model_wrapper.model.zero_grad()
            shift_scores=model_wrapper.model(depth.clone())

            '''compute loss'''
            loss=cumulative_shift_loss(depth,shift_scores,statistics,moving_rates)/(BATCH_SIZE*instances_per_sample)

            '''optimize'''
            loss.backward()
            model_wrapper.optimizer.step()

            statistics.running_loss += loss.item()
            pi.step(i)

        pi.end()
        statistics.print()

        '''export check points'''
        model_wrapper.export_model()
        model_wrapper.export_optimizer()

    moving_rates.save()
    moving_rates.view()
    return statistics

if __name__ == "__main__":
    while True:
        file_ids = sample_positive_buffer(size=100, dict_name=gripper_grasp_tracker)
        statistics=train_(file_ids,learning_rate)

        '''update learning rate'''
        performance_indicator=statistics.confession_matrix.accuracy()
        learning_rate=exponential_decay_lr_(performance_indicator, max_lr, min_lr)
