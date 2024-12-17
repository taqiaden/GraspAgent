import numpy as np
import torch
from colorama import Fore
from torch import nn
from Online_data_audit.data_tracker import sample_positive_buffer, gripper_grasp_tracker
from check_points.check_point_conventions import ModelWrapper
from dataloaders.shift_net_dl import ShiftNetDataset
from interpolate_bin import estimate_object_mask
from lib.IO_utils import custom_print
from lib.dataset_utils import online_data
from lib.depth_map import depth_to_point_clouds
from lib.loss.D_loss import binary_l1
from lib.optimizer import exponential_decay_lr_
from lib.report_utils import progress_indicator
from models.shift_net import ShiftNet
from records.training_satatistics import TrainingTracker, MovingMetrics
from registration import transform_to_camera_frame, camera
from training.joint_quality_lr import model_dependent_sampling
from training.learning_objectives.shift_affordnace import get_shift_parameteres, get_shift_mask, estimate_shift_score
from visualiztion import vis_scene, view_npy_open3d, view_shift_pose, view_score, view_score2

instances_per_sample=10
module_key = r'shift_net'
training_buffer = online_data()

training_buffer.main_modality=training_buffer.depth
print=custom_print
BATCH_SIZE=1
learning_rate=1*1e-5
EPOCHS = 1
weight_decay = 0.000001
workers=2

max_lr=0.01
min_lr=5*1e-5

mes_loss=nn.MSELoss()
l1_loss=nn.L1Loss()

sig=nn.Sigmoid()

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



def cumulative_shift_loss(depth,shift_scores,statistics,moving_rates):
    # shift_scores=sig(shift_scores)
    loss = 0
    for j in range(BATCH_SIZE):
        '''get point clouds'''
        pc, mask = depth_to_point_clouds(depth[j, 0].cpu().numpy(), camera)
        pc = transform_to_camera_frame(pc, reverse=True)

        spatial_mask = estimate_object_mask(pc)

        '''view scores'''
        view_scores(pc, shift_scores[j, 0][mask])

        for k in range(instances_per_sample):
            shift_head_predictions = shift_scores[j, 0][mask]
            shift_head_max_score = torch.max(shift_scores).item()
            shift_head_score_range = (shift_head_max_score - torch.min(shift_scores)).item()
            shift_target_index = model_dependent_sampling(pc, shift_head_predictions, shift_head_max_score,
                                                            shift_head_score_range)
            shift_target_point = pc[shift_target_index]
            direction,start_point,end_point,shifted_start_point=get_shift_parameteres(shift_target_point)
            shift_mask=get_shift_mask(pc, direction, shifted_start_point, end_point,spatial_mask)

            lambda1 = max(moving_rates.tnr - moving_rates.tpr, 0)**0.5
            label=estimate_shift_score(pc, shift_mask, shift_scores, mask, shifted_start_point, j,lambda1)

            '''target prediction and label score'''
            prediction_ = shift_head_predictions[shift_target_index]
            # label = torch.ones_like(prediction_) if np.any(shift_mask==True)  else torch.zeros_like(prediction_)

            '''view shift action'''
            # if k==0:
            #     # if (prediction_>0.5 and label<0.) or(prediction_<0.5 and label>0.5):
            #     print(prediction_.item())
            #     print(label.item())
            #     view_shift(pc,spatial_mask,shift_mask,start_point, end_point)
            '''update confession matrix'''
            statistics.update_confession_matrix(label, prediction_)

            '''instance loss'''
            loss_ = l1_loss(prediction_, label)
            # if label<=0: loss_=loss_+mes_loss(prediction_, label)

            # if (label.item()<=0.0) and (prediction_<=0.): loss_*=0

            # if label>0.0: loss_*=(1+lambda1)
            decayed_loss=binary_l1(shift_scores[j, 0][mask],torch.zeros_like(shift_scores[j, 0][mask])).mean()
            moving_rates.update(label, prediction_)
            # moving_rates.view()
            if loss_ == 0.0:
                statistics.labels_with_zero_loss += 1
            loss += loss_+decayed_loss
    return loss

def train_(file_ids,adaptive_learning_rate):
    moving_rates = MovingMetrics(module_key,decay_rate=0.001)

    '''dataloader'''
    dataset = ShiftNetDataset(data_pool=training_buffer,file_ids=file_ids)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=workers, shuffle=True)

    '''model'''
    model_wrapper=ModelWrapper(model=ShiftNet(),module_key=module_key)
    model_wrapper.ini_model()

    '''optimizer'''
    print(Fore.CYAN,f'Learning rate = {learning_rate}',Fore.RESET)
    model_wrapper.ini_adam_optimizer(learning_rate=learning_rate)
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
