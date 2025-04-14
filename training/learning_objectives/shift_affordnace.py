import numpy as np
import torch
from torch import nn
from Configurations.ENV_boundaries import bin_center
from training.learning_objectives.suction_seal import transform_point_to_normal_in_plane
from visualiztion import view_shift_pose

shift_effective_length=0.09
shift_execution_length=0.15

shift_elevation_threshold = 0.00
shift_contact_margin = 0.003
collision_radius=0.005
interference_allowance=0.003
bce_loss=nn.BCELoss()
l1_loss=nn.L1Loss()

def view_shift(pc,spatial_mask,shift_mask,start_point, end_point,signed_distance_mask,target_normal):
    colors = np.zeros_like(pc)
    colors[spatial_mask, 0] += 1.
    colors[shift_mask, 1] += 1.
    colors[signed_distance_mask, 2] += 1.

    view_shift_pose(start_point, end_point, pc,target_normal, pc_colors=colors)
    spatial_mask[spatial_mask == 0] = 1

def get_shift_parameteres(shift_target_point):
    '''check affected entities'''
    start_point = shift_target_point
    '''get start and end points'''
    target = np.copy(bin_center)
    target[2] = np.copy(start_point[2])
    direction = target - start_point
    end_point = start_point + ((direction * shift_effective_length) / np.linalg.norm(direction))
    shifted_start_point=start_point + ((direction * shift_contact_margin) / np.linalg.norm(direction))
    return direction,start_point,end_point,shifted_start_point

def get_shift_mask(pc,shifted_start_point,end_point,spatial_mask):
    '''signed distance'''
    direction=end_point-shifted_start_point
    d=direction/np.linalg.norm(direction)

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
    shift_mask = shift_mask & (pc[:, 2] < shifted_start_point[2] + 0.05) # set the maximum highest in the shift path
    return shift_mask

def estimate_shift_score(pc,shift_mask,shift_scores):
    '''get collided points'''
    direct_collision_points = pc[shift_mask]
    if direct_collision_points.shape[0] > 0:
        return torch.tensor(1,device=shift_scores.device).float()
    else:
        return torch.tensor(0,device=shift_scores.device).float()

def estimate_weighted_shift_score(pc,shift_mask,shift_scores,mask,shifted_start_point,j):
    if shift_mask.sum() > 0:
        direct_collision_points = pc[shift_mask]
        '''get score of collided points'''
        collision_score = shift_scores[j, 0][mask][shift_mask]
        collision_score = torch.clip(collision_score, 0, 1.0)

        '''distance weighting'''
        dist_weight = (shift_effective_length - np.linalg.norm(shifted_start_point[np.newaxis] - direct_collision_points,
                                                     axis=-1)) / shift_effective_length
        assert np.all(dist_weight < 1),f'{dist_weight}'

        dist_weight = dist_weight ** 2
        dist_weight=torch.from_numpy(dist_weight).to(collision_score.device)

        '''estimate shift score'''
        shift_label_score =0.5+0.5*( (dist_weight * collision_score).sum() / dist_weight.sum())
        # shift_label_score = shift_label_score * (1 - lambda1) + lambda1
        # print(shift_label_score.item())
        return shift_label_score.float()
    else:
        return torch.tensor(0,device=shift_scores.device).float()

def check_collision_for_shift(normals,target_index,pc):
    target_normal = normals[target_index]
    target_point = pc[target_index]
    shifted_pc = pc - target_point[np.newaxis]
    transformed_pc = transform_point_to_normal_in_plane(target_normal, shifted_pc)
    transformed_target_point = transformed_pc[target_index]
    xy_dist = np.linalg.norm(transformed_target_point[np.newaxis, 0:2] - transformed_pc[:, 0:2], axis=-1)
    signed_distance_mask = (xy_dist < collision_radius) & (
                transformed_pc[:, 2] > transformed_target_point[2] + interference_allowance)
    return signed_distance_mask.sum() > 0, signed_distance_mask,target_normal

def shift_affordance_loss(pc,shift_target_point,spatial_mask,statistics,prediction_,normals,target_index,visualize=False):
    direction, start_point, end_point, shifted_start_point = get_shift_parameteres(shift_target_point)
    # shift_mask = get_shift_mask(pc, shifted_start_point, end_point, spatial_mask)
    shift_result=True
    start_randomization_scope=0.001
    end_randomization_scope=0.03

    '''collision check'''
    # collision,signed_distance_mask,target_normal=check_collision_for_shift(normals,target_index,pc)

    # if collision: shift_result=False
    # else:
    for i in range(5):
        start=shifted_start_point.copy()
        start[0]=start[0]+np.random.randn()*start_randomization_scope
        start[1]=start[1]+np.random.randn()*start_randomization_scope

        end=end_point.copy()
        end[0]=end[0]+np.random.randn()*end_randomization_scope
        end[1]=end[1]+np.random.randn()*end_randomization_scope

        # print('S----',shifted_start_point,start)
        # print('E----',end_point,end)

        shift_mask_result = get_shift_mask(pc, start, end, spatial_mask)
        if shift_mask_result.sum()==0:
            shift_result=False
            break

    if shift_result :
        label= torch.tensor(1, device=prediction_.device).float()
    else:
        label= torch.tensor(0, device=prediction_.device).float()
    statistics.update_confession_matrix(label, prediction_.detach())

    if visualize:
        print(f'Shift label={label}, prediction= {prediction_}')
        # view_shift(pc, spatial_mask, shift_mask, start_point, end_point,signed_distance_mask,target_normal)
    return bce_loss(prediction_, label)
