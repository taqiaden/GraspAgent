import numpy as np
import torch
from torch import nn

from Configurations.ENV_boundaries import bin_center


shift_length=0.1
shift_elevation_threshold = 0.001
shift_contact_margin = 0.01
bce_loss=nn.BCELoss()

def get_shift_parameteres(shift_target_point):
    '''check affected entities'''
    start_point = shift_target_point
    '''get start and end points'''
    target = np.copy(bin_center)
    target[2] = np.copy(start_point[2])
    direction = target - start_point
    end_point = start_point + ((direction * shift_length) / np.linalg.norm(direction))
    shifted_start_point=start_point + ((direction * shift_contact_margin) / np.linalg.norm(direction))
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
        dist_weight = (shift_length - np.linalg.norm(shifted_start_point[np.newaxis] - direct_collision_points,
                                                     axis=-1)) / shift_length
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

def shift_affordance_loss(pc,shift_target_point,spatial_mask,statistics,prediction_):
    direction, start_point, end_point, shifted_start_point = get_shift_parameteres(shift_target_point)
    shift_mask = get_shift_mask(pc, direction, shifted_start_point, end_point, spatial_mask)
    if shift_mask.sum() > 0:
        label= torch.tensor(1, device=prediction_.device).float()
    else:
        label= torch.tensor(0, device=prediction_.device).float()
    statistics.update_confession_matrix(label, prediction_)
    return bce_loss(prediction_, label)
