from Configurations import ENV_boundaries
from lib.report_utils import save_error_log

def get_spatial_mask(pc):
    x = pc[:, :, 0:1]
    y = pc[:, :, 1:2]
    z = pc[:, :, 2:3]
    x_mask = (x > ENV_boundaries.x_limits[0]) & (x < ENV_boundaries.x_limits[1])
    y_mask = (y > ENV_boundaries.y_limits[0]) & (y <ENV_boundaries.y_limits[1])
    z_mask = (z > ENV_boundaries.z_limits[0]) & (z < ENV_boundaries.z_limits[1])
    spatial_mask = x_mask & y_mask & z_mask
    return spatial_mask

def accumulate_mask_layer(array,current_mask,maximum_size,mask_inversion=False):
    global score_threshold
    m1=(array>score_threshold) & (current_mask)
    z=array[m1]
    if z.shape[0]<0 or maximum_size is None:
        save_error_log('Mask has no True element, function: accumulate_mask_layer')
        return m1
    if z.shape[0]<=maximum_size: return m1
    else:
        limit_value=sorted(z,reverse= not mask_inversion)[maximum_size]
        value_mask=array<limit_value if  mask_inversion else array>limit_value
        total_mask=(array>score_threshold) & (value_mask) & (current_mask)
        return total_mask

def initialize_masks(grasp_score_pred, suction_score_pred, data_ ,grasp_max_size,suction_max_size,mask_inversion=False):
        # filter bin points
        x = data_[:, 0]
        y = data_[:, 1]
        z = data_[:, 2]
        x_mask = (x > 0.275+0.01) & (x < 0.585-0.01)
        y_mask = (y > -0.20) & (y < 0.20)
        z_mask = (z > ENV_boundaries.z_limits[0]) & (z < ENV_boundaries.z_limits[1])
        mask = x_mask & y_mask & z_mask
        # view_npy_open3d(data_)
        # view_npy_open3d(data_[mask])
        # view_npy_open3d(data_[~mask])
        # exit()
        grasp_score_pred_mask = accumulate_mask_layer(grasp_score_pred, mask, grasp_max_size,mask_inversion)

        suction_score_pred_mask = accumulate_mask_layer(suction_score_pred,mask, suction_max_size,mask_inversion)


        return grasp_score_pred_mask, suction_score_pred_mask
