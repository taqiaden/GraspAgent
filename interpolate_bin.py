import numpy as np

from lib.dataset_utils import online_data
from visualiztion import view_npy_open3d

alpha=np.pi/4
beta=np.pi/6.8
margin=0.005

def estimate_object_mask(pc,custom_margin=None):
    selected_margin = custom_margin if custom_margin is not None else margin
    '''X limits - the small edge'''
    min_x_index = np.argmin(pc[:, 0])
    max_x_index = np.argmax(pc[:, 0])

    min_x_limit = np.tan(beta) * (pc[min_x_index, 2] - pc[:, 2]) + pc[min_x_index, 0]
    max_x_limit = pc[max_x_index, 0] - np.tan(beta) * (pc[max_x_index, 2] - pc[:, 2])

    mask1 = pc[:, 0] > min_x_limit + selected_margin
    mask2 = pc[:, 0] < max_x_limit - selected_margin

    '''Y limits - the long edge'''
    min_y_index = np.argmin(pc[:, 1])
    max_y_index = np.argmax(pc[:, 1])

    min_y_limit = np.tan(alpha) * (pc[min_y_index, 2] - pc[:, 2]) + pc[min_y_index, 1]
    max_y_limit = pc[max_y_index, 1] - np.tan(alpha) * (pc[max_y_index, 2] - pc[:, 2])

    mask3 = pc[:, 1] > min_y_limit + selected_margin
    mask4 = pc[:, 1] < max_y_limit - selected_margin

    '''Z limits'''
    min_z_index = np.argmin(pc[:, 2])
    min_z = pc[min_z_index, 2]

    mask5 = pc[:, 2] > min_z + selected_margin

    return mask1 & mask2 & mask3 & mask4 & mask5

if __name__ == "__main__":
    online_data = online_data()
    while True:
        pc=online_data.load_random_pc()
        total_mask=estimate_object_mask(pc)
        view_npy_open3d(pc[total_mask])
        view_npy_open3d(pc[~total_mask])







