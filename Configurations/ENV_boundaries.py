import numpy as np
import torch
# x_min_dis = 0.27  # limit to prevent collision with the tray edge
x_limits=[0.28,0.582]
y_limits=[-0.21,0.21]
z_limits=[0.048,0.20]

knee_ref_elevation = 0.25

floor_elevation=0.045

ref_pc_center=torch.tensor([0.4364,-0.0091,0.0767],device='cuda')
ref_pc_center_6=torch.tensor([0.4364,-0.0091,0.0767,0,0,0],device='cuda')

depth_lower_bound=1100.
depth_mean=1180.
depth_std=85.

dist_allowance=0.0035

def median_(pair):
    return pair[0]+(pair[1]-pair[0])/2

bin_center=np.array([median_(x_limits),median_(y_limits),median_(z_limits)])