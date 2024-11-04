import torch
# x_min_dis = 0.27  # limit to prevent collision with the tray edge
x_limits=[0.28,0.582]
y_limits=[-0.21,0.21]
z_limits=[0.048,0.20]

floor_elevation=0.05

ref_pc_center=torch.tensor([0.4364,-0.0091,0.0767]).to('cuda')
ref_pc_center_6=torch.tensor([0.4364,-0.0091,0.0767,0,0,0]).to('cuda')

depth_lower_bound=1100.
depth_mean=1180.
depth_std=85.

dist_allowance=0.005