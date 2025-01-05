# from models.spatial_encoder import depth_xy_spatial_data
# import numpy as np
# import open3d as o3d
#
# from records.training_satatistics import TrainingTracker
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    z=0.1
    x=torch.tensor([z,0.])
    x_s=F.softmax(x)
    x_l=F.log_softmax(x)
    print(x_s[0]-x_s[1])
    print(x_s[0])
    print()
    print(x_l[0]-x_l[1])
    print(x_l[0])
