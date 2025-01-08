# from models.spatial_encoder import depth_xy_spatial_data
# import numpy as np
# import open3d as o3d
# from records.training_satatistics import TrainingTracker
import numpy as np
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    x=torch.randn(2,2,3)
    print(F.softmax(x[0].reshape(-1)))
    y=x.view(2,-1)
    y=F.softmax(y)
    # print(F.softmax(x,dim=-1))
    x=y.reshape(x.shape)
    print(x[0])
