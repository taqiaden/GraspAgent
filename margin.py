# from models.spatial_encoder import depth_xy_spatial_data
# import numpy as np
# import open3d as o3d
#
# from records.training_satatistics import TrainingTracker
import numpy as np
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    x=0
    a=0.001
    for i in range(1000):
        a=-np.random.rand()
        b=np.random.rand()
        c=np.random.rand()
        x1=min(a*b,a*c)
        x2=a*min(b,c)
        print(x1==x2)


        x=x*(1-a)+a
    print(x)
