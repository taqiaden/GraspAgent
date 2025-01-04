# from models.spatial_encoder import depth_xy_spatial_data
# import numpy as np
# import open3d as o3d
#
# from records.training_satatistics import TrainingTracker

if __name__ == "__main__":
    f=lambda x: int(x*20+0.5)/20
    a=0.1
    b=0.14
    z=f((f(a)+b)/2)
    d=f((f(b)+a)/2)
    print(z)
    print(d)
    print(f(0.1))