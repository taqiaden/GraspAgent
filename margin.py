from models.spatial_encoder import depth_xy_spatial_data
import numpy as np
import open3d as o3d

from records.training_satatistics import TrainingTracker

if __name__ == "__main__":
    x = TrainingTracker(name='test', iterations_per_epoch=1000,track_prediction_balance=True)

    for i in range(10000):
        x.loss=1+np.random.randn()
        if i%100==0:
            x.print(i+1)

