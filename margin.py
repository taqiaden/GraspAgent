from models.spatial_encoder import depth_xy_spatial_data
import numpy as np
import open3d as o3d

from records.training_satatistics import TrainingTracker

if __name__ == "__main__":
    for i in range(100):
        print(i%2)
