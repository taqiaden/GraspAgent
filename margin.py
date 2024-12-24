from models.spatial_encoder import depth_xy_spatial_data
import numpy as np

if __name__ == "__main__":
    s=depth_xy_spatial_data(1)
    print(s.shape)
    print(s[0,0,0])
    np.random.random_integers(0,711)