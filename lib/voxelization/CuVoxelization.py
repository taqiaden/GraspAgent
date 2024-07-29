import os
from typing import Tuple

import torch
import numpy as np
import numba as nb

import cuvoxel

@nb.jit(nopython=True)
def select_voxels(
            continue

def points_to_voxel_new(
    return voxels, coords_origin, num_points_per_voxel_origin

if __name__ == "__main__":
    points = np.load(os.path.join(os.path.dirname(__file__), "points.npy"))

    voxels, coords_origin, num_points_per_voxel_origin = points_to_voxel_new(
        points,
        np.array([0.1, 0.1, 0.1]),
        coors_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    )
    np.save("voxels.npy", voxels)