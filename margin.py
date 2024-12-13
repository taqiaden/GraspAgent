import math
from typing import List

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from trimesh.path.creation import circle

from lib.cuda_utils import cuda_memory_report

if __name__ == "__main__":
    counter=0
    moving_average=0
    while True:
        counter+=1
        if np.random.rand()<0.003:
            # print(counter)
            moving_average=0.99*moving_average+0.01*counter
            print(moving_average)
            counter=0


