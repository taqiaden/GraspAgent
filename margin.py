import math
from typing import List

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from pyparsing import White
from torch import nn
from trimesh.path.creation import circle

from lib.cuda_utils import cuda_memory_report

if __name__ == "__main__":
    class TrainingTracker:
        def __init__(self):
            self.x_ = 0.

            self.mx = 0
            self.decay_rate = 0.1
            self.counter=0

        @property
        def x(self):
            return self.x_

        @x.setter
        def x(self, new_value):
            self.counter+=1
            self.x_ = new_value
            adaptive_decay=(self.counter/(self.counter+100))**2
            adaptive_decay=max(0.0001,min(adaptive_decay,0.1))
            self.mx = adaptive_decay * new_value/self.counter + self.mx * (1 - adaptive_decay)

    t=TrainingTracker()
    for i in range(5000):
        t.x+=0.01
        if i%100==0:
            print((t.counter/(t.counter+10000))**2)
            # print(t.mx)
            # print(t.x_)


