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
    x=torch.randn((7,5))
    x[x<0]*=0
    i=torch.nonzero(x)
    print(i.shape)
    while True:
        pick=np.random.random_integers(0,i.shape[0]-1)
        print(pick)

        print(x[i[pick][0],i[pick][1]]>0)
        assert (x[i[pick][0],i[pick][1]]>0)==True
        # point_index = math.floor(j / 4)
        # action_index = j - int(4 * point_index)
        # print(x[point_index,action_index]>0)
    exit()
    class TrainingTracker:
        def __init__(self):
            self.x_ = 0.

            self.mx = 0
            self.decay_rate = 0.001

        @property
        def x(self):
            return self.x_

        @x.setter
        def x(self, new_value):
            instance_value=new_value-self.x_
            self.x_ = new_value
            self.mx = self.decay_rate * instance_value + self.mx * (1 - self.decay_rate)

    t=TrainingTracker()
    for i in range(5000):
        t.x+=0.01
        if i%100==0:
            # print((t.counter/(t.counter+10000))**2)
            print(t.mx)
            # print(t.x_)


