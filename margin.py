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
    x=[1,None,[1,2]]
    x=np.array(x)
    print(x[0])
    print(x[1])
    print(x[2])

