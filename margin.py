import numpy as np
import torch
import torchvision
from torch import nn

from lib.cuda_utils import cuda_memory_report

if __name__ == "__main__":

    print(torchvision.__version__)