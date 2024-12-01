import numpy as np
import torch
from torch import nn

if __name__ == "__main__":
    b=torch.tensor([1,2,3,4,5,6,7]).cuda()
    print(b[-2:])

