import asyncio
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import open3d as o3d
import torch
from torch import nn

print('running async test')

def say_boo():
    i = 0
    while True:
        print('...boo {0}'.format(i))
        i += 1


def say_baa():
    i = 0
    while True:
        print('...baa {0}'.format(i))
        i += 1

if __name__ == "__main__":
    x=torch.tensor([0.8])
    y=torch.tensor([1.0])
    l=nn.BCELoss()
    print(l(x,y))
