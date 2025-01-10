# from models.spatial_encoder import depth_xy_spatial_data
# import numpy as np
# import open3d as o3d
# from records.training_satatistics import TrainingTracker
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions.categorical import Categorical
from torchrl.modules import MaskedCategorical

from records.training_satatistics import MovingRate


def get_index(flat_index, ori_size):
    res = torch.zeros(len(ori_size), dtype=torch.int)
    for i in range(len(ori_size)-1, -1, -1):
        j = flat_index % ori_size[i]
        flat_index = flat_index // ori_size[i]
        res[i] = j
    return res

def soft_square_pulse(x,start=-1.,end=1.,a=0.5*np.pi,b=-0.5*np.pi,epsilon=0.0,no_grad=True):
    def f_(x):
        mean = (end - start) / 2

        shift = start + mean
        shifted_x = x / mean - shift
        mask = torch.abs(shifted_x) < (2 / a)
        # print(x)
        # print(torch.abs(shifted_x) )
        # print(2/a)
        # print(mask)
        # exit()
        x = a * shifted_x - b
        sin_ = torch.sin(x)
        return 0.5 * ((sin_ / torch.sqrt(sin_ ** 2 + epsilon)) + 1) * mask
    if no_grad:
        with torch.no_grad():
            return f_(x)
    else:
        return f_(x)

def soft_clipping(value,min_,max_,a=0.5*np.pi,b=-0.5*np.pi,epsilon=0.01):
    mean=(max_-min_)/2

    shift=min_+mean
    shifted_value=value/mean-shift

    if np.abs(shifted_value)<2/a:
        x = a * shifted_value - b
        sin_ = np.sin(x)
        return 0.5  * ((sin_/np.sqrt(sin_**2+epsilon))+1) * value
    else:
        return 0.

if __name__ == "__main__":
    x=[]
    y=[]
    t=torch.tensor([-1.0])
    while True:
        x.append(t.clone())
        y.append(soft_square_pulse(t,start=0.8,end=1.2))
        t+=0.1
        if t>2:break



    plt.plot(x,y)
    plt.show()

    # print(soft_square_pulse(1.1))
    # print(soft_clipping(1.0,0.8,1.2))
    # x=torch.randn(100,4)
    # reshaped_x=x.reshape(-1)
    #
    # # x=F.softmax(x,dim=-1)
    #
    # # dist=MaskedCategorical(probs=reshaped_x, mask=x>0.)
    # m=MovingRate('test')
    # while True:
    #     m.update(0.5)
    #     m.view()

        # index = dist.sample()
        # orig_index=
        #
        # print(x[index])
    # ori_index=get_index(view_index,x.shape)

    # print(x[ori_index[0],ori_index[1],ori_index[2],ori_index[3]])
    # print(z[0,view_index])
    #
    # print(ori_index)
    # print(view_index)
    # x_index=

    # probs = torch.squeeze(dist.log_prob(action)).item()

