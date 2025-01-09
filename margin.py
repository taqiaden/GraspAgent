# from models.spatial_encoder import depth_xy_spatial_data
# import numpy as np
# import open3d as o3d
# from records.training_satatistics import TrainingTracker
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torchrl.modules import MaskedCategorical


def get_index(flat_index, ori_size):
    res = torch.zeros(len(ori_size), dtype=torch.int)
    for i in range(len(ori_size)-1, -1, -1):
        j = flat_index % ori_size[i]
        flat_index = flat_index // ori_size[i]
        res[i] = j
    return res


if __name__ == "__main__":
    x=torch.randn(100,4)
    reshaped_x=x.reshape(-1)

    # x=F.softmax(x,dim=-1)

    dist=MaskedCategorical(probs=reshaped_x, mask=x>0.)
    while True:
        index = dist.sample()
        orig_index=

        print(x[index])
    # ori_index=get_index(view_index,x.shape)

    # print(x[ori_index[0],ori_index[1],ori_index[2],ori_index[3]])
    # print(z[0,view_index])
    #
    # print(ori_index)
    # print(view_index)
    # x_index=







    # probs = torch.squeeze(dist.log_prob(action)).item()

