import torch.nn.functional as F
from torch import nn

from lib.models_utils import reshape_for_layer_norm
from models.resunet import res_unet
from registration import standardize_depth, camera

suction_sampler_model_state_path=r'suction_sampler_model_state'
use_bn=False
use_in=True


class suction_sampler_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1,Batch_norm=use_bn,Instance_norm=use_in).to('cuda')
        self.decoder= nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(),
            nn.Linear(16, 3),
        ).to('cuda')
    def forward(self, depth ):
        depth = standardize_depth(depth)

        features=self.back_bone(depth)
        features_2d = reshape_for_layer_norm(features, camera=camera, reverse=False)

        output_2d=self.decoder(features_2d)
        output = reshape_for_layer_norm(output_2d, camera=camera, reverse=True)
        output=F.normalize(output,dim=1)
        return output