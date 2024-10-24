from torch import nn
from models.resunet import res_unet
import torch.nn.functional as F
from registration import standardize_depth, camera

suction_sampler_model_state_path=r'suction_sampler_model_state'
use_bn=False
use_in=True

def reshape_for_layer_norm(tensor,camera=camera,reverse=False):
    if reverse==False:
        channels=tensor.shape[1]
        tensor=tensor.permute(0,2,3,1).reshape(-1,channels)
        return tensor
    else:
        batch_size=int(tensor.shape[0]/(camera.width*camera.height))
        channels=tensor.shape[-1]
        tensor=tensor.reshape(batch_size,camera.height,camera.width,channels).permute(0,3,1,2)
        return tensor

class suction_sampler_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1,Batch_norm=use_bn,Instance_norm=use_in).to('cuda')
        self.decoder= nn.Sequential(
            nn.Linear(64, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(True),
            nn.Linear(32, 16, bias=False),
            nn.LayerNorm([16]),
            nn.ReLU(True),
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