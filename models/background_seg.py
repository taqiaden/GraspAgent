from torch import nn

from models.decoders import res_block_mlp_LN
from models.resunet import res_unet
from registration import camera, standardize_depth

background_seg_model_state_path=r'background_seg_model_state'

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

class BackgroundSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')
        self.res=res_block_mlp_LN(in_c=64,medium_c=32,out_c=16,activation=nn.ReLU(True)).to('cuda')
        self.decoder= nn.Sequential(
            nn.LayerNorm(16),
            nn.ReLU(True),
            nn.Linear(16, 1),
        ).to('cuda')

    def forward(self, depth ):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''depth backbone'''
        depth_features = self.back_bone(depth)

        '''flatten'''
        depth_features_2d = reshape_for_layer_norm(depth_features, camera=camera, reverse=False)

        '''decode'''
        output_2d = self.decoder(self.res(depth_features_2d))

        '''unflatten'''
        output = reshape_for_layer_norm(output_2d, camera=camera, reverse=True)
        return output
