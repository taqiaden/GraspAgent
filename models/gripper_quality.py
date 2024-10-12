import torch
from torch import nn
from lib.depth_map import depth_to_mesh_grid
from models.decoders import att_res_decoder_A, res_block, res_block_mlp_LN, att_res_mlp_LN
from models.resunet import res_unet, batch_norm_relu
from registration import camera, standardize_depth

gripper_quality_model_state_path=r'gripper_quality_model_state'

gripper_scope_model_state_path=r'gripper_scope_model_state'

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

class compact_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.att_block = att_res_mlp_LN(in_c1=64, in_c2=5, out_c=1,drop_out_ratio=0.35).to('cuda')
        # self.decoder = res_block_mlp_LN(64+5,32,1,drop_out_ratio=0.5).to('cuda')
        # self.tanh=nn.Tanh()
        self.decoder= nn.Sequential(
            nn.Linear(64+9, 32, bias=False),
            nn.LayerNorm([32]),
            nn.ReLU(True),
            nn.Linear(32, 1),
        ).to('cuda')


    def forward(self, features,pose ):
        b = pose.shape[0]

        '''get spatial information'''
        xymap = depth_to_mesh_grid(camera)
        xymap = xymap.repeat(b, 1, 1, 1)

        '''reshape and concatenate'''
        xymap_2d = reshape_for_layer_norm(xymap, camera=camera, reverse=False)
        features_2d = reshape_for_layer_norm(features, camera=camera, reverse=False)

        '''backbone'''
        pose_2d = reshape_for_layer_norm(pose, camera=camera, reverse=False)
        pose_2d = torch.cat([pose_2d, xymap_2d], dim=1)

        '''decode'''
        # output_2d = self.att_block(features_2d, pose_2d)
        f = torch.cat([pose_2d, features_2d], dim=1)
        output_2d = self.decoder(f)
        # output_2d=self.tanh(output_2d)

        output = reshape_for_layer_norm(output_2d, camera=camera, reverse=True)

        return output


class gripper_quality_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=use_bn, Instance_norm=use_in).to('cuda')
        self.decoder=compact_decoder().to('cuda')

    def forward(self, depth,pose ):
        '''input standardization'''
        depth = standardize_depth(depth)

        '''backbone'''
        features = self.back_bone(depth)

        '''decode'''
        output = self.decoder(features, pose)
        return output
