import torch
from torch import nn
from lib.depth_map import depth_to_mesh_grid
from models.decoders import att_res_decoder_A, res_block
from models.resunet import res_unet, batch_norm_relu
from registration import camera

gripper_quality_model_state_path=r'gripper_quality_model_state'

gripper_scope_model_state_path=r'gripper_scope_model_state'

class gripper_scope_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_block = res_block(7+2,16,8,Batch_norm=True,Instance_norm=False).to('cuda')
        self.b1 = batch_norm_relu(8, Batch_norm=True, Instance_norm=False).to('cuda')
        self.d = nn.Conv2d(8, 1, kernel_size=1).to('cuda')

    def forward(self, pose ):
        b=pose.shape[0]
        xymap=depth_to_mesh_grid(camera)
        xymap=xymap.repeat(b,1,1,1)
        features=torch.cat([xymap,pose],dim=1)

        x=self.res_block(features)
        x=self.b1(x)
        output=self.d(x)
        return output

class gripper_quality_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1,Batch_norm=True,Instance_norm=False).to('cuda')
        self.decoder=att_res_decoder_A(in_c1=64,in_c2=7,out_c=1,Batch_norm=True,Instance_norm=False).to('cuda')

    def forward(self, depth,pose ):
        spatial_features=self.back_bone(depth)
        output=self.decoder(spatial_features,pose)
        return output
