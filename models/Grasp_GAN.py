import torch
from torch import nn
import torch.nn.functional as F
from lib.depth_map import depth_to_mesh_grid
from models.decoders import att_res_decoder_A, res_block
from models.resunet import res_unet, batch_norm_relu
from registration import camera

dropout_p=0.0
gripper_sampler_path=r'gripper_sampler_model_state'
gripper_critic_path=r'gripper_critic_model_state'

def gripper_output_normalization(output):
    '''approach normalization'''
    approach=output[:, 0:3, :]
    approach=F.normalize(approach, dim=1)

    '''beta normalization'''
    beta=output[:, 3:5, :]
    beta=F.normalize(beta, dim=1)

    dist=output[:, 5:6, :]
    width=output[:, 6:7, :]
    normalized_output=torch.cat([approach,beta,dist,width],dim=1)
    return normalized_output

class gripper_sampler_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=True, Instance_norm=False).to('cuda')
        self.get_approach=att_res_decoder_A(in_c1=64, in_c2=2, out_c=3, Batch_norm=True, Instance_norm=False).to('cuda')
        self.get_beta=att_res_decoder_A(in_c1=64, in_c2=5, out_c=2, Batch_norm=True, Instance_norm=False).to('cuda')
        self.get_dist=att_res_decoder_A(in_c1=64, in_c2=7, out_c=1, Batch_norm=True, Instance_norm=False).to('cuda')
        self.get_width=att_res_decoder_A(in_c1=64, in_c2=8, out_c=1, Batch_norm=True, Instance_norm=False).to('cuda')

        # self.decoder = res_block(in_c=64, medium_c=32, out_c=16, Batch_norm=True, Instance_norm=False).to('cuda')
        # self.b1 = batch_norm_relu(16, Batch_norm=True, Instance_norm=False).to('cuda')
        # self.c1 = nn.Conv2d(16, 7, kernel_size=1).to('cuda')
        # self.test = nn.Conv2d(1, 7, kernel_size=1).to('cuda')

    def forward(self, depth):
        # return self.test(depth)
        representation = self.back_bone(depth)

        # x=self.decoder(representation)
        # x=self.b1(x)
        # x=self.c1(x)
        # output = gripper_output_normalization(x)
        # return output

        '''get spatial information'''
        b=depth.shape[0]
        xymap=depth_to_mesh_grid(camera)

        params1=xymap.repeat(b,1,1,1)
        approach=self.get_approach(representation,params1)

        params2=torch.cat([approach,params1],dim=1)
        beta=self.get_beta(representation,params2)

        params3=torch.cat([beta,params2],dim=1)
        dist=self.get_dist(representation,params3)

        params4=torch.cat([dist,params3],dim=1)
        width=self.get_width(representation,params4)

        output = torch.cat([approach, beta, dist, width], dim=1)
        output=gripper_output_normalization(output)
        return output

class critic_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1, Batch_norm=True, Instance_norm=False).to('cuda')
        self.decoder = att_res_decoder_A(in_c1=64, in_c2=7, out_c=1, Batch_norm=True, Instance_norm=False).to('cuda')

        # self.test = nn.Conv2d(1+7, 1, kernel_size=1).to('cuda')
    def forward(self, depth,pose):
        # test_f=torch.cat([depth,pose],dim=1)
        # return self.test(test_f)
        features = self.back_bone(depth)
        output = self.decoder(features, pose)
        return output

