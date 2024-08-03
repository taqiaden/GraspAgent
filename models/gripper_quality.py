from torch import nn
from models.decoders import decoder2, att_res_decoder_A
from models.resunet import res_unet

gripper_quality_model_state_path=r'gripper_quality_model_state'

class gripper_quality_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.back_bone = res_unet(in_c=1,Batch_norm=False,Instance_norm=True).to('cuda')
        self.decoder=att_res_decoder_A(in_c1=64,in_c2=7,out_c=1,Batch_norm=False,Instance_norm=True).to('cuda')
    def forward(self, depth,pose ):
        spatial_features=self.back_bone(depth)
        output=self.decoder(spatial_features,pose)
        return output
