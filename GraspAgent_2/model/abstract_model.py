import torch
from torch import nn
from GraspAgent_2.model.sparse_encoder import SparseEncoderIN
from GraspAgent_2.model.Decoders import ContextGate_1d
from GraspAgent_2.utils.model_init import init_weights_he_normal


class D(nn.Module):
    def __init__(self,n_params):
        super().__init__()

        self.back_bone = SparseEncoderIN().to('cuda')

        self.att_block = ContextGate_1d(in_c1=512 , in_c2=n_params  ).to('cuda')

        self.back_bone.apply(init_weights_he_normal)
        self.att_block.apply(init_weights_he_normal)


    def forward(self,  pose,  cropped_spheres, detach_backbone=False):

        if detach_backbone:
            with torch.no_grad():
                anchor = self.back_bone(cropped_spheres)
        else:
            anchor = self.back_bone(cropped_spheres)

        print('D max val= ', anchor.max().item(), 'mean:', anchor.mean().item(),
              ' std:',
              anchor.std(dim=1).mean().item())

        scores = self.att_block(anchor[:,None], pose)

        return scores
