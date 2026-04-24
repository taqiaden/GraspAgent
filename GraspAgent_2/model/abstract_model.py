import torch
from torch import nn
from GraspAgent_2.model.sparse_encoder import SparseEncoderIN
from GraspAgent_2.model.Decoders import ContextGate_1d, ContextGate_2d
from GraspAgent_2.utils.model_init import init_weights_he_normal
from models.resunet import res_unet


class G(nn.Module):
    def __init__(self,sampler_decoder,n_params):
        super().__init__()
        self.back_bone = res_unet(in_c=2, Batch_norm=False, Instance_norm=True,
                                  relu_negative_slope=0., activation=nn.ReLU(), IN_affine=False,
                                  activate_skip=False).to('cuda')

        self.back_bone.apply(init_weights_he_normal)


        self.back_bone2_ = res_unet(in_c=1, Batch_norm=False, Instance_norm=False,
                                    relu_negative_slope=0., activation=None, IN_affine=False, activate_skip=False).to(
            'cuda')


        self.PoseSampler = sampler_decoder

        self.back_bone2_.apply(init_weights_he_normal)


        self.grasp_quality_=ContextGate_2d( 64, n_params, 1, in_c3=0, relu_negative_slope=0.,
        activation=nn.SiLU(), use_sin=False, normalize=False, bias=True, cyclic=False).to('cuda')



        self.grasp_collision_ =ContextGate_2d( 64, n_params, 1, in_c3=0, relu_negative_slope=0.,
        activation=nn.SiLU(), use_sin=False, normalize=False, bias=True, cyclic=False).to('cuda')

    def forward(self, depth, target_mask, latent_vector=None, model_B_poses=None, detach_backbone=False):
        max_ = 1.3
        min_ = 1.15
        standarized_depth_ = (depth.clone() - min_) / (max_ - min_)

        standarized_depth_ = (standarized_depth_ - 0.5) / 0.5
        print('Depth max=', standarized_depth_.max().item(), ', min=', standarized_depth_.min().item(), ', std=',
              standarized_depth_.std().item(), ', mean=', standarized_depth_.mean().item())


        input = torch.cat([standarized_depth_, target_mask], dim=1)

        if detach_backbone:
            with torch.no_grad():
                features = self.back_bone(input)  # if backbone is None else backbone(input)
                features2 = self.back_bone2_(standarized_depth_)  # *scale

        else:
            features = self.back_bone(input)  # if backbone is None else backbone(input)
            features2 = self.back_bone2_(standarized_depth_)  # *scale

        # features2=torch.cat([features2,scaled_depth_,depth_],dim=1)
        # features=torch.cat([features,scaled_depth_,depth_],dim=1)
        print('G b1 max val= ', features.max().item(), 'mean:', features.mean().item(), ' std:',
              features.std(dim=1).mean().item())
        print('G b2 max val= ', features2.max().item(), 'mean:', features2.mean().item(), ' std:',
              features2.std(dim=1).mean().item())
        # print('G b3 max val= ',features3.max().item(), 'mean:',features3.mean().item(),' std:',features3.std(dim=1).mean().item())

        depth_data = standarized_depth_

        gripper_pose = self.PoseSampler(features, depth_data, latent_vector)

        detached_gripper_pose = gripper_pose.detach().clone()


        detached_gripper_pose = torch.cat([detached_gripper_pose, depth_data], dim=1)


        grasp_collision = self.grasp_collision_(features2.detach(), detached_gripper_pose)
        grasp_collision = torch.cat(
            [grasp_collision, self.grasp_collision3(features2.detach(), detached_gripper_pose)], dim=1)
        grasp_collision = torch.cat(
            [grasp_collision, self.grasp_collision2(features2.detach(), detached_gripper_pose)], dim=1)

        # grasp_collision=self.grasp_collision_(features2,rotation=encoded_quat,transition=torch.cat([transition,depth_data],dim=1),fingers=None)

        # grasp_collision=self.sig(grasp_collision)

        grasp_quality = self.grasp_quality_(features2, detached_gripper_pose)

        # grasp_quality=grasp_quality-grasp_quality[~target_mask].mean()+self.bias
        # grasp_quality=grasp_quality*self.scale
        #
        # print(f'bias: {self.bias.item()}, scale: {self.scale.item()}')


        if model_B_poses is not None:
            gripper_pose_B = torch.cat([model_B_poses, depth_data], dim=1)
            B_grasp_quality = self.grasp_quality_(features2.detach(), gripper_pose_B)
        else:
            B_grasp_quality = None
        # grasp_quality=self.sig(grasp_quality)

        # grasp_quality=torch.rand_like(gripper_pose[:,0:1])
        # grasp_collision=torch.rand_like(gripper_pose[:,0:2])

        return gripper_pose, grasp_quality, grasp_collision, features2.detach(), B_grasp_quality

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
