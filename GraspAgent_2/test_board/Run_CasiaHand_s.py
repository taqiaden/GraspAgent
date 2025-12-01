from pathlib import Path
import torch
from GraspAgent_2.model.CH_model import CH_G, CH_model_key
from GraspAgent_2.sim_hand_s.Casia_hand_env import CasiaHandEnv
from GraspAgent_2.training.CH_training import process_pose
from check_points.check_point_conventions import GANWrapper
import torch.nn.functional as F

from lib.rl.masked_categorical import MaskedCategorical

if __name__ == "__main__":
    '''Load model'''
    gan = GANWrapper(CH_model_key, CH_G)
    gan.ini_generator(train=False)
    model=gan.generator

    '''Mojoco env'''
    ch_env = CasiaHandEnv(root=str(Path(__file__).resolve().parent.parent)+ "/sim_hand_s/speed_hand/")
    ch_env.initialize()

    '''Inference loop'''
    while True:
        print('------------------------------------------------')
        '''get perception'''
        depth = torch.load(str(Path(__file__).resolve().parent)+'/depth_ch_tmp')
        point_cloud, floor_mask = ch_env.depth_to_pointcloud(depth.cpu().numpy(), ch_env.intr, ch_env.extr)
        floor_mask=torch.from_numpy(floor_mask).cuda()


        '''dense processing'''
        with torch.no_grad():
            grasp_pose, grasp_quality_logits, grasp_collision_logits = model(
                depth[None, None, ...], ~floor_mask.reshape(1,1,600,600))

            grasp_quality = torch.clamp(grasp_quality_logits, 0, 1).reshape(-1)
            objects_collision=F.sigmoid(grasp_collision_logits[0,0]).reshape(-1)
            floor_collision=F.sigmoid(grasp_collision_logits[0,1]).reshape(-1)
            grasp_pose=grasp_pose[0].permute(1, 2, 0).reshape(360000,8)

            selection_p=grasp_quality**2
            selection_mask=(~floor_mask) & (objects_collision<0.5) & (floor_collision<0.5)

        '''pick action'''
        # option 1: probabilistic
        dist = MaskedCategorical(probs=selection_p, mask=selection_mask)
        action_id = dist.sample().item()

        # option 2: deterministic
        # probs = selection_p.masked_fill(~selection_mask, -float('inf'))
        # action_id = probs.argmax().item()

        '''process action'''
        target_point=point_cloud[action_id]
        target_grasp_pose=grasp_pose[action_id]

        quat,fingers,shifted_point=process_pose(target_point, target_grasp_pose)

        '''view in Mojoco'''
        ch_env.passive_viewer(pos=shifted_point,quat=quat)

        '''execute grasp'''


