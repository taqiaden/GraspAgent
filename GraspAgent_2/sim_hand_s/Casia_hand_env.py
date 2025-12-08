import os
import time

import torch.nn.functional as F
import mujoco
import numpy as np
import torch
from colorama import Fore
from matplotlib import pyplot as plt

from GraspAgent_2.training.sample_random_grasp import quat_between_batch
from GraspAgent_2.utils.Multi_finger_hand_env import MojocoMultiFingersEnv
from GraspAgent_2.utils.quat_operations import quat_rotate_vector, quat_mul, bulk_quat_mul, quaternion_angular_distance, \
    combine_quaternions, quaternion_pairwise_angular_distance


class CasiaHandEnv(MojocoMultiFingersEnv):
    def __init__(self,root,max_obj_per_scene=2):
        super().__init__(root=root,max_obj_per_scene=max_obj_per_scene)

        self.default_finger_joints = [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.default_ctrl=self.decode_finger_ctrl(0.,0.,0.)

        self.contact_pads_geom_ids=[[2,3,4],[16],[23,30,37,44]] # (pad1,pad2,pad3), ft1, (ft2,ft3,ft4,ft5)

        # self.intilize_finger_joints()

        # self.contact_pads_info()

    def intilize_finger_joints(self):
        self.initialize()

        self.d.mocap_pos[0] = [0,0,0]
        self.d.mocap_quat[0] = [1,0,0,0]
        self.d.ctrl=self.default_ctrl

        n_joints=len(self.default_finger_joints)
        for i in range(10):
            mujoco.mj_step(self.m, self.d)

            f_joints=self.d.qpos[7:7+n_joints]
            print(f_joints)

        exit()



    def  decode_finger_ctrl(self,j1, j2, j3):
        # j form 0 to 1 represent open to close
        j_th = 0.091 - j1 * 0.027
        j_fm = 0.091 - j2 * 0.037  # forefinger and midmiddle finger
        j_rl = 0.091 - j3 * 0.037  # ring finger and little finger
        return [j_th, j_fm, j_fm, j_rl, j_rl]

    def check_fingers_scope(self,fingers):
        fingers=np.array(fingers)

        return  np.all((fingers >= 0) & (fingers <= 1))

    def clip_fingers_to_scope(self,hand_fingers):
        return torch.clamp(torch.tensor(hand_fingers),min=0.01,max=0.99).tolist()

    def safety_fingers_check(self):
        fingers_state=self.d.qpos[3+4:3+4+15]
        for i in range(5):
            cumulative=fingers_state[i*5:i*5+3]
            if sum(cumulative)>4: return False

        return True


    def check_graspness(self,hand_pos,hand_quat,hand_fingers,obj_pose=None,view=False,iterations=600,hard_level=0.):
        self.restore_simulation_state()
        if obj_pose is None: obj_pose=self.objects_poses

        in_scope = self.check_fingers_scope(hand_fingers)
        if not in_scope: hand_fingers = self.clip_fingers_to_scope(hand_fingers)
        # v2 = quat_rotate_vector(hand_quat, [0, 1, 0])
        # if v2[-1]<0:in_scope=False

        # if not in_scope:
        #     return in_scope,None,None, None

        # if not in_scope: return False, None, None,None,None

        self.d.time = 0.0
        self.d.mocap_pos[0] = hand_pos
        self.d.mocap_quat[0] = hand_quat
        self.d.qpos = hand_pos + hand_quat + self.default_finger_joints + obj_pose
        self.d.ctrl *= 0
        mujoco.mj_step(self.m, self.d)
        ini_contact_with_obj, ini_contact_with_floor = self.check_hand_contact()
        if ini_contact_with_obj or ini_contact_with_floor:
            return in_scope, False, ini_contact_with_obj, ini_contact_with_floor,None,None
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++',self.default_finger_joints)

        delta=[0, 0, 0.001]
        self.d.ctrl = self.decode_finger_ctrl(hand_fingers[0],hand_fingers[1],hand_fingers[2])
        for i in range(300):
            if 70 < i < 300:
                self.d.mocap_pos[0] = self.d.mocap_pos[0] + delta
            # elif i > 300:
            #     if i == 301:
            #         self.d.ctrl = self.decode_finger_ctrl(min(hand_fingers[0] + 0.3, 1), min(hand_fingers[1] + 0.3, 1),
            #                                               min(hand_fingers[2] + 0.3, 1))
            #     elif i == 351:
            #         self.d.ctrl = self.decode_finger_ctrl(hand_fingers[0], hand_fingers[1], hand_fingers[2])

            # for _ in range(20):
            mujoco.mj_step(self.m, self.d)
        # After stepping
        # grasp_success = self.check_grasped_obj()
        grasp_success,n_grasp_contact,self_collide = self.check_valid_grasp(minimum_contact_points=1)
        # if grasp_success:grasp_success= self.safety_fingers_check()
        # if view:self.static_view(1000)

        return in_scope,grasp_success,ini_contact_with_obj, ini_contact_with_floor,n_grasp_contact,self_collide

    def view_grasp(self,hand_pos,hand_quat,hand_fingers,obj_pose=None,view=False,iterations=300,hard_level=0.   ):
        self.restore_simulation_state()
        if obj_pose is None: obj_pose=self.objects_poses

        in_scope = self.check_fingers_scope(hand_fingers)
        if not in_scope:hand_fingers = self.clip_fingers_to_scope(hand_fingers)
        # v2 = quat_rotate_vector(hand_quat, [0, 1, 0])
        # if v2[-1]<0:in_scope=False

        # if not in_scope:
        #     return in_scope,None,None, None

        # if not in_scope: return False, None, None,None,None

        self.d.time = 0.0
        self.d.mocap_pos[0] = hand_pos
        self.d.mocap_quat[0] = hand_quat
        self.d.qpos = hand_pos + hand_quat + self.default_finger_joints + obj_pose
        self.d.ctrl *= 0
        mujoco.mj_step(self.m, self.d)
        # self.static_view(1000)

        ini_contact_with_obj, ini_contact_with_floor = self.check_hand_contact()
        # if ini_contact_with_obj or ini_contact_with_floor:
        #     return in_scope, False, ini_contact_with_obj, ini_contact_with_floor

        # print(Fore.CYAN,f'initial d.mocap_quat[0] {self.d.mocap_quat[0]}',Fore.RESET)
        # print(Fore.CYAN,f'initial d.qpos[3:3+4] {self.d.qpos[3:3+4]}',Fore.RESET)

        delta=[0, 0, 0.001]
        self.d.ctrl = self.decode_finger_ctrl(hand_fingers[0],hand_fingers[1],hand_fingers[2])

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            for i in range(300):

                step_start = time.time()

                if 70 < i < 300:
                    self.d.mocap_pos[0] = self.d.mocap_pos[0] + delta
                # elif i>300:
                #     if i==301:
                #         self.d.ctrl = self.decode_finger_ctrl(min(hand_fingers[0]+0.3,1), min(hand_fingers[1]+0.3,1), min(hand_fingers[2]+0.3,1))
                #     elif i==351:
                #         self.d.ctrl = self.decode_finger_ctrl(hand_fingers[0], hand_fingers[1], hand_fingers[2])

                mujoco.mj_step(self.m, self.d)

                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        # After stepping
        # grasp_success = self.check_grasped_obj()
        grasp_success,n_grasp_contact,self_collide = self.check_valid_grasp(minimum_contact_points=1)
        # if grasp_success:grasp_success= self.safety_fingers_check()
        # print(Fore.CYAN,f'final d.mocap_quat[0] {self.d.mocap_quat[0]}',Fore.RESET)
        # print(Fore.CYAN,f'final d.qpos[3:3+4] {self.d.qpos[3:3 + 4]}',Fore.RESET)
        # self.static_view(1000)

        return in_scope,grasp_success,ini_contact_with_obj, ini_contact_with_floor,n_grasp_contact,self_collide

def sample_quat(size,f=0.5,ref_quat=None):
    ref_quat = torch.tensor([[0., 1., 0., 0.]],device='cuda') if ref_quat is None else ref_quat

    beta_quat=torch.zeros((size,4),device='cuda')
    beta_quat[:,[0,3]]=torch.randn((size, 2), device='cuda')
    beta_quat = F.normalize(beta_quat, dim=-1)

    approach=(torch.rand((10000, 3), device='cuda'))

    approach[:,[0,2]]=2*(approach[:,[0,2]]-0.5)
    U=approach[:,1]
    k=2
    approach[:, 1] = (1 + torch.sign(2*U - 1) * torch.abs(2*U - 1) ** (1 / (k + 1))) / 2 # this is the CDF inversion of the Probability function defined as (x/0.5-1)^k

    # y_np =x.cpu().numpy()
    # # Plot histogram
    # plt.hist(y_np, bins=50, range=(0, 1), density=True, alpha=0.7, color='skyblue')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # plt.title('Histogram of Tensor Data')
    # plt.show()

    approach[:, :2]*=f
    approach = F.normalize(approach, dim=-1)

    approach_quat=quat_between_batch(torch.tensor([0.0, 1.0, 0.0],device='cuda'),approach)
    approach_quat = F.normalize(approach_quat, dim=-1)

    quat=bulk_quat_mul(beta_quat,ref_quat)

    quat=bulk_quat_mul(approach_quat,quat)
    quat = F.normalize(quat, dim=-1)

    return quat

if __name__ == "__main__":
    root_dir = os.getcwd()  # current working directory

    env=CasiaHandEnv(root=root_dir + "/speed_hand/",max_obj_per_scene=1)

    env.view_geom_names_and_ids()

    ctrl=env.decode_finger_ctrl(1.,1.,1.)

    quats=torch.tensor([[-0.5518, -0.3340, -0.0619,  0.7617],
        [ 0.0717, -0.9683,  0.1292,  0.2014],
        [ 0.0374, -0.6781,  0.7262,  0.1065],
        [ 0.3026, -0.3890,  0.7988,  0.3452],
        [-0.1493, -0.8564,  0.4533,  0.1969],
        [ 0.1176, -0.0580,  0.9850,  0.1120],
        [ 0.1841, -0.0416,  0.7295,  0.6574],
        [-0.2887, -0.6570, -0.3511,  0.6014],
        [-0.2109, -0.9254, -0.1640,  0.2688],
        [-0.2043, -0.8114,  0.0863,  0.5408],
        [-0.1061, -0.2837,  0.7451,  0.5943],
        [-0.1094, -0.5798, -0.0821,  0.8032],
        [ 0.3223,  0.3637,  0.6345,  0.6010],
        [-0.3917, -0.7099, -0.5677,  0.1428],
        [ 0.1041, -0.6481,  0.5042,  0.5611],
        [-0.4685, -0.5503,  0.3955,  0.5668]], device='cuda:0')


    # z=quaternion_pairwise_angular_distance(quats, eps=1e-7, degrees=True)
    # print(z)


    for i in range(1000):
        env.drop_new_obj(selected_index=i,obj_pose=[0, 0.3, 0.3], stablize=True)

        # quat = sample_quat(1,f=1.,ref_quat = torch.tensor([[1., 1., 0., 0.]],device='cuda'))[0].cpu().tolist()
        # delta = quat_rotate_vector(np.array(quat), np.array([0, 0, 1]))
        # print('delta= ', delta)
        quat=quats[i].cpu().tolist()
        # print(f'combine {quats[i]}, {quats[i+1]}')
        # z = quaternion_angular_distance(quats[i], quats[i+1])
        # print(z)
        # z = quaternion_angular_distance(quats[i], -quats[i+1])
        # print(z)
        # env.passive_viewer(pos=[0.0, 0.0, 0.3],quat=quats[i].cpu().tolist(),ctrl=ctrl)
        # env.passive_viewer(pos=[0.0, 0.0, 0.3],quat=quats[i+1].cpu().tolist(),ctrl=ctrl)
        #
        # quat=combine_quaternions(quats[i], quats[i+1], 0.5, 0.5, eps=1e-8).cpu().tolist()
        # print(f'get {quat}')
        v2=quat_rotate_vector(quat, [1,0,0])
        # print(v2)
        v2*=0.1

        v3=np.copy(v2)*2
        v3[-1]+=0.2
        v2[-1]+=0.2

        # env.passive_viewer(pos=[0.0, 0.0, 0.3],quat=quat,ctrl=ctrl)

        # quat = [.0, 0., 0., 1.]
        # delta = quat_rotate_vector(np.array(quat), np.array([0, 0, 1]))
        # print('delta= ', delta)
        # env.check_collision( [0,0,0], quat, hand_fingers=None, view=True)
        # env.passive_viewer(pos=[0,0,0], quat=quat,ctrl=ctrl)
        env.manual_view(pos=(v3).tolist(), quat=quat)


        print(env.d.qpos[7:])
        print(env.d.ctrl)
        print(env.obj_xy_positions)
        # env.get_scene_preception(view=True)


