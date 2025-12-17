import math
import os
import time
import torch.nn.functional as F
import mujoco
import numpy as np
import torch

from GraspAgent_2.training.sample_random_grasp import quat_between_batch
from GraspAgent_2.utils.Multi_finger_hand_env import MojocoMultiFingersEnv
from GraspAgent_2.utils.quat_operations import quat_rotate_vector, quat_mul, bulk_quat_mul, quat_between


class CasiaHandEnv(MojocoMultiFingersEnv):
    def __init__(self,root,max_obj_per_scene=2,is_tendon_control=False):
        self.scene_xml_file='/scene.xml' if is_tendon_control else '/scene_s.xml'
        self.hand_xml_file="hand.xml" if is_tendon_control else "hand_s.xml"
        super().__init__(root=root,max_obj_per_scene=max_obj_per_scene,key='CasiaHand_s')
        self.is_tendon_control=is_tendon_control

        self.root=root

        self.default_finger_joints = [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.default_ctrl=self.decode_finger_ctrl([0.,0.,0.])

        self.contact_pads_geom_ids=[[2,3,4],[12,17,22,23],[26,31,36,37,40,45,50,51,54,59,64,65,68,73,78,79]] # (pad1,pad2,pad3), ft1, (ft2,ft3,ft4,ft5)

        # self.intilize_finger_joints()

        # self.contact_pads_info()


    def  decode_finger_ctrl(self,fingers):
        # print(args)
        if self.is_tendon_control:
            j_th = 0.091 - fingers[0] * 0.027
            j_fm = 0.091 - fingers[1] * 0.037  # forefinger and midmiddle finger
            j_rl = 0.091 - fingers[2] * 0.037  # ring finger and little finger
        else:
            # j form 0 to 1 represent open to close
            j_th = fingers[0] *  7
            j_fm = fingers[1] *  7.5
            j_rl = fingers[2] *  7.5
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





    def check_graspness(self,hand_pos,hand_quat,hand_fingers,obj_pose=None,view=False,iterations=600,hard_level=0.,shake=True):
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
            # self.static_view(1000)
            return in_scope, False, ini_contact_with_obj, ini_contact_with_floor,None,None,None
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++',self.default_finger_joints)

        delta=[0, 0, 0.01]
        self.d.ctrl = self.decode_finger_ctrl(hand_fingers)
        shake_amp = .1
        shake_f = 200  # Hz

        for i in range(60):
            #Rise phase
            if 7 < i < 30:
                self.d.mocap_pos[0] = self.d.mocap_pos[0] + delta

            # shake phase
            if 50 > i > 30:
                if i==31:
                    grasp_success, n_grasp_contact1, self_collide1 = self.check_valid_grasp(minimum_contact_points=2)
                    if not grasp_success or not shake: break
                t = i * self.m.opt.timestep
                phase = 2 * np.pi * shake_f * t
                shake = shake_amp * np.array([np.sin(phase),
                                              np.sin(phase + 2.1),
                                              np.sin(phase + 4.2)])
                # shake = shake_amp * np.sin(2 * np.pi * shake_f * t)
                self.d.mocap_pos[0] += shake  # vertical shake (z)
            mujoco.mj_step(self.m, self.d)

        # After stepping
        # grasp_success = self.check_grasped_obj()
        if grasp_success:
            stable_grasp,n_grasp_contact2,self_collide2 = self.check_valid_grasp(minimum_contact_points=2)
            return in_scope,grasp_success,ini_contact_with_obj, ini_contact_with_floor,min(n_grasp_contact1,n_grasp_contact2),self_collide1 or self_collide2,stable_grasp
        else:
            return in_scope,grasp_success,ini_contact_with_obj, ini_contact_with_floor,n_grasp_contact1,self_collide1,None

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

        delta=[0, 0, 0.01]
        self.d.ctrl = self.decode_finger_ctrl(hand_fingers)
        shake_amp = .1
        shake_f = 200  # Hz
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            for i in range(60):

                step_start = time.time()

                if 7 < i < 30:
                    self.d.mocap_pos[0] = self.d.mocap_pos[0] + delta

                # shake phase
                if 50>i > 30:
                    t = i * self.m.opt.timestep
                    phase = 2 * np.pi * shake_f * t
                    shake = shake_amp * np.array([np.sin(phase),
                                                  np.sin(phase + 2.1),
                                                  np.sin(phase + 4.2)])
                    # shake = shake_amp * np.sin(2 * np.pi * shake_f * t)
                    self.d.mocap_pos[0] += shake  # vertical shake (z)

                mujoco.mj_step(self.m, self.d)

                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                # time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                # if time_until_next_step > 0:
                #     time.sleep(time_until_next_step)

        # After stepping
        # grasp_success = self.check_grasped_obj()
        grasp_success,n_grasp_contact,self_collide = self.check_valid_grasp(minimum_contact_points=2)
        # if grasp_success:grasp_success= self.safety_fingers_check()
        # print(Fore.CYAN,f'final d.mocap_quat[0] {self.d.mocap_quat[0]}',Fore.RESET)
        # print(Fore.CYAN,f'final d.qpos[3:3+4] {self.d.qpos[3:3 + 4]}',Fore.RESET)
        self.static_view(1000)

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

    env=CasiaHandEnv(root=root_dir + "/speed_hand/",max_obj_per_scene=1,is_tendon_control=False)

    # env.view_geom_names_and_ids()

    pose=torch.tensor([[0.0,  1.0, 0.0,  0.9955,  0.0,  1.0,  0.4976,  0.5088,
         -0.0086],
        [ 0.0412, -0.8468, -0.5293,  0.6404,  0.7680, -0.4819,  0.9882, -0.4003,
         -0.5279],
        [ 0.7783, -0.6193, -0.1017,  0.7983,  0.6008, -0.3374,  0.0456,  0.1272,
         -0.2949],
        [ 0.0598, -0.5058, -0.8436,  0.9527,  0.2877,  0.9470,  0.8910, -0.5385,
         -0.6717],
        [-0.5446, -0.4429, -0.7113,  0.1697,  0.9850,  0.9860,  0.5995,  0.8459,
         -0.4329],
        [ 0.1984, -0.6787, -0.6952,  0.9118,  0.3926,  0.3555,  0.9793,  0.3933,
         -0.3045],
        [ 0.3261, -0.7242, -0.6064,  0.7649,  0.6428,  0.4673,  0.4235, -1.3137,
         -0.5884],
        [ 0.2857, -0.6924, -0.6465, -0.3780,  0.9182,  0.3714,  0.3567,  0.0802,
         -0.4702],
        [-0.8274, -0.0254, -0.5611,  0.9550, -0.2966, -0.5362,  0.9900,  0.0914,
         -0.4519],
        [-0.3275,  0.4794, -0.7937,  0.9104, -0.4063,  0.3464,  0.8701,  0.1624,
         -0.4628],
        [-0.0855,  0.6678, -0.7362, -0.1635, -0.9865,  0.5630, -0.0383, -0.0365,
         -0.4395],
        [ 0.2654, -0.2295, -0.9234,  0.5171, -0.8519,  0.8147,  0.9345,  0.2407,
         -0.4695],
        [-0.0247,  0.0914, -0.9941,  0.4385,  0.8954, -0.4626,  0.7124, -0.9952,
         -0.5359],
        [ 0.8844, -0.0857,  0.4587,  0.2890, -0.9573,  0.6619,  0.6681, -1.1052,
         -0.5013],
        [-0.8085,  0.1053, -0.5656,  0.6538,  0.7551, -0.3634,  0.6063,  0.3103,
         -0.4192],
        [-0.2032, -0.3109, -0.9285, -0.9953, -0.0964, -0.2044,  0.8615, -1.3325,
         -0.4563],
        [ 0.1133,  0.7149, -0.6900,  0.7998,  0.6003, -1.2266, -0.6499,  0.6076,
         -0.4110],
        [-0.0570, -0.1956, -0.9743, -0.5406, -0.8376,  0.7440,  0.7304, -0.1940,
         -0.4682],
        [ 0.9356, -0.3153, -0.1591, -0.8731,  0.4876,  0.3246,  0.6528,  0.5598,
         -0.5414],
        [-0.1478, -0.1511, -0.9766,  0.3836,  0.9172,  0.1407, -0.3656, -0.6420,
         -0.6463],
        [ 0.4242,  0.8453,  0.3247,  0.2268,  0.9736,  0.4951,  0.9003, -0.7752,
         -0.2776],
        [-0.2190,  0.5156, -0.8127, -0.3273,  0.9344,  0.2336,  0.3139, -0.2569,
         -0.2908],
        [ 0.2845,  0.4746, -0.8284,  0.2821, -0.9531, -0.0511,  0.9877, -0.4836,
         -0.6749],
        [ 0.3338,  0.1419,  0.9242,  0.8043,  0.5943, -0.1340,  0.7367, -0.0839,
         -0.6806],
        [-0.2861,  0.7300, -0.6207,  0.5403,  0.8415,  0.1771, -1.9467,  0.3248,
         -0.4917],
        [-0.4559,  0.6004, -0.6564,  0.3961, -0.9130,  0.9145,  0.2623,  0.9900,
         -1.0213],
        [-0.2283,  0.7369,  0.6363, -0.3982,  0.9173,  0.2500,  0.8522,  0.2802,
         -0.6063],
        [-0.3527, -0.9341, -0.0360,  0.9048,  0.4247,  0.9900,  0.2981,  0.9895,
         -0.7158],
        [-0.0082, -0.9864, -0.1642,  0.1323,  0.9912,  0.3528, -1.3462,  0.9900,
         -0.4100],
        [ 0.6297, -0.7519, -0.1877,  0.9389, -0.3367,  0.7204, -0.1334,  0.5342,
         -0.5760]], device='cuda:0')








    # z=quaternion_pairwise_angular_distance(quats, eps=1e-7, degrees=True)
    # print(z)

    psoe2 = pose[0]


    for i in range(1000):
        env.drop_new_obj(selected_index=1,obj_pose=[0, 0, -0.09],obj_quat=[1,0,0,0], stablize=True)


        # quat = sample_quat(1,f=1.,ref_quat = torch.tensor([[1., 1., 0., 0.]],device='cuda'))[0].cpu().tolist()
        # delta = quat_rotate_vector(np.array(quat), np.array([0, 0, 1]))
        # print('delta= ', delta)
        # quat=pose[i,0:4].cpu().tolist()
        # fingers=pose[i,4:4+3].cpu().tolist()
        # transition=pose[i,4+3:].cpu().tolist()
        from GraspAgent_2.training.CH_training import process_pose
        quat, fingers, shifted_point = process_pose(np.array([0,0,0.5]), psoe2, view=True)
        psoe2[4]+=0.1
        psoe2[5]-=0.1

        print(psoe2)
        # print(f'combine {quats[i]}, {quats[i+1]}')
        # z = quaternion_angular_distance(quats[i], quats[i+1])
        # print(z)
        # z = quaternion_angular_distance(quats[i], -quats[i+1])
        # print(z)
        # env.passive_viewer(pos=[0.0, 0.0, 0.3],quat=quats[i].cpu().tolist(),ctrl=ctrl)
        # env.passive_viewer(pos=[0.0, 0.0, 0.3],quat=quats[i+1].cpu().tolist(),ctrl=ctrl)

        # quat=combine_quaternions(quats[i], quats[i+1], 0.5, 0.5, eps=1e-8).cpu().tolist()
        # print(f'get {quat}')
        # quat = [1.0, 0., 0., 0.]
        angle_rad = math.radians(30)  # convert degrees → radians
        cos_30 = math.cos(angle_rad)
        sin_30 = math.sin(angle_rad)

        # quat=quat_from_two_frames(v1=np.array([1.,0.,0.]),u1=np.array([0.,1.,0.]),v2=np.array([0.,sin_30,-cos_30]),u2=np.array([0.,cos_30,sin_30])).tolist()
        # print(quat)

        # transform_frame()
        def xy_to_quat_torch(v):
            """
            v: (..., 2) normalized XY vectors
            returns (..., 4) quaternion [w, x, y, z]
            """
            theta = torch.atan2(v[..., 1], v[..., 0])
            half = 0.5 * theta
            qw = torch.cos(half)
            qz = torch.sin(half)

            return torch.stack([
                qw,
                torch.zeros_like(qw),
                torch.zeros_like(qw),
                qz
            ], dim=-1)

        beta=torch.randn((2,))
        beta=F.normalize(beta,p=2,dim=0,eps=1e-8)
        beta_quat=xy_to_quat_torch(beta)

        # v2=quat_rotate_vector(quat, [cos_30,-sin_30,0])
        default_quat=quat_between(torch.tensor([cos_30,-sin_30,0]),torch.tensor([0.,0.,-1.]))

        quat=quat_mul(beta_quat,default_quat)


        alpha=torch.randn((3,))/5
        alpha[-1]=-1
        alpha=F.normalize(alpha,p=2,dim=0,eps=1e-8)

        print(alpha,'------')

        alpha_quat=quat_between(torch.tensor([0.,0.,-1.0]),alpha)
        quat=quat_mul(alpha_quat,quat).tolist()

        def grasp_frame_to_quat(alpha,beta,default_quat):
            beta = F.normalize(beta, p=2, dim=0, eps=1e-8)
            beta_quat = xy_to_quat_torch(beta)
            quat = quat_mul(beta_quat, default_quat)
            alpha = F.normalize(alpha, p=2, dim=0, eps=1e-8)
            alpha_quat = quat_between(torch.tensor([0., 0., -1.0]), alpha)
            quat = quat_mul(alpha_quat, quat).tolist()
            quat = F.normalize(quat, p=2, dim=0, eps=1e-8)
            return quat

        # print(v2)
        # v2*=0

        # v3=np.copy(v2)*2
        # v3[-1]+=0.2
        # v2[-1]+=0.2

        # env.passive_viewer(pos=[0.0, 0.0, 0.3],quat=quat,ctrl=ctrl)

        # delta = quat_rotate_vector(ncheck_graspnessp.array(quat), np.array([0, 0, 1]))
        # print('delta= ', delta)
        # print(env.check_collision([0,0,0],quat))
        # continue
        # env.view_grasp( shifted_point,quat, hand_fingers=fingers, view=True)
        # print(        env.check_graspness( [0,0,0],quat, hand_fingers=[1,1,1], view=True))
        # print(env.d.ctrl)
        # print(env.obj_xy_positions)
        # env.passive_viewer(pos=[0,0,0], quat=quat,ctrl=ctrl)
        # while True:
        env.manual_view(pos=shifted_point, quat=quat,fingers=fingers)
            # for i in range(3):
            #     k=7+i
            #     print(f'----joint {i+1}')
            #     print(env.d.qpos[k:k+1])
            #     print(env.d.qpos[k+3:3+k+1])
            #     print(env.d.qpos[k+6:6+k+1])
            #     print(env.d.qpos[k+9:9+k+1])
            #     print(env.d.qpos[k+12:12+k+1])

        # env.get_scene_preception(view=True)