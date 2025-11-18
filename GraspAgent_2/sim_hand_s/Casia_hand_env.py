import os
import time

import torch.nn.functional as F
import mujoco
import numpy as np
import torch
from matplotlib import pyplot as plt

from GraspAgent_2.training.sample_random_grasp import quat_between_batch
from GraspAgent_2.utils.Multi_finger_hand_env import MojocoMultiFingersEnv
from GraspAgent_2.utils.quat_operations import quat_rotate_vector, quat_mul, bulk_quat_mul, quaternion_angular_distance, \
    combine_quaternions, quaternion_pairwise_angular_distance


class CasiaHandEnv(MojocoMultiFingersEnv):
    def __init__(self,root,max_obj_per_scene=2):
        super().__init__(root=root,max_obj_per_scene=max_obj_per_scene)

        self.default_finger_joints = [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.last_hand_geom_id=44

    def decode_finger_ctrl(self,j1, j2, j3):
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

    def check_graspness(self,hand_pos,hand_quat,hand_fingers,obj_pose=None,view=False,iterations=1000,shake_intensity=0.05):

        if obj_pose is None: obj_pose=self.objects_poses

        in_scope = self.check_fingers_scope(hand_fingers)
        # v2 = quat_rotate_vector(hand_quat, [0, 1, 0])
        # if v2[-1]<0:in_scope=False

        if not in_scope:
            return in_scope,None,None, None

        hand_fingers = self.clip_fingers_to_scope(hand_fingers)

        # if not in_scope: return False, None, None,None,None

        self.d.time = 0.0
        self.d.mocap_pos[0] = hand_pos
        self.d.mocap_quat[0] = hand_quat

        self.d.qpos = hand_pos + hand_quat + self.default_finger_joints + obj_pose


        delta=[0, 0, 0.001]

        self.d.ctrl = self.decode_finger_ctrl(hand_fingers[0],hand_fingers[1],hand_fingers[2])

        mujoco.mj_step(self.m, self.d)
        ini_contact_with_obj, ini_contact_with_floor = self.check_hand_contact()
        if ini_contact_with_obj or ini_contact_with_floor:
            return in_scope, False, ini_contact_with_obj, ini_contact_with_floor

        for i in range(iterations):
            if i > 200:
                if i < 600:
                    self.d.mocap_pos[0] = self.d.mocap_pos[0] + delta
                elif i == 600:
                    final_contact_with_obj, contact_with_floor = self.check_hand_contact()
                    if not final_contact_with_obj or contact_with_floor:
                        return in_scope, False, ini_contact_with_obj, ini_contact_with_floor
                else:
                    if shake_intensity is None: break
                    if i % 6 == 0:
                        self.d.mocap_pos[0] = self.d.mocap_pos[0] + [shake_intensity, 0, 0]
                    elif i % 6 == 1:
                        self.d.mocap_pos[0] = self.d.mocap_pos[0] + [-shake_intensity, 0, 0]
                    elif i % 6 == 2:
                        self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, shake_intensity, 0]
                    elif i % 6 == 3:
                        self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, -shake_intensity, 0]
                    elif i % 6 == 4:
                        self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, 0, shake_intensity]
                    elif i % 6 == 5:
                        self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, 0, -shake_intensity]

            # for _ in range(20):
            mujoco.mj_step(self.m, self.d)
        # After stepping
        # grasp_success = self.check_grasped_obj()
        grasp_success, contact_with_floor = self.check_hand_contact()
        grasp_success=grasp_success and not contact_with_floor
        # if grasp_success:grasp_success= self.safety_fingers_check()
        # if view:self.static_view(1000)

        return in_scope,grasp_success,ini_contact_with_obj, ini_contact_with_floor
    def view_grasp(self,hand_pos,hand_quat,hand_fingers,obj_pose=None,view=False,iterations=1000,shake_intensity=0.05):


        if obj_pose is None: obj_pose=self.objects_poses

        in_scope = self.check_fingers_scope(hand_fingers)
        # v2 = quat_rotate_vector(hand_quat, [0, 1, 0])
        # if v2[-1]<0:in_scope=False

        if not in_scope:
            return in_scope,None,None, None

        hand_fingers = self.clip_fingers_to_scope(hand_fingers)

        # if not in_scope: return False, None, None,None,None

        self.d.time = 0.0
        self.d.mocap_pos[0] = hand_pos
        self.d.mocap_quat[0] = hand_quat

        self.d.qpos = hand_pos + hand_quat + self.default_finger_joints + obj_pose



        delta=[0, 0, 0.001]

        self.d.ctrl = self.decode_finger_ctrl(hand_fingers[0],hand_fingers[1],hand_fingers[2])

        mujoco.mj_step(self.m, self.d)
        ini_contact_with_obj, ini_contact_with_floor = self.check_hand_contact()
        if ini_contact_with_obj or ini_contact_with_floor:
            return in_scope, False, ini_contact_with_obj, ini_contact_with_floor

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            for i in range(iterations):
                step_start = time.time()
                if i > 200:
                    if i < 600:
                        self.d.mocap_pos[0] = self.d.mocap_pos[0] + delta
                    elif i == 600:
                        final_contact_with_obj, contact_with_floor = self.check_hand_contact()
                        if not final_contact_with_obj or contact_with_floor:
                            return in_scope, False, ini_contact_with_obj, ini_contact_with_floor
                    else:
                        if shake_intensity is None: break
                        if i % 6 == 0:
                            self.d.mocap_pos[0] = self.d.mocap_pos[0] + [shake_intensity, 0, 0]
                        elif i % 6 == 1:
                            self.d.mocap_pos[0] = self.d.mocap_pos[0] + [-shake_intensity, 0, 0]
                        elif i % 6 == 2:
                            self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, shake_intensity, 0]
                        elif i % 6 == 3:
                            self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, -shake_intensity, 0]
                        elif i % 6 == 4:
                            self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, 0, shake_intensity]
                        elif i % 6 == 5:
                            self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, 0, -shake_intensity]

                # for _ in range(20):
                mujoco.mj_step(self.m, self.d)

                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            # After stepping
            # grasp_success = self.check_grasped_obj()
            grasp_success, contact_with_floor = self.check_hand_contact()
            grasp_success=grasp_success and not contact_with_floor
            # if grasp_success:grasp_success= self.safety_fingers_check()
            # if view:self.static_view(1000)

        return in_scope,grasp_success,ini_contact_with_obj, ini_contact_with_floor

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

    env=CasiaHandEnv(root=root_dir + "/speed_hand/")
    env.drop_new_obj(selected_index=4,obj_pose=[0,0.3,0.3],stablize=True)

    ctrl=env.decode_finger_ctrl(0.4,0.3,0.3)

    quats=torch.tensor([[-0.0819,  0.9438, -0.3183,  0.0353],
        [-0.8408, -0.1210, -0.4862, -0.2047],
        [ 0.3671,  0.7792,  0.3374, -0.3798],
        [ 0.0487, -0.3993,  0.1784,  0.8980],
        [-0.5468, -0.3568,  0.3844,  0.6527],
        [-0.5869,  0.2709, -0.4614, -0.6077],
        [-0.5654, -0.1649, -0.7486,  0.3045],
        [-0.6323, -0.0258, -0.1032,  0.7674],
        [-0.8252, -0.3111, -0.2879,  0.3733],
        [ 0.5186, -0.3838, -0.2266, -0.7297],
        [ 0.1109, -0.5026, -0.8505,  0.1089],
        [-0.2653, -0.4505, -0.7359, -0.4302],
        [-0.0905,  0.5569, -0.8234,  0.0605],
        [-0.3720,  0.1203, -0.8742, -0.2878],
        [-0.0714, -0.3710, -0.4017,  0.8342],
        [-0.8844, -0.1979, -0.2963, -0.3014]], device='cuda:0')


    # z=quaternion_pairwise_angular_distance(quats, eps=1e-7, degrees=True)
    # print(z)


    for i in range(1000):
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
        print(v2)
        v2*=0.1

        v3=np.copy(v2)*2
        v3[-1]+=0.2
        v2[-1]+=0.2

        # env.passive_viewer(pos=[0.0, 0.0, 0.3],quat=quat,ctrl=ctrl)

        # quat = [.707, 0., 0.707, 0.]
        # delta = quat_rotate_vector(np.array(quat), np.array([0, 0, 1]))
        # print('delta= ', delta)
        env.manual_view(pos=v2.tolist(), quat=quat)
        env.manual_view(pos=(v3).tolist(), quat=quat)


