import os
import time

import mujoco
import numpy as np
import torch

from GraspAgent_2.hands_config.sh_config import fingers_max, fingers_min
from GraspAgent_2.utils.Multi_finger_hand_env import MojocoMultiFingersEnv
from GraspAgent_2.utils.quat_operations import quat_rotate_vector, quat_between

class ShadowHandEnv(MojocoMultiFingersEnv):
    def __init__(self,root,max_obj_per_scene=2,objects_path=None):
        self.hand_xml_file = "shadow_hand/right_hand.xml"
        super().__init__(root=root,max_obj_per_scene=max_obj_per_scene,key='shadow_hand',objects_path=objects_path)
        self.root = root
        self.default_finger_joints = [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.default_ctrl = None

        # self.last_hand_geom_id=101

        self.contact_pads_geom_ids=[[23,28,34],[55,60,66],[87,92,98]] # (pad1,pad2,pad3)

    def  decode_fingers_initial_state(self,fingers):
        return [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0]

    def  decode_finger_ctrl(self,fingers):
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]
        return [fingers[0]*1.746-0.873, -1.4, 0, fingers[1]*1.746-0.873, -1.4, 0, fingers[2]*1.746-0.873, -1.4, 0]

    def  close_grip(self,fingers):
        return [fingers[0]*1.746-0.873, 0.785, 0.5, fingers[1]*1.746-0.873, 0.785, 0.5, fingers[2]*1.746-0.873, 0.785 , 0.5]
    def check_collision(self,hand_pos,hand_quat,hand_fingers=None,view=False):
        self.restore_simulation_state()


        self.d.mocap_pos[0] = hand_pos
        self.d.mocap_quat[0] = hand_quat

        self.d.qpos =hand_pos + hand_quat + self.decode_fingers_initial_state(hand_fingers)+ self.objects_poses

        self.d.ctrl=self.decode_finger_ctrl(hand_fingers)

        mujoco.mj_step(self.m, self.d)

        '''check initial contact'''
        contact_with_obj, contact_with_floor = self.check_hand_contact()

        if view:
            print(f'contact_with_obj , contact_with_floor: {contact_with_obj , contact_with_floor}')
            self.static_view(1000)

        return  contact_with_obj , contact_with_floor
    def check_graspness(self,hand_pos,hand_quat,hand_fingers,obj_pose=None,view=False,iterations=600,hard_level=0.,shake=True,update_obj_prob=False):

        self.restore_simulation_state()

        if obj_pose is None: obj_pose=self.objects_poses

        in_scope = max(hand_fingers)<=1. and min(hand_fingers)>=0.
        if not in_scope: hand_fingers = torch.clamp(torch.tensor(hand_fingers),min=0.01,max=0.99).tolist()
        grasped_obj=None

        warning_flag = False

        self.d.time = 0.0
        self.d.mocap_pos[0] = hand_pos
        self.d.mocap_quat[0] = hand_quat
        # try:
        self.d.qpos = hand_pos + hand_quat + self.decode_fingers_initial_state(hand_fingers) + obj_pose
        # except:
        #     print(len(self.default_finger_joints),' ',len(hand_pos),' ',len(hand_quat),' ',len(obj_pose),' ',len(self.objects))
        #     assert False
        self.d.ctrl = self.decode_finger_ctrl(hand_fingers)
        mujoco.mj_step(self.m, self.d)

        # self.static_view(1000)
        ini_contact_with_obj, ini_contact_with_floor = self.check_hand_contact()
        if ini_contact_with_obj or ini_contact_with_floor:
            # self.static_view(1000)
            return in_scope, False, ini_contact_with_obj, ini_contact_with_floor,None,None,None,warning_flag,grasped_obj
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++',self.default_finger_joints)
        delta=[0, 0, 0.003]
        decoded_fingers = self.close_grip(hand_fingers)
        # max_fingers = self.max_finger_ctrl()
        self.d.ctrl = decoded_fingers
        shake_amp = .003
        shake_f = 20  # Hz

        for i in range(600):
            #Rise phase
            if i==200:
                _, collide_with_floor = self.check_hand_contact()
                if collide_with_floor:
                    # self.static_view(1000)
                    return in_scope, False, ini_contact_with_obj, collide_with_floor, None, None, None, warning_flag, grasped_obj

            if 200 < i < 400:
                self.d.mocap_pos[0] = self.d.mocap_pos[0] + delta

            # shake phase
            if 500 > i > 400:
                if i==401:
                    grasp_success, n_grasp_contact1, self_collide1,max_force1,max_penetration1 = self.check_valid_grasp(minimum_contact_points=0)
                    if not grasp_success or not shake: break
                # self.d.ctrl = max_fingers #if i < 400 else decoded_fingers
                t = i * self.m.opt.timestep
                phase = 2 * np.pi * shake_f * t
                shake = shake_amp * np.array([np.sin(phase),
                                              np.sin(phase + 2.1),
                                              np.sin(phase + 4.2)])
                # shake = shake_amp * np.sin(2 * np.pi * shake_f * t)
                self.d.mocap_pos[0] += shake  # vertical shake (z)
            mujoco.mj_step(self.m, self.d)


            qpos = self.d.qpos
            qvel = self.d.qvel
            qacc=self.d.qacc
            MAX_MAG = 1e6
            bad = (
                    (not np.all(np.isfinite(qpos))) or
                    (not np.all(np.isfinite(qvel))) or
                    (not np.all(np.isfinite(qacc))) or
                    np.any(np.abs(qpos) > MAX_MAG) or
                    np.any(np.abs(qvel) > MAX_MAG)
            )
            if bad:
                warning_flag = True

        if grasp_success:
            stable_grasp,n_grasp_contact2,self_collide2,max_force2,max_penetration2 = self.check_valid_grasp(minimum_contact_points=0)
            # print(f'---test------------------------------{max_force1,max_penetration1,max_force2,max_penetration2}')
            grasped_obj = self.get_grasped_obj()
            if update_obj_prob and not warning_flag:

                # print(f'grasped_obj_: {grasped_obj}')

                # s=1.0 if stable_grasp else 0.9
                # if stable_grasp:print(Fore.GREEN,f"object {grasped_obj} grasped successfully",Fore.RESET)
                self.step_obj_prop(grasped_obj)

            return in_scope,grasp_success,ini_contact_with_obj, ini_contact_with_floor,min(n_grasp_contact1,n_grasp_contact2),self_collide1 or self_collide2,stable_grasp,warning_flag,grasped_obj
        else:
            return in_scope,grasp_success,ini_contact_with_obj, ini_contact_with_floor,n_grasp_contact1,self_collide1,None,warning_flag,grasped_obj

    def view_grasp(self,hand_pos,hand_quat,hand_fingers,obj_pose=None,view=False,iterations=300,hard_level=0.   ):
        self.restore_simulation_state()

        if obj_pose is None: obj_pose = self.objects_poses

        in_scope = max(hand_fingers) <= 1. and min(hand_fingers) >= 0.
        if not in_scope: hand_fingers = torch.clamp(torch.tensor(hand_fingers), min=0.01, max=0.99).tolist()
        grasped_obj = None

        warning_flag = False

        self.d.time = 0.0
        self.d.mocap_pos[0] = hand_pos
        self.d.mocap_quat[0] = hand_quat
        # try:
        self.d.qpos = hand_pos + hand_quat + self.decode_fingers_initial_state(hand_fingers) + obj_pose
        # except:
        #     print(len(self.default_finger_joints),' ',len(hand_pos),' ',len(hand_quat),' ',len(obj_pose),' ',len(self.objects))
        #     assert False
        self.d.ctrl = self.decode_finger_ctrl(hand_fingers)
        mujoco.mj_step(self.m, self.d)
        self.static_view(1000)

        ini_contact_with_obj, ini_contact_with_floor = self.check_hand_contact()
        # self.static_view(1000)

        delta=[0, 0, 0.003]
        decoded_fingers=self.close_grip(hand_fingers)
        # max_fingers=self.max_finger_ctrl()
        self.d.ctrl = decoded_fingers
        shake_amp = .003
        shake_f = 20  # Hz

        # video_path = next_video_name("CasiaHand_sim_clips", prefix="simulation")
        # writer = imageio.get_writer(video_path, fps=30)

        # Off-screen renderer for recording
        # renderer = mujoco.Renderer(self.m, width=640, height=480)

        # for i in range(70):
        #     mujoco.mj_step(self.m, self.d)
        #     if i==10:self.static_view(1000)



        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

            for i in range(70,600):
                step_start = time.time()


                if 200 < i < 400:
                    self.d.mocap_pos[0] = self.d.mocap_pos[0] + delta

                # shake phase
                if 500 > i > 400:
                    # self.d.ctrl = max_fingers #if i<400 else decoded_fingers

                    t = i * self.m.opt.timestep
                    phase = 2 * np.pi * shake_f * t
                    shake = shake_amp * np.array([
                        np.sin(phase),
                        np.sin(phase + 2.1),
                        np.sin(phase + 4.2)
                    ])
                    self.d.mocap_pos[0] += shake

                mujoco.mj_step(self.m, self.d)
                viewer.sync()

                # -------- capture frame from renderer --------
                # renderer.update_scene(self.d)
                # frame = renderer.render()  # RGB uint8
                # writer.append_data(frame)

                # maintain real-time speed
                dt = self.m.opt.timestep
                time.sleep(max(0, dt - (time.time() - step_start)))

        # writer.close()
        # print("Saved video to simulation.mp4")

                # Rudimentary time keeping, will drift relative to wall clock.
                # time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                # if time_until_next_step > 0:
                #     time.sleep(time_until_next_step)

        # After stepping
        # grasp_success = self.check_grasped_obj()
        grasp_success,n_grasp_contact,self_collide,max_force2,max_penetration2 = self.check_valid_grasp(minimum_contact_points=0,view=True)
        # if grasp_success:grasp_success= self.safety_fingers_check()
        # print(Fore.CYAN,f'final d.mocap_quat[0] {self.d.mocap_quat[0]}',Fore.RESET)
        # print(Fore.CYAN,f'final d.qpos[3:3+4] {self.d.qpos[3:3 + 4]}',Fore.RESET)

        grasped_obj=self.get_grasped_obj()
        print(f'grasped_obj: {grasped_obj}')

        if grasp_success:self.static_view(1000)

        return in_scope,grasp_success,ini_contact_with_obj, ini_contact_with_floor,n_grasp_contact,self_collide,None

if __name__ == "__main__":
    root_dir = os.getcwd()  # current working directory

    env=ShadowHandEnv(root=root_dir + "/GraspAgent_2/sim_dexee/hands_and_objects/")

    # env.view_geom_names_and_ids()

    # env.view_hand()
    env.drop_new_obj(selected_index='58', obj_pose=[0, 0.3, 0.2], obj_quat=[1, 0, 0, 0], stablize=True)
    env.view_geom_names_and_ids()

    print('test-------------------',env.last_hand_geom_id)

    # env.passive_viewer(pos=[0.0, 0.0, 0.2],quat=[.0, 1., 0., 0.],ctrl=None)
    depth, pointcloud, floor_mask = env.get_scene_preception()

    target_point = torch.tensor([.0, 0., 0.1]).cuda()
    target_pose = torch.tensor([0.,0.,-1,0,1,-0.5,0,0.5,1.]).cuda()

    from GraspAgent_2.training.SH_training import process_pose

    quat, fingers, shifted_point = process_pose(target_point, target_pose, view=True)

    env.manual_view(pos=shifted_point,quat=quat,fingers=None)

    # approach_ref = torch.tensor([0.0, 0., 1.0], device='cuda')
    # quat = quat_between(approach_ref, torch.tensor([0., 0., -1.], device='cuda')).cpu().tolist()
    print(quat)
    # quat=[1,0,0,0]
    # env.check_collision(hand_pos=shifted_point, hand_quat=quat,view=True)

    env.view_grasp(shifted_point,quat, hand_fingers=fingers)

