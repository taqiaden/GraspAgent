import time
from collections import deque

import mujoco.viewer
import torch
import trimesh
from mujoco.renderer import Renderer

import numpy as np
import xml.etree.ElementTree as ET
from random import sample
import os

from GraspAgent_2.hands_config.sh_config import fingers_max, fingers_min

hand_body_ids_range=[3,21]

def batch_quat_mul(q1, q2):
    """
    Multiply two batches of quaternions q1 * q2 element-wise.

    Args:
        q1: Tensor of shape [n, 4], quaternions [w, x, y, z]
        q2: Tensor of shape [n, 4], quaternions [w, x, y, z]

    Returns:
        Tensor of shape [n, 4], resulting quaternions.
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=1)
def quat_from_z_to_vec_single(v):
    """
    Compute quaternion that rotates [0,0,1] to vector v.
    Input: v -> torch tensor of shape [3]
    Output: q -> torch tensor [4] (w, x, y, z)
    """
    v = v / (v.norm() + 1e-8)
    z = torch.tensor([0.0, 0.0, 1.0], device=v.device)

    # Cross and dot products
    axis = torch.cross(z, v)
    dot = torch.dot(z, v)

    if dot > 0.999999:  # almost same direction
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=v.device)  # identity quaternion
    elif dot < -0.999999:  # opposite direction (180°)
        # Rotate around any perpendicular axis, e.g. x-axis
        return torch.tensor([0.0, 1.0, 0.0, 0.0], device=v.device)

    w = torch.sqrt((1.0 + dot) / 2.0)
    xyz = axis / (torch.norm(axis) + 1e-8) * torch.sqrt((1.0 - dot) / 2.0)
    q = torch.cat([w.view(1), xyz])
    return q / q.norm()
def random_quaternion():
    """
    Generate a random unit quaternion (w, x, y, z).
    """
    u1 = np.random.rand()
    u2 = np.random.rand()
    u3 = np.random.rand()

    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])
    return q  # (x, y, z, w) convention


def quat_between(v_from, v_to):
    v_from = v_from / torch.norm(v_from)
    v_to = v_to / torch.norm(v_to)
    cross = torch.cross(v_from, v_to)
    dot = torch.dot(v_from, v_to)
    w = torch.sqrt((torch.norm(v_from)**2) * (torch.norm(v_to)**2)) + dot
    quat = torch.cat([torch.tensor([w]), cross])
    return quat / torch.norm(quat)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def draw_point(viewer, pos, radius=0.02, rgba=[1, 0, 0, 1]):
    """Draw a sphere at the specified position"""
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=rgba
    )
    viewer.user_scn.ngeom = 1

def quat_rotate_vector(q, v):
    """
    Rotate 3D vector v by quaternion q.
    q: [w, x, y, z]
    v: [x, y, z]
    """
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    w, x, y, z = q
    q_vec = np.array([x, y, z])

    # Cross products for quaternion rotation
    t = 2.0 * np.cross(q_vec, v)
    v_rot = v + w * t + np.cross(q_vec, t)

    return v_rot

class grasp_env():
    def __init__(self,obj_nums_in_scene=3,root = "shadow_dexee/" ,selected_idx=None,max_obj_per_scene=10):
        super().__init__()
        '''
        geom id 0 : floor
        geom id between and include 1, and 101 : for hand
        body id 0 : floor
        body id between and include 3, and 21 : for hand

        '''

        self.root=root
        self.objects_path = root+"/mesh/"
        self.object_nums_all = len(os.listdir(self.objects_path))
        self.obj_nums_in_scene = obj_nums_in_scene
        assert obj_nums_in_scene <= self.object_nums_all, f'{self.object_nums_all}'
        self.m=None
        self.d=None

        self.idx=self.sample_random_obj() if selected_idx is None else selected_idx
        self.prepare_obj_mesh(self.idx)

        self.initiate_mojoco()

        '''camera info'''
        self.height = 600
        self.width = 600
        self.camera_id = None
        self.renderer = None
        self.intr=None
        self.extr=None

        self.ini_renderer()

        # print('Current Solver ID: ',self.m.opt.solver)


        # self.view_geom_names_and_ids()
        # self.verify_mass_properties()

        self.f0_sensors_ids=[32,27,38]
        self.f1_sensors_ids=[64,59,70]
        self.f2_sensors_ids=[96,91,102]

        '''sensors are under these names'''
        # r3_finger_middle_magtac
            #F0/middle_magtac_geom
        # r3_finger_proximal_magtac
            #F1/proximal_magtac_geom
        # r3_finger_distal_sensor
            #F1/distal_sensor_geom

        self.far_hand_pos = [10, 10., 10.]
        self.far_hand_quat = [0, 1, 0, 0]
        self.far_finger_joints = [0, -1.4, 0, 0, 0, -1.4, 0, 0, 0, -1.4, 0, 0]

        self.objects = deque([])
        self.objects_poses = []
        self.max_obj_per_scene=max_obj_per_scene


    def drop_new_obj(self):
        for j in range(1000):
            new_obj_id=self.sample_random_obj()[0]
            if new_obj_id not in self.objects: break
        else: assert False

        print('Newly added object ID: ',new_obj_id)

        self.objects.append(new_obj_id)

        obj_pose = [(np.random.rand() - 0.5)*0.3, (np.random.rand() - 0.5)*0.3, 0.3]

        obj_quat = torch.randn((4,))
        obj_quat[[1, 2]] *= 0
        obj_quat = obj_quat / torch.norm(obj_quat)
        obj_quat = obj_quat.tolist()

        self.objects_poses+=obj_pose+obj_quat

        if len(self.objects)>self.max_obj_per_scene:
            print('Drop object ID: ',self.objects[0])
            self.objects.popleft()
            self.objects_poses=self.objects_poses[7:]


        self.prepare_obj_mesh(self.objects)

        self.initiate_mojoco()
        self.camera_id = None
        self.renderer = None
        self.intr=None
        self.extr=None
        self.ini_renderer()

        self.objects_poses=self.get_stable_object_pose(self.objects_poses).tolist()


    def set_new_scene(self):
        self.m=None
        self.d=None
        self.idx = self.sample_random_obj() #if selected_idx is None else selected_idx

        print('object index :',self.idx)

        self.prepare_obj_mesh(self.idx)

        self.initiate_mojoco()
        self.camera_id = None
        self.renderer = None
        self.intr=None
        self.extr=None
        self.ini_renderer()

    def view_geom_names_and_ids(self):
        for geom_id in range(self.m.ngeom):
            # print(geom_id,'----',mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
            print(self.get_geom_body_info(geom_id))

    def get_geom_body_info(self, geom_id):
        """Get comprehensive information about geom and its parent body"""
        body_id = self.m.geom_bodyid[geom_id]

        geom_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, body_id)

        return {
            'geom_id': geom_id,
            'geom_name': geom_name,
            'body_id': body_id,
            'body_name': body_name,
            'is_mocap': self.m.body_mocapid[body_id] >= 0
        }

    def verify_mass_properties(self):
        """
        Check mass properties of all bodies
        """
        print("Body Mass Properties:")
        print("-" * 50)

        for i in range(self.m.nbody):
            body_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i)
            mass = self.m.body_mass[i]
            inertia = self.m.body_inertia[i]

            print(f"Body {i} ({body_name}):")
            print(f"  Mass: {mass:.4f} kg")
            print(f"  Inertia: [{inertia[0]:.6f}, {inertia[1]:.6f}, {inertia[2]:.6f}]")
            print()

    def sample_random_obj(self):
        idxs = sample(range(self.object_nums_all), self.obj_nums_in_scene)
        return idxs

    def prepare_obj_mesh(self,idxs):
        tree = ET.parse(self.root+'/scene.xml')
        root = tree.getroot()
        for idx in idxs:
            new_mesh = ET.Element('include')
            new_mesh.set('file', 'mesh/mesh_' + str(idx) + '.xml')
            root.insert(1, new_mesh)
        tree.write(self.root+'/temp.xml')

    # def prepare_obj_mesh(self, idxs):
    #     tree = ET.parse(self.root + '/scene.xml')
    #     root = tree.getroot()
    #
    #     # --- Add/modify arena memory ---
    #     # Check if <size> already exists
    #     size_elem = root.find('size')
    #     if size_elem is None:
    #         size_elem = ET.Element('size')
    #         root.insert(0, size_elem)  # insert at the top
    #     size_elem.set('memory', '64M')  # set memory to 64M
    #
    #     # --- Add the meshes ---
    #     for idx in idxs:
    #         new_mesh = ET.Element('include')
    #         new_mesh.set('file', 'mesh/mesh_' + str(idx) + '.xml')
    #         root.insert(1, new_mesh)  # insert after <size>
    #
    #     tree.write(self.root + '/temp.xml')


    def initiate_mojoco(self):
        self.m = mujoco.MjModel.from_xml_path(self.root+'/temp.xml')

        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)

    def ini_renderer(self):
        # Define camera parameters and init renderer.
        self.camera_id = self.m.cam("camera_1").id
        self.renderer = Renderer(self.m, height=self.height, width=self.width)
        self.intr=self.get_camera_intrinsic()
        self.extr=self.get_camera_extrinsic()

    def get_camera_intrinsic(self):
        # Intrinsic matrix.
        fov = self.m.cam_fovy[self.camera_id]
        theta = np.deg2rad(fov)
        fx = self.width / 2 / np.tan(theta / 2)
        fy = self.height / 2 / np.tan(theta / 2)
        cx = (self.width - 1) / 2.0
        cy = (self.height - 1) / 2.0
        intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intr

    def get_camera_extrinsic(self):
        cam_pos = self.d.cam_xpos[self.camera_id]
        cam_rot = self.d.cam_xmat[self.camera_id].reshape(3, 3)
        extr = np.eye(4)
        extr[:3, :3] = -cam_rot
        extr[:3, 3] = cam_pos
        return extr

    def render_depth(self,
            renderer: mujoco.Renderer,
            camera_id: int,
    ) -> np.ndarray:
        renderer.update_scene(self.d, camera=camera_id)
        renderer.enable_depth_rendering()
        depth = renderer.render()
        return depth

    def depth_to_pointcloud(self,
            depth: np.ndarray,
            intr: np.ndarray,
            extr: np.ndarray,
            depth_trunc: float = 20.0,
    ) -> np.ndarray:
        cc, rr = np.meshgrid(np.arange(self.width), np.arange(self.height), sparse=True)
        valid = (depth > 0) & (depth < depth_trunc)
        z = np.where(valid, depth, np.nan)
        x = np.where(valid, z * (cc - intr[0, 2]) / intr[0, 0], 0)
        y = np.where(valid, z * (rr - intr[1, 2]) / intr[1, 1], 0)
        xyz = np.vstack([e.flatten() for e in [x, y, z]]).T
        mask = np.isnan(xyz[:, 2])
        floor_mask = np.abs(xyz[:, 2] - 1.3) < 1e-6

        xyz = xyz[~mask]
        xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        xyz_t = (extr @ xyz_h.T).T
        return xyz_t[:, :3],floor_mask

    def generate_random_obj_pose_(self):
        quat = random_quaternion()
        position = np.random.uniform(-.2, 0.2, size=3)

        return quat.tolist() + position.tolist()

    def generate_random_obj_poses(self):
        obj_poses = []
        for i in range(self.obj_nums_in_scene):
            obj_poses+=self.generate_random_obj_pose_()
        return obj_poses

    def set_configuration(self,hand_position,hand_quat,finger_joints):
        # d.mocap_pos shape=[1, 3]
        # d.mocap_quat shape=[1, 4] wxyz
        self.d.mocap_pos[0] = hand_position
        self.d.mocap_quat[0] = hand_quat

        # d.qpos shape=7+12+7*obj_nums_in_scene, first 7 for gripper_base, next 12 for 12 finger joints, then each mesh has 7, 3 for pos and 4 for quat(wxyz)
        # set initial qpos
        # self.d.qpos = [0, 0.5, 0.35, 0, 1, 0, 0, 0, -0.8, 0, 0, 0, -0.8, 0, 0, 0, -0.8, 0, 0, 0.2, 0, -0.07, 1, 0, 0, 0, 0,
        #           0, -0.07, 1, 0, 0, 0, -0.2, 0, -0.07, 1, 0, 0, 0]

        self.d.qpos=hand_position+hand_quat+finger_joints+self.generate_random_obj_poses()



    def check_hand_contact(self,margin=0,report=False):
        is_hand_geom= lambda x: x>=1 and x<=105
        contact_with_floor=False
        contact_with_obj=False
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.dist < margin and (is_hand_geom(c.geom1) + is_hand_geom(c.geom2) ==1) :
                # if is_hand_geom(c.geom1) and is_hand_geom(c.geom2): return False,False,True # fingers collision
                if report:
                    geom1_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                    geom2_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                    print(f"⚠️ Interference between geom {geom1_name} and geom {geom2_name}, depth = {c.dist:.6f}")
                    print(is_hand_geom(c.geom1) + is_hand_geom(c.geom2) ,'---- ',is_hand_geom(c.geom1) , is_hand_geom(c.geom2))
                    print(c.geom1,'-----',c.geom2)
                if c.geom1==0 or c.geom2==0:contact_with_floor=True
                else: contact_with_obj=True
                # print(f"⚠️ Interference between geom {c.geom1} and geom {c.geom2}, depth = {c.dist:.6f}")

        return contact_with_obj,contact_with_floor

    def check_grasped_obj(self,margin=0):
        is_hand_geom = lambda x: x >= 1 and x <= 105
        sensors_ids=self.f0_sensors_ids+self.f1_sensors_ids+self.f2_sensors_ids

        f0_contact = 0
        f1_contact=0
        f2_contact=0
        sensor_contact_ids=[]
        contacts = []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if  (is_hand_geom(c.geom1) + is_hand_geom(c.geom2) == 1):

                if c.geom1 == 0 or c.geom2 == 0:
                    continue
                else:

                    print('-----------------',c.geom1,c.geom2, 'distance= ',c.dist)
                    if c.geom1 in sensors_ids: sensor_contact_ids.append(c.geom1)
                    if c.geom2 in sensors_ids: sensor_contact_ids.append(c.geom2)

        print('------------------sensor_contact_ids: ',sensor_contact_ids)
        for id in sensor_contact_ids:
            if id in self.f0_sensors_ids: f0_contact=1
            if id in self.f1_sensors_ids: f1_contact=1
            if id in self.f2_sensors_ids: f2_contact=1

        return f0_contact+f1_contact+f2_contact>=2

    def static_view(self,period=5):
        # print('mocap_pos=', self.d.mocap_pos[0])
        # print('mocap_quat=', self.d.mocap_quat[0])
        # print('qpos=', self.d.qpos)
        timer=0
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            # Update loop — this keeps the window open until the user closes it
            while viewer.is_running():
                viewer.opt.flags[
                mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1  # shows the contact points in the simulation (where objects touch)
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 0  # shows camera frustums if enabled
                # Render continuously
                viewer.sync()
                time.sleep(1)  # Prevent CPU overuse
                timer+=1
                if timer>period: return

    def check_fingers_scope(self,fingers):
        fingers=np.array(fingers)
        result=True
        for i in range(3):
            b=(fingers[i*4:i*4+3]<=fingers_max[0:3]) & (fingers[i*4:i*4+3]>=fingers_min[0:3])
            # print(b)
            result= np.all(b) & result
        return  result

    def check_collision(self,hand_pos,hand_quat,hand_fingers,view=False):

        fingers_min_ = torch.from_numpy(fingers_min).repeat(3)
        fingers_max_ = torch.from_numpy(fingers_max).repeat(3)
        hand_fingers = torch.clamp(torch.tensor(hand_fingers), min=fingers_min_ + 0.01,
                                   max=fingers_max_ - 0.01).tolist()

        self.d.mocap_pos[0] = hand_pos
        self.d.mocap_quat[0] = hand_quat

        self.d.qpos =hand_pos + hand_quat + hand_fingers + self.objects_poses
        # print(self.d.time)

        mujoco.mj_step(self.m, self.d)

        '''check initial contact'''
        contact_with_obj, contact_with_floor = self.check_hand_contact()
        # print(f'Initial contact result with obj = {contact_with_obj}, with floor = {contact_with_floor}')

        if view:
            self.static_view()

        return  contact_with_obj or contact_with_floor

    def check_graspness(self,hand_pos,hand_quat,hand_fingers,obj_pose=None,view=False,iterations=50,shake_intensity=0.05):

        if obj_pose is None: obj_pose=self.objects_poses

        in_scope = self.check_fingers_scope(hand_fingers)

        fingers_min_ = torch.from_numpy(fingers_min).repeat( 3)
        fingers_max_ = torch.from_numpy(fingers_max).repeat( 3)
        hand_fingers = torch.clamp(torch.tensor(hand_fingers),min=fingers_min_+0.01,max=fingers_max_-0.01).tolist()

        # if not in_scope: return False, None, None,None,None

        self.d.time = 0.0
        self.d.mocap_pos[0] = hand_pos
        self.d.mocap_quat[0] = hand_quat

        self.d.qpos = hand_pos + hand_quat + hand_fingers + obj_pose

        delta=quat_rotate_vector(np.array(hand_quat),np.array([0,0,1]))
        if delta[-1]>0:in_scope=False

        delta*=-0.02
        # self.d.ctrl = [0, -0.4, 0, 0, 0, -0.4, 0, 0, 0, -0.4, 0, 0]
        a=0.
        b=1.
        c=0.2
        self.d.ctrl = [a, b, c,  a, b, c, a, b, c]

        # if view:
        #     # Run with viewer
        #     mujoco.mj_step(self.m, self.d)
        #     # self.static_view()
        #     with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
        #
        #         viewer.sync()
        #         contact_with_obj, contact_with_floor = self.check_hand_contact()
        #         if contact_with_obj or contact_with_floor:
        #             return in_scope, None, contact_with_obj, contact_with_floor
        #
        #         for i in range(iterations):
        #             if i>10:
        #                 if i<30:
        #                     self.d.mocap_pos[0] = self.d.mocap_pos[0] + delta.tolist()
        #                 elif i==30:
        #                     final_contact_with_obj, contact_with_floor = self.check_hand_contact()
        #                     if not final_contact_with_obj: break
        #                 else:
        #                     if shake_intensity is None: break
        #                     if i%6==0:
        #                         self.d.mocap_pos[0] = self.d.mocap_pos[0]+[shake_intensity,0,0]
        #                     elif i % 6 == 1:
        #                         self.d.mocap_pos[0] = self.d.mocap_pos[0] + [-shake_intensity, 0, 0]
        #                     elif i % 6 == 2:
        #                         self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, shake_intensity, 0]
        #                     elif i % 6 == 3:
        #                         self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, -shake_intensity, 0]
        #                     elif i % 6 == 4:
        #                         self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, 0,shake_intensity]
        #                     elif i % 6 == 5:
        #                         self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, 0, -shake_intensity]
        #
        #
        #             for _ in range(20):
        #                 mujoco.mj_step(self.m, self.d)
        #
        #             viewer.sync()
        #     self.static_view(1000)
        # else:
        # Run headless (no viewer)
        mujoco.mj_step(self.m, self.d)
        contact_with_obj, contact_with_floor = self.check_hand_contact()
        if contact_with_obj or contact_with_floor:
            return in_scope, None, contact_with_obj, contact_with_floor
        for i in range(iterations):
            if i > 10:
                if i < 30:
                    self.d.mocap_pos[0] = self.d.mocap_pos[0] + delta.tolist()
                elif i == 30:
                    final_contact_with_obj, contact_with_floor = self.check_hand_contact()
                    if not final_contact_with_obj: break
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

            for _ in range(20):
                mujoco.mj_step(self.m, self.d)
        # After stepping
        # grasp_success = self.check_grasped_obj()
        grasp_success, contact_with_floor = self.check_hand_contact()

        if view:self.static_view(1000)

        return in_scope,grasp_success,False,False



    def visualize(self,obj_pos_quat,duration=300):
        self.d.mocap_pos[0] = self.far_hand_pos
        self.d.mocap_quat[0] = self.far_hand_quat

        self.d.qpos = self.far_hand_pos + self.far_hand_quat + self.far_finger_joints + obj_pos_quat.tolist()

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:

            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1 # shows the contact points in the simulation (where objects touch)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 0 # shows camera frustums if enabled

            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()

            steps_counter=0
            while viewer.is_running() and time.time() - start < duration:

                if steps_counter==0:
                    mujoco.mj_step(self.m, self.d)
                    '''get preception'''
                    depth = self.render_depth(self.renderer, self.camera_id)
                    from lib.image_utils import view_image
                    view_image(depth)
                    pointcloud ,floor_mask= self.depth_to_pointcloud(depth, self.intr, self.extr)
                    floor_mask=pointcloud[:,2]<0.01
                    print(pointcloud[:,2].max())
                    print(pointcloud[:,2].min())
                    pc = trimesh.PointCloud(pointcloud)
                    pc.show()
                    '''set to target pose'''
                    hand_pos=[0, 0, 0.25]
                    hand_quat=[0, 1, 0, 0]
                    hand_fing=self.far_finger_joints

                    self.d.mocap_pos[0] = hand_pos
                    self.d.mocap_quat[0] = hand_quat

                    self.d.qpos[:7 + 12] = hand_pos + hand_quat + hand_fing

                    # '''check initial contact'''
                    # contact_with_obj, contact_with_floor = self.check_hand_contact()
                    # print(f'Initial contact result with obj = {contact_with_obj}, with floor = {contact_with_floor}')

                #set pos and quat of gripper base
                else:
                    if self.d.time > 0:
                        self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, 0, 0.001]
                    self.d.mocap_quat[0] = [0, 1, 0, 0]

                    # set control of finger joints
                    self.d.ctrl = [0, -0.4, 0, 0, 0, -0.4, 0, 0, 0, -0.4, 0, 0]

                    for _ in range(20): # physics steps per viewer iteration.
                        mujoco.mj_step(self.m, self.d)

                    viewer.sync()
                steps_counter += 1

    def get_scene_preception(self,obj_pos_quat=None,view=False):
        if obj_pos_quat is None: obj_pos_quat=self.objects_poses

        self.d.mocap_pos[0] = self.far_hand_pos
        self.d.mocap_quat[0] = self.far_hand_quat

        self.d.qpos = self.far_hand_pos + self.far_hand_quat + self.far_finger_joints + obj_pos_quat#.tolist()
        # mujoco.mj_step(self.m, self.d)

        depth = self.render_depth(self.renderer, self.camera_id)
        pointcloud ,floor_mask= self.depth_to_pointcloud(depth, self.intr, self.extr)
        pointcloud[:, 0] *= -1
        # if pointcloud.shape[0]!=600*600:
        #     from lib.image_utils import view_image
        #     view_image(depth)
        #     self.static_view()
        assert pointcloud.shape[0]==600*600, f'{pointcloud.shape}'

        if view:
            from lib.image_utils import view_image
            view_image(depth)
            pc = trimesh.PointCloud(pointcloud)
            pc.show()

        return depth,pointcloud,floor_mask

    def get_stable_object_pose(self,obj_pos_quat,threshold=1e-4,window_size=30):
        self.d.mocap_pos[0] = self.far_hand_pos
        self.d.mocap_quat[0] = self.far_hand_quat

        self.d.qpos= self.far_hand_pos+ self.far_hand_quat+self.far_finger_joints+obj_pos_quat

        self.m.opt.timestep = 0.01
        last_obj_pos=None
        for i in range(10000):
            # print(self.d.qpos[7 + 12:])
            for j in range(window_size):
                mujoco.mj_step(self.m, self.d)
            new_obj_pos=np.copy(self.d.qpos[7 + 12:])
            if last_obj_pos is not None:
                diff=np.sum(last_obj_pos-new_obj_pos)
                if diff<threshold:
                    # print(self.d.qpos[7 + 12:])
                    break

            last_obj_pos=new_obj_pos
        else:
            print('Cannot find a stable pose')
            return None

        self.m.opt.timestep = 0.002

        return  self.d.qpos[7 + 12:]


    def simulate_headless(self, duration=300.0, sim_rate=20):
        """
        Run the MuJoCo simulation without any viewer (headless mode).

        Args:
            duration (float): Wall-clock duration (seconds) to run the simulation.
            sim_rate (int): Physics steps per outer iteration (similar to viewer loop).
        """

        start = time.time()
        while time.time() - start < duration:
            # Move mocap body (e.g., gripper base)
            if self.d.time > 0:
                self.d.mocap_pos[0] = self.d.mocap_pos[0] + [0, 0, 0.01]
            self.d.mocap_quat[0] = [0, 1, 0, 0]

            # Apply control inputs to actuators (finger joints etc.)
            self.d.ctrl = [0, -0.4, 0, 0,  -0.4, 0, 0,  -0.4, 0]

            # Step physics forward
            for _ in range(sim_rate):
                mujoco.mj_step(self.m, self.d)

            # Optionally capture depth or images offscreen
            depth = self.render_depth(self.renderer, self.camera_id)
            # e.g., process depth → point cloud here if needed
            # pointcloud = depth_to_pointcloud(depth, intr, extr)\

    def manual_view(self):
        pos=[0., 0., .3]
        # v1 = torch.tensor([0., 0., 1.])
        # approach = torch.tensor([0.5, 0., 0.7])  # example direction
        # approach = approach / torch.norm(approach)
        q1 = torch.tensor([0., 1., 0., 0.])  # 180° around X
        # beta_quat=torch.tensor([0.7, .0, 0., 0.7])
        # beta_quat = beta_quat / torch.norm(beta_quat)
        # approach_quat = quat_between(v1, approach)
        # approach_quat = approach_quat / torch.norm(approach_quat)
        #
        #
        # q_r=quat_mul(beta_quat, q1)
        # # print('q_r: ',q_r)
        # q = quat_mul(approach_quat, q_r)
        # q = q / torch.norm(q)
        # quat=q.tolist()
        # # q1=quat_from_z_to_vec_single(torch.tensor([0,0,9]).float()).numpy()
        # quat=np.array([1.0,0,0,0]).tolist()
        # quat=  quat_mul(q2,q1).tolist()
        # first_quat = torch.tensor([[0., 1., 0., 0.]])
        # size=1000
        # beta_quat = torch.zeros((size, 4))
        # beta_quat[:, [0, 3]] = torch.randn((size, 2))
        # import torch.nn.functional as F

        # beta_quat = F.normalize(beta_quat, dim=-1)
        # batch_quat = batch_quat_mul(beta_quat, first_quat)
        def get_beta_quat():
            quat=torch.randn((4,))
            quat[1:3]*=0
            # quat=torch.tensor([0.7, .0, 0., 0.7])
            quat = quat / torch.norm(quat)
            quat = quat_mul(quat, q1)
            quat = quat / torch.norm(quat)

            quat=quat.tolist()
            return quat
        quat=get_beta_quat()

        self.d.mocap_pos[0] = pos
        self.d.mocap_quat[0] = quat

        obj_quat=torch.tensor([0.5, 0.,0., 0.])
        obj_quat=obj_quat / torch.norm(obj_quat)


        self.d.qpos = pos + quat + self.far_finger_joints + [0.3, 0.3, 0.]+obj_quat.tolist()

        # a = 0.
        # b = 0.7
        # c = 0.

        # self.d.ctrl = [a, b, c, a, b, c, a, b, c]
        counter=0
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD  # show world frame
            # optional: show contact points
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

            # mujoco.mj_step(self.m, self.d)

            # depth, pc, floor_mask = self.get_scene_preception(np.array([0.3, 0.3, 0.]+[1, 0, 0, 0]), view=False)
            # obj_pc=pc[~floor_mask]
            # print(obj_pc.mean(axis=0))
            # print(obj_pc.max(axis=0))
            # print(obj_pc.min(axis=0))

            # pc_ = trimesh.PointCloud(pc[~floor_mask])
            # pc_.show()
            while viewer.is_running():
                # quat=batch_quat[counter].tolist()
                # self.d.mocap_quat[0] =quat
                # self.d.qpos = pos + quat + self.far_finger_joints + [0.3, 0.3, 0.] + obj_quat.tolist()

                mujoco.mj_step(self.m, self.d)
                counter+=1
                # print(self.d.ctrl)
                # self.d.ctrl[0]=self.d.qpos[3+4]
                # self.d.ctrl[1]=self.d.qpos[3+4+1]
                # self.d.ctrl[2]=self.d.qpos[3+4+2]

                viewer.sync()
                # time.sleep(1)


if __name__ == "__main__":
    print(mujoco.__version__)

    shadow_hand_env=grasp_env(obj_nums_in_scene=1,selected_idx=[6])
    obj_pose=[0., 0., 0.]
    obj_quat=[1, 0, 0, 0]
    finger_joints=[0, -0.8, 0, 0, 0, -0.8, 0, 0, 0, -0.8, 0, 0]
    stable_obj_pose_quat=shadow_hand_env.get_stable_object_pose(obj_pose+obj_quat)
    # print(stable_obj_pose_quat)

    hand_pos = [0,0,0]
    hand_quat = np.array([-1,1,0,0])
    hand_quat = hand_quat / np.linalg.norm(hand_quat)
    hand_fing = shadow_hand_env.far_finger_joints
    hand_fing=[-0.6147308349609375, -1.2909823656082153, 0.053371675312519073, -0.08044019341468811, -0.24372847378253937, -0.7953169941902161, 0.13495197892189026, 0.14904902875423431, -0.3114812672138214, -0.6036136150360107, 0.26515403389930725, -0.13743282854557037]
    # r=shadow_hand_env.check_graspness(hand_pos=hand_pos,hand_quat=hand_quat.tolist(),hand_fingers=hand_fing,obj_pose=stable_obj_pose_quat.tolist(),view=True)
    # print(r)
    # shadow_hand_env.visualize(stable_obj_pose_quat)
    shadow_hand_env.manual_view()
