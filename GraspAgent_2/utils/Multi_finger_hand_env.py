import json
import time
from abc import abstractmethod
from collections import deque

import mujoco.viewer
import torch
import trimesh
from colorama import Fore
from mujoco.renderer import Renderer

import numpy as np
import xml.etree.ElementTree as ET
from random import sample
import random

import os

from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from GraspAgent_2.hands_config.sh_config import fingers_max, fingers_min
from GraspAgent_2.utils.quat_operations import random_quaternion, quat_mul, quat_rotate_vector


def find_group(num, groups):
    for i, group in enumerate(groups):
        if num in group:
            return i
    return None  # not found

def farthest_point_from_points(points, xmin, xmax, ymin, ymax, resolution=200):
    pts = np.array(points).reshape(-1, 2)  # shape: (N,2)

    # Create grid in box
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (res^2, 2)

    # Compute distance to each point and take the minimum
    dists = np.linalg.norm(grid[:, None, :] - pts[None, :, :], axis=2)
    min_dist = dists.min(axis=1)  # closest object distance

    # Pick the grid point whose closest-object distance is maximal
    idx = np.argmax(min_dist)
    return grid[idx]

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



class MojocoMultiFingersEnv():
    def __init__(self,obj_nums_in_scene=3,root = "shadow_dexee/" ,max_obj_per_scene=2):
        super().__init__()
        '''


        '''

        self.root=root
        self.objects_path = root+"/mesh/"
        self.object_nums_all = len(os.listdir(self.objects_path))
        self.obj_nums_in_scene = obj_nums_in_scene
        assert obj_nums_in_scene <= self.object_nums_all, f'{self.object_nums_all}'
        self.m=None
        self.d=None

        self.prepare_obj_mesh()
        self.initiate_mojoco()

        '''camera info'''
        self.height = 600
        self.width = 600
        self.camera_id = None
        self.renderer = None
        self.intr=None
        self.extr=None

        # self.ini_renderer()


        self.far_hand_pos = [10, 10., 10.]
        self.far_hand_quat = [0, 1, 0, 0]
        self.default_finger_joints = None

        self.objects = deque([])
        self.objects_poses = []
        self.max_obj_per_scene=max_obj_per_scene

        self.last_hand_geom_id=None


        self.dict_file_path=self.root+'/obj_data.json'
        self.obj_dict= self.load_obj_dict()

        self.saved_state = None

        self.contact_pads_geom_ids =None





        assert len(self.obj_dict)<=self.object_nums_all, f'{len(self.obj_dict)}, {self.object_nums_all}'

    def update_obj_info(self,score):
        for obj in self.objects:
            if str(obj) in self.obj_dict:
                self.obj_dict[str(obj)]=max(0.9*self.obj_dict[str(obj)]+.1*score,0.01)
                # print(f'test -----------------------------------{len(self.obj_dict)}---------------------,{obj}, {self.obj_dict[obj]}')
                # if not len(self.obj_dict) <= self.object_nums_all:
                #     for key, value in self.obj_dict.items():
                #         print(f"Key: {key}, type: {type(key)}, Value: {value}, type: {type(value)}")
                assert len(self.obj_dict) <= self.object_nums_all, f'{len(self.obj_dict)}, {self.object_nums_all}, {obj}'
            else:
                self.obj_dict[str(obj)]=1


    def load_obj_dict(self):
        if os.path.exists(self.dict_file_path):
            with open(self.dict_file_path, "r") as f:
                data = json.load(f)
            return data if data is not None else {}
        else:
            return {}

    def save_obj_dict(self):
        with open(self.dict_file_path, "w") as f:
            json.dump(self.obj_dict, f, indent=4)


    @property
    def obj_positions(self):
        return [self.objects_poses [i:i+3] for i in range(0, len(self.objects_poses ), 7)]

    @property
    def obj_xy_positions(self):
        return [self.objects_poses [i:i+2] for i in range(0, len(self.objects_poses ), 7)]

    def drop_new_obj(self,selected_index=None,obj_pose=None,stablize=True):
        while True:
            if selected_index is None:
                for j in range(1000):
                    new_obj_id=self.sample_random_obj()
                    if new_obj_id not in self.objects: break
                else: assert False
            else: new_obj_id=selected_index

            print('Newly added object ID: ',new_obj_id)

            self.objects.append(new_obj_id)

            if obj_pose is None:
                if len(self.objects)>1:
                    xy_farthest=farthest_point_from_points(self.obj_xy_positions, xmin=-.3, xmax=0.3, ymin=-0.3, ymax=0.3, resolution=200)
                    obj_pose = [xy_farthest[0], xy_farthest[1], 0.2]
                else:
                    obj_pose = [(np.random.rand() - 0.5)*0.4, (np.random.rand() - 0.5)*0.4, 0.2]

            r=np.random.random()
            if r<0.3:
                obj_quat = torch.randn((4,))
                obj_quat[[1, 2]] *= 0
                obj_quat = obj_quat / torch.norm(obj_quat)
                obj_quat = obj_quat.tolist()
                # print(obj_quat)
            elif r>0.7:
                obj_quat = torch.randn((4,))
                obj_quat = obj_quat / torch.norm(obj_quat)
                obj_quat = obj_quat.tolist()
            else:
                basis_quats = torch.tensor([
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]
                ])  # shape [4,4]

                # Select one at random
                idx = torch.randint(0, 4, (1,))
                obj_quat = basis_quats[idx][0].tolist()
                # print(obj_quat)

            self.objects_poses=obj_pose+obj_quat+self.objects_poses # new object at the begining

            if len(self.objects)>self.max_obj_per_scene:
                print('Remove object ID: ',self.objects[0])
                self.objects.popleft()
                self.objects_poses=self.objects_poses[:-7]

            self.prepare_obj_mesh(self.objects)

            self.initialize()

            new_poses=self.get_stable_object_pose(self.objects_poses,stablize=stablize)
            if new_poses is not None:
                self.objects_poses=new_poses.tolist()
                break
            else:
                self.remove_all_objects()

        self.save_simulation_state()

    def remove_obj(self):
        if len(self.objects) >=1 :
            print('Remove object ID: ', self.objects[0])
            self.objects.popleft()
            self.objects_poses = self.objects_poses[:-7]
    def initialize(self):
        self.initiate_mojoco()
        self.camera_id = None
        self.renderer = None
        self.intr = None
        self.extr = None
        self.ini_renderer()

    def save_simulation_state(self):
        """Save complete simulation state"""
        self.saved_state ={
            'qpos': self.d.qpos.copy(),
            'qvel': self.d.qvel.copy(),
            'act': self.d.act.copy(),
            'ctrl': self.d.ctrl.copy(),
            'time': self.d.time,
            'mocap_pos': self.d.mocap_pos.copy(),
            'mocap_quat': self.d.mocap_quat.copy(),
            'sensordata': self.d.sensordata.copy(),
            'userdata': self.d.userdata.copy() if hasattr(self.d, 'userdata') else None
        }

    def restore_simulation_state(self ):
        """Restore to saved state"""
        if self.saved_state is not None:
            self.d.qpos[:] = self.saved_state['qpos']
            self.d.qvel[:] = self.saved_state['qvel']
            self.d.act[:] = self.saved_state['act']
            self.d.ctrl[:] = self.saved_state['ctrl']
            self.d.time = self.saved_state['time']
            self.d.mocap_pos[:] = self.saved_state['mocap_pos']
            self.d.mocap_quat[:] = self.saved_state['mocap_quat']
            self.d.sensordata[:] = self.saved_state['sensordata']
            if self.saved_state['userdata'] is not None and hasattr(self.d, 'userdata'):
                self.d.userdata[:] = self.saved_state['userdata']
        else:
            raise ValueError("No state saved. Call save_current_state() first.")

        mujoco.mj_step(self.m, self.d)

    def remove_all_objects(self):
        for i in range(len(self.objects)):
            print('Drop object ID: ', self.objects[0])
            self.objects.popleft()
            self.objects_poses = self.objects_poses[:-7]

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
        idxs = sample(range(self.object_nums_all), 1)[0]
        if len(self.obj_dict) == 0: return idxs
        if str(idxs) not in self.obj_dict: return idxs
        # print(self.obj_dict,'----',idxs)

        keys = list(self.obj_dict.keys())
        weights = list(self.obj_dict.values())
        idxs = random.choices(keys, weights=weights, k=1)[0]

        # while True:
        #     if idxs not in self.obj_dict: break
        #     p=self.obj_dict[idxs]
        #     if p>np.random.random(): break
        #     idxs = sample(range(self.object_nums_all), self.obj_nums_in_scene)

        return idxs

    def prepare_obj_mesh(self,idxs=None):
        tree = ET.parse(self.root+'/scene.xml')
        root = tree.getroot()
        if idxs is not None:
            for idx in idxs:
                new_mesh = ET.Element('include')
                new_mesh.set('file', 'mesh/mesh_' + str(idx) + '.xml')
                root.insert(1, new_mesh)
        tree.write(self.root+'/temp.xml')

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
        self.d.mocap_pos[0] = hand_position
        self.d.mocap_quat[0] = hand_quat

        self.d.qpos=hand_position+hand_quat+finger_joints+self.generate_random_obj_poses()

    def contact_pads_info(self):
        if self.contact_pads_geom_ids is None:
            print('No contact pads defined')
        else:
            print('contact pads info:')
            for group_id in self.contact_pads_geom_ids:
                for geom_id in group_id:
                    geom = self.m.geom(geom_id)
                    print(f'geom name: {geom.name}, group: {geom.group}, type: {geom.type}')



    def check_hand_contact(self,margin=0,report=False):
        is_hand_geom= lambda x: x>=1 and x<=self.last_hand_geom_id
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

    def check_valid_grasp(self,margin=0,minimum_contact_points=2,report=False):
        is_hand_geom= lambda x: x>=1 and x<=self.last_hand_geom_id
        contact_with_floor=False
        n_contact=0
        # contacts = [] # store (pos, force)
        geom_groups=[0]*len(self.contact_pads_geom_ids)
        contains_id = lambda n: any(n == x or (isinstance(x, list) and n in x) for x in self.contact_pads_geom_ids)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.dist < margin and (is_hand_geom(c.geom1) + is_hand_geom(c.geom2) ==1) :

                if c.geom1==0 or c.geom2==0:
                    contact_with_floor=True
                else:
                    contact_with_obj = True
                    pad_geom_id=None
                    obj_geom_id=None
                    if contains_id(c.geom1):
                        pad_geom_id=c.geom1
                        obj_geom_id=c.geom2
                    elif contains_id(c.geom2):
                        pad_geom_id=c.geom2
                        obj_geom_id=c.geom1

                    if pad_geom_id is not None:
                        # force = np.zeros(6)
                        # mujoco.mj_contactForce(self.m, self.d, i, force)
                        # f = np.array(force[:3])  # world-frame force
                        # p = np.array(c.pos)  # world-frame contact point
                        # contacts.append((p, f))
                        group_id=find_group(pad_geom_id,self.contact_pads_geom_ids)
                        geom_groups[group_id]+=1
                        n_contact+=1

        contacted_groups = sum(x > 0 for x in geom_groups)
        # if n_contact>0:
        #     print(Fore.LIGHTCYAN_EX,geom_groups,'--',contacted_groups,'---',n_contact,Fore.RESET)
        return contacted_groups>=minimum_contact_points and not contact_with_floor



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

    @abstractmethod
    def check_fingers_scope(self,fingers):
        pass

    @abstractmethod
    def clip_fingers_to_scope(self,hand_fingers):
        pass

    def check_collision(self,hand_pos,hand_quat,hand_fingers=None,view=False):
        self.restore_simulation_state()
        if hand_fingers is None: hand_fingers=self.default_finger_joints
        else: hand_fingers = self.clip_fingers_to_scope(hand_fingers)

        self.d.mocap_pos[0] = hand_pos
        self.d.mocap_quat[0] = hand_quat

        self.d.qpos =hand_pos + hand_quat + hand_fingers + self.objects_poses

        self.d.ctrl*=0

        mujoco.mj_step(self.m, self.d)

        '''check initial contact'''
        contact_with_obj, contact_with_floor = self.check_hand_contact()

        if view:
            self.static_view()

        return  contact_with_obj , contact_with_floor

    @abstractmethod
    def check_graspness(self,hand_pos,hand_quat,hand_fingers,obj_pose=None,view=False,iterations=50,shake_intensity=0.05):
        pass



    def visualize(self,obj_pos_quat,duration=300):
        self.d.mocap_pos[0] = self.far_hand_pos
        self.d.mocap_quat[0] = self.far_hand_quat

        self.d.qpos = self.far_hand_pos + self.far_hand_quat + self.default_finger_joints + obj_pos_quat.tolist()

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
                    hand_fing=self.default_finger_joints

                    self.d.mocap_pos[0] = hand_pos
                    self.d.mocap_quat[0] = hand_quat

                    self.d.qpos[:7 + len(self.default_finger_joints)] = hand_pos + hand_quat + hand_fing

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

        self.d.qpos = self.far_hand_pos + self.far_hand_quat + self.default_finger_joints + obj_pos_quat#.tolist()
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
            # pc = trimesh.PointCloud(pointcloud)
            # pc.show()

        return depth,pointcloud,floor_mask

    def check_masses(self):
        for body_id in range(self.m.nbody):
            body_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, body_id)
            mass = self.m.body_mass[body_id]

            if mass > 0:
                print(f"Body '{body_name}' has mass: {mass}")
            else:
                print(f"Body '{body_name}' is massless (mass=0)")

    def get_stable_object_pose(self,obj_pos_quat,threshold=1e-4,window_size=20,stablize=True):
        self.d.mocap_pos[0] = self.far_hand_pos
        self.d.mocap_quat[0] = self.far_hand_quat

        self.d.qpos= self.far_hand_pos+ self.far_hand_quat+self.default_finger_joints+obj_pos_quat

        # self.m.opt.timestep = 0.01
        counter=0
        last_obj_pos=None
        for i in range(1000):
            for j in range(window_size):
                mujoco.mj_step(self.m, self.d)
            new_obj_pos=np.copy(self.d.qpos[7 + len(self.default_finger_joints):])
            if last_obj_pos is not None:
                diff=np.sum(last_obj_pos-new_obj_pos)
                if diff<threshold:
                    counter+=1
                    self.d.qvel[:] = 0
                    if not stablize:self.d.cvel[:] = 0
                    self.d.qacc[:] = 0.0
                    self.d.qfrc_applied[:] = 0.0
                    self.d.qfrc_actuator[:] = 0.0
                    self.d.qfrc_constraint[:] = 0.0
                    self.d.ctrl[:] = 0.0
                    if counter>window_size:
                        break
                else:
                    counter=0

            last_obj_pos=new_obj_pos
        else:
            print('Cannot find a stable pose')
            return None

        self.m.opt.timestep = 0.002

        return  self.d.qpos[7 + len(self.default_finger_joints):]

    def manual_view(self,pos=None,quat=None):
        pos=[0., 0., .3] if pos is None else pos
        quat = [0., 1., 0., 0.] if quat is None else quat

        self.d.mocap_pos[0] = pos
        self.d.mocap_quat[0] = quat

        self.d.qpos[0:7]=pos + quat

        # self.d.qpos = pos + quat + self.default_finger_joints+list(self.objects_poses)
        k=3+4+len(self.default_finger_joints)
        self.d.qpos[k:] = list(self.objects_poses)


        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD  # show world frame
            # optional: show contact points
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

            while viewer.is_running():
                step_start = time.time()
                # self.d.qpos[k:]=list(self.objects_poses)

                mujoco.mj_step(self.m, self.d)

                viewer.sync()
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    def passive_viewer(self,pos=None,quat=None,delta_pos=None,ctrl=None):
        pos=[0., 0., .3] if pos is None else pos
        quat = [0., 1., 0., 0.] if quat is None else quat

        self.d.mocap_pos[0] = pos
        self.d.mocap_quat[0] = quat

        self.d.qpos = pos + quat + self.default_finger_joints +list(self.objects_poses)

        self.d.ctrl *= 0

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD  # show world frame
            # Close the viewer automatically after 30 wall-seconds.
            start = time.time()
            while viewer.is_running() and time.time() - start < 30:
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                if self.d.time > 0.5 and delta_pos is not None:
                    self.d.mocap_pos[0] = self.d.mocap_pos[0] + delta_pos
                self.d.mocap_quat[0] = quat

                if ctrl is not None: self.d.ctrl = ctrl
                # print(d.qpos)
                mujoco.mj_step(self.m, self.d)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
if __name__ == "__main__":
    print(mujoco.__version__)

    shadow_hand_env=MojocoMultiFingersEnv(obj_nums_in_scene=1,selected_idx=[6])
    obj_pose=[0., 0., 0.]
    obj_quat=[1, 0, 0, 0]
    finger_joints=[0, -0.8, 0, 0, 0, -0.8, 0, 0, 0, -0.8, 0, 0]
    stable_obj_pose_quat=shadow_hand_env.get_stable_object_pose(obj_pose+obj_quat)
    # print(stable_obj_pose_quat)

    hand_pos = [0,0,0]
    hand_quat = np.array([-1,1,0,0])
    hand_quat = hand_quat / np.linalg.norm(hand_quat)
    hand_fing = shadow_hand_env.default_finger_joints
    hand_fing=[-0.6147308349609375, -1.2909823656082153, 0.053371675312519073, -0.08044019341468811, -0.24372847378253937, -0.7953169941902161, 0.13495197892189026, 0.14904902875423431, -0.3114812672138214, -0.6036136150360107, 0.26515403389930725, -0.13743282854557037]
    # r=shadow_hand_env.check_graspness(hand_pos=hand_pos,hand_quat=hand_quat.tolist(),hand_fingers=hand_fing,obj_pose=stable_obj_pose_quat.tolist(),view=True)
    # print(r)
    # shadow_hand_env.visualize(stable_obj_pose_quat)
    shadow_hand_env.manual_view()
