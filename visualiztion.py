import math
import torch
import trimesh
import copy
import random

from Configurations import config, ENV_boundaries
from lib.bbox import decode_gripper_pose
from lib.pc_utils import numpy_to_o3d
import matplotlib.pyplot as plt
from lib.report_utils import distribution_summary
from masks import get_spatial_mask
from pose_object import approach_vec_to_theta_phi, output_processing

parallel_jaw_model= 'new_gripper.ply'

object_prediction_threshold = 0.5

def dense_grasps_visualization(pc, generated_pose_7,grasp_score_pred,target_mask):


    # Method 1
    pose_good_grasp_list = []
    for i in range(5000):
        random_index = np.random.randint(0, config.num_points)
        if target_mask[0, random_index, 0] == False: continue

        if grasp_score_pred[0, 0, random_index] < 0.4: continue

        tmp_pose = generated_pose_7[0:1, :, random_index]

        poses_7 = tmp_pose[:, None, :].clone()

        # visualize verfication
        pc_ = pc[0, :, 0:3].cpu().numpy()  # + mean

        center_point = pc_[random_index]
        target_pose = poses_7[0, :, :]
        approach = target_pose[:, 0:3]

        theta, phi_sin, phi_cos = approach_vec_to_theta_phi(approach)
        target_pose[:, 0:1] = theta
        target_pose[:, 1:2] = phi_sin
        target_pose[:, 2:3] = phi_cos
        pose_5 = output_processing(target_pose[:, :, None]).squeeze(-1)

        pose_good_grasp = decode_gripper_pose(pose_5, center_point[0:3])

        # extreme_z = center_point[-1] - poses_7[0,0, -2] * config.distance_scope
        # if  extreme_z < config.z_limits[0]: continue

        pose_good_grasp_list.append(pose_good_grasp)
    if len(pose_good_grasp_list) == 0: return
    pose_good_grasp = np.concatenate(pose_good_grasp_list, axis=0)
    # print(pose_good_grasp.shape)
    # print(target_mask.squeeze().shape)
    # print(pc.shape)
    # masked_pc=pc[0,target_mask.squeeze(),0:3]
    # print(masked_pc.shape)
    # print(pc[0, :, 0:3].shape)

    vis_scene(pose_good_grasp[:, :], npy=pc[0, :, 0:3].cpu().numpy())

    return


def view_o3d(pcd,view_coordinate=True,geometries_list=None):
    o = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0]) if view_coordinate else o3d.geometry.PointCloud()
    list=[] if geometries_list is None else geometries_list
    list.append(pcd)
    list.append(o)
    o3d.visualization.draw_geometries(list)
def view_o3d_objects(list_of_objects):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for obj in list_of_objects:
        vis.add_geometry(obj)
        vis.run()
        vis.destroy_window()

def view_npy_open3d(npy, view_coordinate=True,geometries_list=None):
    pcd = numpy_to_o3d(npy=npy)
    view_o3d(pcd,view_coordinate,geometries_list)

def get_random_color():
    r=random.randint(0,255)
    g=random.randint(0,255)
    b=random.randint(0,255)
    return [r,g,b]

def view_npy_trimesh(npy_list,color_list=[],pick_random_colors=False):
    pc_=[]
    for i in range(len(npy_list)):
        if len(npy_list[i])<=1:continue
        # continue
        if len(color_list)>i:
            pc_.append(trimesh.PointCloud(npy_list[i], colors=color_list[i]))
        else:
            if pick_random_colors:
                pc_.append(trimesh.PointCloud(npy_list[i], colors=get_random_color()))
            else:
                pc_.append(trimesh.PointCloud(npy_list[i]))
    if pc_!=[]:
        scene_ = trimesh.Scene(pc_)
        scene_.show()

def visualize_suction_pose(suction_xyz, suction_pose, T, end_effecter_mat,  npy):
    suction_xyz = suction_xyz.squeeze()
    suction_pose = suction_pose.squeeze()

    pcd = highlight_background(npy)

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    axis_left_arm = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    # draw line
    points = [[suction_xyz[0], suction_xyz[1], suction_xyz[2]],
              [suction_xyz[0] + suction_pose[0] / 10,
               suction_xyz[1] + suction_pose[1] / 10,
               suction_xyz[2] + suction_pose[2] / 10]]
    lines = [[0, 1]]

    points = o3d.utility.Vector3dVector(points)
    lines = o3d.utility.Vector2iVector(lines)

    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    scene_suction = list()
    scene_suction.append(line_set)
    scene_suction.append(pcd)

    axis_pcd.transform(T)
    axis_left_arm.transform(end_effecter_mat)
    o3d.visualization.draw_geometries([pcd, axis_pcd, axis_left_arm, line_set])

def highlight_background(npy):
    colors=np.zeros_like(npy)
    background_mask=get_spatial_mask(npy).squeeze()
    colors[background_mask]=[0.,0.,0.24]
    colors[~background_mask] = [0.65, 0.65, 0.65]
    pcd = numpy_to_o3d(npy=npy,color=colors)
    return pcd
def vis_scene(grasp, npy=None):
    # visualize
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    axis_right_arm = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

    # load gripper
    mesh = o3d.io.read_triangle_mesh(parallel_jaw_model)
    width = grasp[0, 0]
    scale_k = width / 0.04
    mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices * np.array([8, scale_k, 1]))

    mesh.translate([0, -0.02 * scale_k, 0.016])
    T_ = np.eye(4)
    T_[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, -np.pi / 2, 0))
    mesh.transform(T_)

    pcd = highlight_background(npy)

    scene_list = []
    for i in range(grasp.shape[0]):
        # if i > 0:
        #     break

        T_xyz = grasp[i, [1, 2, 3]]
        T_R = grasp[i, [4, 5, 6, 7, 8, 9, 10, 11, 12]].reshape(-1, 3, 3)
        T_R = T_R.transpose(0, 2, 1)
        T = np.eye(4)
        T[:3, :3] = T_R
        T[:3, 3] = T_xyz.T

        mesh_ = copy.deepcopy(mesh).transform(T)
        axis_pcd.transform(T)
        rot = [[0, 0, 1, -0.18],
               [-1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, 0, 1]]

        end_effecter_mat = np.dot(T, rot)

        axis_right_arm.transform(end_effecter_mat)

        mesh_.paint_uniform_color([1,0, 0])
        scene_list.append(mesh_)
    scene_list.append(pcd)
    o3d.visualization.draw_geometries(scene_list)

def visualize_detected_objects(objectness_pred, data_, object_prediction_threshold=object_prediction_threshold):
    objectness_pred_mask = objectness_pred > object_prediction_threshold
    ground_mask = ~objectness_pred_mask

    pc_objectness = trimesh.PointCloud(data_[objectness_pred_mask], colors=[0, 0, 255])
    pc_ground = trimesh.PointCloud(data_[ground_mask], colors=[0, 255, 0])
    scene_ = trimesh.Scene([pc_objectness, pc_ground])
    scene_.show()

def visualize_grasp_and_suction_points(suction_cls_pred_mask, grasp_cls_pred_mask, data_):
    grasp_suction_all = suction_cls_pred_mask & grasp_cls_pred_mask
    suction_only = suction_cls_pred_mask & ~grasp_cls_pred_mask
    grasp_only = grasp_cls_pred_mask & ~suction_cls_pred_mask

    postive_mask = suction_cls_pred_mask | grasp_cls_pred_mask
    negtive_mask = ~postive_mask

    scene_all = trimesh.Scene()

    if not True in grasp_suction_all:
        print('NO grasp_suction_all')
    else:
        pointcloud_blue = trimesh.PointCloud(data_[grasp_suction_all], colors=[0, 0, 255])
        scene_all.add_geometry(pointcloud_blue)

    if not True in suction_only:
        print('NO suction points')
    else:
        pointcloud_suction_only = trimesh.PointCloud(data_[suction_only], colors=[255, 165, 0])
        scene_all.add_geometry(pointcloud_suction_only)

    if not True in grasp_only:
        print('NO grasp points')
    else:
        pointcloud_grasp_only = trimesh.PointCloud(data_[grasp_only], colors=[255, 0, 255])
        scene_all.add_geometry(pointcloud_grasp_only)
    neg=data_[negtive_mask]
    if neg.shape[0]>0:
        pointcloud_blue_ = trimesh.PointCloud(neg, colors=[0, 255, 0])
        scene_all.add_geometry(pointcloud_blue_)
    scene_all.show()

def create_point_cloud_from_depth_image(depth, camera, organized=False):
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width), 'depth shape error! depth.shape = {}'.format(depth.shape)
    # camera.height= depth.shape[0]
    # camera.width=depth.shape[1]
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
        mask = cloud[:, 2] != 0
        cloud = cloud[mask]
    return cloud

def create_depth_image_from_point_cloud(pc, camera):
    depth_map = np.zeros([camera.height, camera.width])

    mask = pc[:, 2] != 0
    pc = pc[mask]
    depth = pc[:, 2]
    y = pc[:, 0] * camera.fx / depth + camera.cx
    x = pc[:, 1] * camera.fy / depth + camera.cy
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    x = np.clip(x, 0, camera.height - 1)
    y = np.clip(y, 0, camera.width - 1)
    depth = depth * camera.scale

    for i, j, d in zip(x, y, depth):
        depth_map[i, j] = d
    return depth_map[175:535, 220:700]

class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def vis_depth_map(depth, view_as_point_cloud=True):
    if view_as_point_cloud:
        if isinstance(depth,torch.Tensor):
            depth=depth.numpy()
        camera = CameraInfo(480, 360, 1122.375, 1122.375, 296, 211, 1000)
        cloud = create_point_cloud_from_depth_image(depth, camera)

        points = cloud.reshape(-1, 3)
        point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        point.transform(np.array([[0.010182, -0.999944, 0.003005, 0.39310000],
                           [-0.985716, -0.009532, 0.168148, -0.2809940],
                           [-0.168110, -0.004674, -0.985757, 1.3378300],
                           [0.0, 0.0, 0.0, 1.0]]))
        o3d.visualization.draw_geometries([point, axis_pcd])
    else:
        plt.imshow(depth, cmap='gray')
        plt.show()

def transform_coordinate(pc):
    matrix = np.array([[0.010182, -0.999944, 0.003005, 0.39310000],
                       [-0.985716, -0.009532, 0.168148, -0.2809940],
                       [-0.168110, -0.004674, -0.985757, 1.3378300],
                       [0.0, 0.0, 0.0, 1.0]])

    matrix_inv = np.linalg.inv(matrix)
    column = np.ones(len(pc))
    stacked = np.column_stack((pc, column))
    transformed = np.dot(matrix_inv, stacked.T).T[:, :3]
    transformed = np.ascontiguousarray(transformed)
    return transformed

import open3d as o3d
import numpy as np

def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def get_arrow(end, origin=np.array([0, 0, 0]), scale=1):
    assert(not np.all(end == origin))
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))
    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
        cone_height=size*0.2 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*(1 - 0.2*scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))

    mesh.translate(origin)
    return(mesh)

def draw_arrow_implementation_example():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    arrow=get_arrow(origin=np.array([0, 0, 0]), end=np.array([1, 1, 1]), scale=1 / np.sqrt(3))
    vis.add_geometry(arrow)
    # vis.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame())
    vis.run()
    vis.destroy_window()

def score_to_color(score, RGB_variant=0):
    # print(np.isnan(score).any())
    # assert ~np.isnan(score).any()
    max_score,min_score,average,std=distribution_summary(score,data_name='Score')

    color=np.zeros((score.shape[0],3))
    for i in range(score.shape[0]):
        if score.shape[0]==1:color_intensity=0
        else:
            assert max_score-min_score!=0
            color_intensity=math.floor((1-(score[i]-min_score)/(max_score-min_score))*255)
            color_intensity=min(color_intensity,255)
            color_intensity=max(color_intensity,0)

        color[i,RGB_variant]=255
        color[i,(RGB_variant+1)%3]=color_intensity
        color[i,(RGB_variant+2)%3]=color_intensity
    return color.astype(int)

def score_visualization(npy_points,npy_score):
    colors = score_to_color(npy_score, RGB_variant=0)
    p = trimesh.points.PointCloud(vertices=npy_points, colors=colors)
    p.show()

def view_score(data_,mask,score):
    masked_data = data_[mask]
    if masked_data.shape[0] == 0: return
    rest_of_data = data_[~mask]

    masked_score = score[mask]

    colors = score_to_color(masked_score, RGB_variant=0)

    p_data = trimesh.PointCloud(masked_data, colors=colors)
    if rest_of_data.shape[0] > 0: p_rest = trimesh.PointCloud(rest_of_data, colors=[255, 255, 255])

    scene = trimesh.Scene()
    scene.add_geometry(p_data)
    if rest_of_data.shape[0] > 0: scene.add_geometry(p_rest)

    scene.show()

if __name__ == '__main__':
    x=np.random.random((10000,3))
    s=np.random.random((10000))
    score_visualization(x,s)
    # view_npy_open3d(x)
    # view_npy_trimesh([x])


    # draw_arrow_implementation_example()
    # exit()
    # pc = np.load('dataset/realtime_dataset/point_cloud/000770_pointdata.npy')
    # print(pc.shape)
    # pc=transform_coordinate(pc)
    #nympy to int
    # dp = create_depth_image_from_point_cloud(pc, CameraInfo(1032, 772, 1122.375, 1122.375, 516, 386, 1000))
    # print(dp.shape)
    # vis_depth_map(dp, view_as_point_cloud=True)
