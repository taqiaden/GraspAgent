import copy
import numpy as np
import trimesh
import open3d as o3d

parallel_jaw_model= 'new_gripper.ply'

parallel_jaw_mesh_trimesh=None
parallel_jaw_mesh_o3d=None

def construct_gripper_mesh(width,T_d):
    global  parallel_jaw_mesh_trimesh
    if parallel_jaw_mesh_trimesh is None:
        parallel_jaw_mesh_trimesh=trimesh.load(parallel_jaw_model)
    mesh = copy.deepcopy(parallel_jaw_mesh_trimesh)

    T1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, -0.02],
                   [0, 0, 1, 0.02],
                   [0, 0, 0, 1]])
    mesh.apply_transform(T1)
    mesh.vertices = mesh.vertices * np.array([14, 1, 1])
    # save_trimesh_mesh_as_ply(mesh,"gripper.ply" )
    mesh.vertices[[2, 3, 18, 19], 1] = -0.0005 - width / 2
    mesh.vertices[[12, 13, 28, 29], 1] = 0.0005 + width / 2
    mesh.vertices[[8, 9, 10, 11, 24, 25, 26, 27], 1] = -0.0005 - width / 2 - 0.004
    mesh.vertices[[4, 5, 6, 7, 20, 21, 22, 23], 1] = 0.0005 + width / 2 + 0.004
    shape_change = np.ones_like(mesh.vertices)
    shape_change[:, 2] = 2.5
    shape_change[[4, 8, 20, 24], 2] = 1
    mesh.vertices = mesh.vertices * shape_change
    T1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, -0.004],
                   [0, 0, 0, 1]])
    # move the gripper a distance -0.004 in the z direction
    mesh.apply_transform(T1)

    T2 = np.array([[0, 0, -1, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1]])


    mesh.apply_transform(T2)

    mesh.apply_transform(T_d)


    return mesh

def construct_gripper_mesh_2(width,T_d):
    '''get the raw gripper mesh'''
    global parallel_jaw_mesh_o3d
    if parallel_jaw_mesh_o3d is None:
        parallel_jaw_mesh_o3d = o3d.io.read_triangle_mesh(parallel_jaw_model)

    mesh = copy.deepcopy(parallel_jaw_mesh_o3d)

    '''adjust the width'''
    scale_k = width / 0.04
    mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices * np.array([8, scale_k, 1]))
    mesh.translate([0, -0.02 * scale_k, 0.016])

    '''transformation'''
    T_ = np.eye(4)
    T_[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, -np.pi / 2, 0))
    mesh.transform(T_)
    mesh = mesh.transform(T_d)

    return mesh

if __name__ == '__main__':
    pass