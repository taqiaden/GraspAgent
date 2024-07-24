import numpy as np
import trimesh


def construct_gripper_mesh(width,T):
    mesh = trimesh.load("new_gripper.ply")

    # Add limit to the distance width

    # width = min(width * 1.1, 0.05)
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

    # print(f'mesh vertics = {mesh.vertices[[2, 3, 18, 19], 1] }')


    # transform from vertical to horizontal; Z+ becomes x-.
    # The gripper closing direction is a long the Y direction in both cases
    mesh.apply_transform(T2)

    # assert T[0:3, 0:3].sum()>0.0,f'{T}'

    # return mesh
    # print(Fore.RED,T,Fore.RESET)
    mesh.apply_transform(T)


    return mesh