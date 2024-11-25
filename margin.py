import asyncio
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import open3d as o3d

print('running async test')

def say_boo():
    i = 0
    while True:
        print('...boo {0}'.format(i))
        i += 1


def say_baa():
    i = 0
    while True:
        print('...baa {0}'.format(i))
        i += 1

if __name__ == "__main__":
    def compute_curvature(pcd, radius=0.5):
        points = np.asarray(pcd.points)

        # Build a KDTree for efficient neighbor search
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        curvature = []
        for point in points:
            # Find neighbors within the specified radius
            [k, idx, _] = kdtree.search_radius_vector_3d(point, radius)

            # If not enough neighbors, set curvature to 0
            if k < 3:
                curvature.append(0)
                continue

            # Compute the covariance matrix of the neighborhood
            M = np.cov(np.asarray(pcd.points)[idx].T)

            # Calculate eigenvalues and eigenvectors
            eigenvalues, _ = np.linalg.eigh(M)

            # Curvature is approximated by the ratio of the smallest eigenvalue to the sum of eigenvalues
            curvature.append(eigenvalues[0] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]))

        return curvature


    # Load point cloud
    pcd = o3d.io.read_point_cloud("your_point_cloud.pcd")

    # Compute curvature
    curvature = compute_curvature(pcd)

    # Add curvature as a scalar field to the point cloud
    pcd.estimate_normals()
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((np.asarray(pcd.points).shape[0], 3)))
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))

    # Visualize the point cloud with curvature as color
    o3d.visualization.draw_geometries([pcd])
