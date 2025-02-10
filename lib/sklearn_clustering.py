import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from lib.pc_utils import numpy_to_o3d
from visualiztion import view_npy_open3d

def dbscan_clustering(pc,eps=0.15, min_samples=10,view=False):
    # Normalisation:
    scaled_points = StandardScaler().fit_transform(pc)
    # Clustering:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(scaled_points)
    # Get labels:
    labels = model.labels_

    if view:
        view_clusters(pc,labels)
    return labels

def view_clusters(points,labels):
    n_clusters = len(set(labels))
    # Mapping the labels classes to a color map:
    colors = plt.get_cmap("tab20")(labels / (n_clusters if n_clusters > 0 else 1))
    # Attribute to noise the black color:
    colors[labels < 0] = 0
    # Update points colors:
    pcd = numpy_to_o3d(points)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # Display:
    o3d.visualization.draw_geometries([pcd])

def kmeans_clustering(pc,n_clusters=4,view=False):
    # Normalisation:
    scaled_points = StandardScaler().fit_transform(pc)
    # Clustering:
    model = KMeans(n_clusters=n_clusters)
    model.fit(scaled_points)
    # Get labels:
    labels = model.labels_

    if view:
        view_clusters(points,labels)
    return labels

if __name__ == '__main__':
    points=np.load(r'/media/taqiaden/42c447a4-49c0-4d74-9b1f-4b4b5cbe7486/GraspAgent/lib//test.npy')
    # pcd = numpy_to_o3d(pc)
    # Read point cloud:
    # pcd = o3d.io.read_point_cloud("../data/depth_2_pcd_downsampled.ply")
    # Get points and transform it to a numpy array:
    # points = np.asarray(pcd.points)

    # Normalisation:
    scaled_points = StandardScaler().fit_transform(points)

    # Clustering:
    model = DBSCAN(eps=0.15, min_samples=10)
    # model = KMeans(n_clusters=4)
    model.fit(scaled_points)

    # Get labels:
    labels = model.labels_
    print(set(labels))
    # Get the number of colors:
    n_clusters = len(set(labels))

    # Mapping the labels classes to a color map:
    colors = plt.get_cmap("tab20")(labels / (n_clusters if n_clusters > 0 else 1))
    # Attribute to noise the black color:
    colors[labels < 0] = 0
    # Update points colors:
    pcd = numpy_to_o3d(points)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # Display:
    o3d.visualization.draw_geometries([pcd])