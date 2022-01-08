import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.neighbors import KDTree

reference_points = o3d.io.read_point_cloud("data/1_8_2_df - point.xyz")
o3d.io.write_point_cloud("data/1_8_2_df - point.pcd", reference_points)
# np.savetxt('reference_points.txt', reference_points)

# rng = np.random.RandomState(0)
# X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
# tree = KDTree(X, leaf_size=2)
# print(X[:1])
# print(tree.query_radius(X[:1], r=0.3, count_only=True))
#
# ind = tree.query_radius(X[:1], r=0.3)
# print(ind)  # indices of neighbors within distance 0.3
