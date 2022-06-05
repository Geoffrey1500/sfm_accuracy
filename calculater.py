import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.neighbors import KDTree
import pickle


def storeTree(inputTree, filename):
    # 序列化决策树,存入文件
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


reference_points = o3d.io.read_point_cloud("data/for_paper/Laser_B.pcd")
# o3d.io.write_point_cloud("data/UAV_B_mesh.ply", reference_points)

# np.savetxt('reference_points.txt', reference_points)

# rng = np.random.RandomState(0)
# X = rng.random_sample((10, 3))  # 10 points in 3 dimensions

xyz_load = np.asarray(reference_points.points)
tree = KDTree(xyz_load, leaf_size=2)
# s = pickle.dumps(tree)
storeTree(tree, "data/for_paper/Laser_B_tree.txt")
# print(X[:1])
# print(tree.query_radius(X[:1], r=0.3, count_only=True))
#
# ind = tree.query_radius(X[:1], r=0.3)
# print(ind)  # indices of neighbors within distance 0.3
