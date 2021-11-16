import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skl
import open3d as o3d


def gaussian(dist, mu=0, sigma=1.0):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.e ** (-0.5*((dist-mu)/sigma)**2)


pcd_base = o3d.io.read_point_cloud("Low_LoD.ply")
pcd_ref = o3d.io.read_point_cloud("Sparse.ply")

core = pcd_base.points[1000]
radius = 0.2
n_sigma = 3

pcd_tree = o3d.geometry.KDTreeFlann(pcd_ref)

[k, idx, _] = pcd_tree.search_radius_vector_3d(core, radius)

neighbors = np.asarray(pcd_ref.points)[idx[1:], :]
print(neighbors.shape)

dist_set = np.sqrt(np.sum((core-neighbors)**2, axis=1))
print("距离", dist_set)

# test_x = np.linspace(-400, 400, 200)
# ss = skl.MinMaxScaler(feature_range=(-radius/2, radius/2))
# test_x = ss.fit_transform(dist_set.reshape(-1, 1))
test_x = dist_set - np.min(dist_set)
test_y = gaussian(test_x, sigma=(radius-np.min(dist_set))/3)
print("权重", test_y.flatten())
plt.scatter(test_x, test_y)
plt.show()
# print(gaussian(0))

temp = dist_set.reshape(-1, 1)*test_y.reshape(-1, 1)
print("距离和权重乘积", temp.flatten())
weighted_dist = np.sum(dist_set.reshape(-1, 1)*test_y.reshape(-1, 1))/np.sum(test_y)
print(weighted_dist, np.average(dist_set))

plt.scatter(np.arange(len(dist_set)), dist_set.flatten())
plt.show()
