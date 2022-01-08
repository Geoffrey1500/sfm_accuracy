import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skl
from sklearn.neighbors import KDTree
import open3d as o3d
import time
import pickle


def gaussian(dist, mu=0, sigma=1.0):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.e ** (-0.5*((dist-mu)/sigma)**2)


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb') #以二进制读写方式打开文件
    pickle.dump(inputTree, fw)  #pickle.dump(对象, 文件，[使用协议])。序列化对象
    # 将要持久化的数据“对象”，保存到“文件”中，使用有3种，索引0为ASCII，1是旧式2进制，2是新式2进制协议，不同之处在于后者更高效一些。
    #默认的话dump方法使用0做协议
    fw.close() #关闭文件


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


pcd_base = o3d.io.read_point_cloud("data/1_8_2.ply")
point_base = np.asarray(pcd_base.points)

start_time = time.process_time()
pcd_ref = o3d.io.read_point_cloud("data/1_8_2_df - point.pcd")
end_time = time.process_time()
print("读取点云数据总耗时 ：" + str(end_time - start_time) + "s")
points_in_ref = np.asarray(pcd_ref.points)

points = pcd_ref.points

# core = points_in_ref[5000]
# core = np.array([[-2606.13036, -1718.20796, 280.52866]])/1000
core = point_base[4086]
print("核心点坐标：", core)
radius = 9.74/1000
n_sigma = 3

# start_time = time.process_time()
# kdt = KDTree(points_in_ref, leaf_size=5, metric='euclidean')
# end_time = time.process_time()
# print("knn建树总耗时 ：" + str(end_time - start_time) + "s")
#
# storeTree(kdt, 'tree.txt')
#
start_time = time.process_time()
kdt = grabTree("tree.txt")
end_time = time.process_time()
print("读取knn树总耗时 ：" + str(end_time - start_time) + "s")


start_time = time.process_time()
dis2, idx2 = kdt.query(core.reshape((1, -1)), k=100)
dis2 = dis2[0]
idx2 = idx2[0]

idx, dis = kdt.query_radius(core.reshape((1, -1)), r=radius, return_distance=True)
end_time = time.process_time()
print("临近点搜寻总耗时 ：" + str(end_time - start_time) + "s")

neighbors = np.asarray(pcd_ref.points)[idx[0], :]
print(neighbors.shape)

dist_set = np.sqrt(np.sum((core-neighbors)**2, axis=1))
dist_set_2 = dis[0]
print("距离", dist_set)

print("原生距离", dis[0])

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
