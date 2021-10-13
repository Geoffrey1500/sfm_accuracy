import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("Model.ply")
mesh = o3d.io.read_triangle_mesh("Model.ply")
diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=5, origin=[0, 0, 0])

print(diameter)
print(np.asarray(pcd.get_max_bound()), np.asarray(pcd.get_min_bound()))
# o3d.visualization.draw_geometries([pcd])
print("Define parameters used for hidden_point_removal")
camera = [0, 0, 40] # 定义用于隐藏点删除的参数，获取从给定视图中可见的所有点，可视化结果
radius = diameter * 100

print("Get all points that are visible from given view point")
_, pt_map = pcd.hidden_point_removal(camera, radius)

print("Visualize result")
pcd_new = mesh.select_by_index(pt_map)
# print(pt_map)
# print(pcd_new)
# o3d.visualization.draw_geometries([pcd_new, mesh_frame])

point_set_after = pcd.select_by_index(pt_map)
np_point = np.asarray(point_set_after.points)
original_point = np.asarray(pcd.points)
print(np_point)
print(len(np_point), len(original_point))
print(original_point[0])
mask = np.isin(original_point[0], np_point)
print(np_point[0] in original_point)

print("查看原始数据")
print(np_point)
print("查看原始数据")
print(original_point)

ans = []
for i in range(len(original_point)):

    index_help_3 = np.where(np.sum(np.absolute(np_point - original_point[i]), axis=1) <= 10**(-6))[0]
    # print(tmp3.tolist())
    # print(tmp3.shape, "stop here")
    # print(tmp3 == 27)
    # print(np.where(tmp3 == 0)[0])

    if index_help_3.size:
        ans.append(i)
    # # print(index_help_2)
    # if original_point[i] in np_point:
    #     # print(i)
    #     # print("hello")
    #     ans.append(i)
    # # print((index_help_2 == 0).all())
    # # print(i)


print(len(ans), len(np_point), len(pt_map), len(original_point))
print(ans)
print(np.sort(pt_map).tolist())
print(original_point[ans])
print(np_point)
# o3d.io.write_point_cloud("temp.ply", pcd)

# mesh = o3d.io.read_triangle_mesh("Model.ply")
# print(mesh)
# o3d.io.write_triangle_mesh("test.ply", pcd_new)


print()
