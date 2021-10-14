import open3d as o3d
import numpy as np
import pyvista as pv

pcd = o3d.io.read_point_cloud("Sparse.ply")
diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])

camera = [-21.90558665, -13.96051597,  31.58213869] # 定义用于隐藏点删除的参数，获取从给定视图中可见的所有点，可视化结果
radius = 52078

points_coor = np.asarray(pcd.points)*1000

_, pt_map = pcd.hidden_point_removal(camera, radius)

print(type(pt_map))
pcd_new = pcd.select_by_index(pt_map)

pcd_new_2 = points_coor[pt_map]

tmp = []

# for i in range(len(points_coor)):
#     if (np.asarray(pcd_new.points)*1000 == points_coor[i]).any():
#         tmp.append(i)
#         # print(index_help.size)

print(len(tmp), len(np.asarray(pcd_new.points)), len(pt_map), len(pcd_new_2), len(points_coor))
view_point = o3d.geometry.TriangleMesh.create_sphere()
view_point.translate(np.array([camera]).T)

# o3d.visualization.draw_geometries([pcd_new, mesh_frame, view_point])


a = np.arange(12).reshape((4, 3))
print(a)

# print((a == np.array([[0, 1, 2]])).all())

if np.array([[0, 1, 3]]) in a:
    print("very good")

start = a[1]
end = a[2]

s_f = np.ones(3)*0.5
ref = (start + end)/2
print(start)
print(end)
print(ref)
print(ref[0], ref[1], ref[2])


def scale_with_ref(scale_factor, reference, point_):
    scale_mat_ = np.array([[scale_factor[0], 0, 0, (1 - scale_factor[0]) * reference[0]],
                           [0, scale_factor[1], 0, (1 - scale_factor[1]) * reference[1]],
                           [0, 0, scale_factor[2], (1 - scale_factor[2]) * reference[2]],
                           [0, 0, 0, 1]])
    point_ = np.vstack((point_.reshape((-1, 1)), np.array([[1]])))
    point_new_ = np.dot(scale_mat_, point_)
    point_new_ = point_new_[0:3, :].flatten()

    return point_new_


start_new = scale_with_ref(s_f, ref, start)
end_new = scale_with_ref(s_f, ref, end)


print(start_new, end_new)
print(start_new.shape)

ray_1 = pv.Line(start, end)
ray_2 = pv.Line(start_new, end_new)

plotter = pv.Plotter()
plotter.add_mesh(ray_1, color="green", label="Ray Segment", opacity=0.75)
plotter.add_mesh(ray_2, color="red", label="Scaled Ray Segment", opacity=0.75)
plotter.show()

