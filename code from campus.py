import pyvista as pv
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation as R


'''
大疆Phantom 4 Pro
传感器大小：1英寸 13.2 mm x 8.8 mm
分辨率：5472×3648
像元大小：2.4123 um
焦距：8.8 mm
FOV：84°？

'''
print(np.arctan((13.2/2)/8.8)/np.pi*180*2)
w, h = 13.2, 8.8
f = 8.8
resol_x, resol_y = 5472, 3648
data = pd.read_csv("internal_and_external_parameters_4.csv")
print(data.head(5))
cam_loc = data[["x", "y", "z"]].values*1000
euler_ang = data[["heading", "pitch", "roll"]].values * np.array([[-1, 1, 1]]) + np.array([[0, 0, 0]])


def scale_with_ref(scale_factor, reference, point_):
    '''
    使得物体以参考点为中心进行缩放
    :param scale_factor: 缩放系数 (1 x N)
    :param reference: 参考点 (1 x N)
    :param point_: 目标点坐标 (1 x N)
    :return: 转换后的点坐标 (1 x N)
    '''
    scale_mat_ = np.array([[scale_factor[0], 0, 0, (1 - scale_factor[0]) * reference[0]],
                           [0, scale_factor[1], 0, (1 - scale_factor[1]) * reference[1]],
                           [0, 0, scale_factor[2], (1 - scale_factor[2]) * reference[2]],
                           [0, 0, 0, 1]])
    point_ = np.vstack((point_.reshape((-1, 1)), np.array([[1]])))
    point_new_ = np.dot(scale_mat_, point_)
    point_new_ = point_new_[0:3, :].flatten()

    return point_new_

# To create sensor plane
n_points = 2
scale_factor = 300
X1 = np.linspace(-w/2*scale_factor, w/2*scale_factor, 3*n_points)
Y1 = np.linspace(-h/2*scale_factor, h/2*scale_factor, 2*n_points)
X, Y = np.meshgrid(X1, Y1)
Z = (np.zeros_like(X) - f)*scale_factor

X_reshaped = np.reshape(X, (-1, 1))
Y_reshaped = np.reshape(Y, (-1, 1))
Z_reshaped = np.reshape(Z, (-1, 1))

point_data = np.hstack((X_reshaped, Y_reshaped,  Z_reshaped))

# plotter = pv.Plotter()

mesh_tower = pv.read("Sparse.ply")
mesh_tower.scale([1000, 1000, 1000])


pcd = o3d.io.read_point_cloud("Sparse.ply")

points_coor = np.asarray(pcd.points)*1000
points_color = np.asarray(pcd.colors)

for j in np.arange(900, 905):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_tower, show_edges=True, color="white")

    start = points_coor[j]
    # pcd_copy = copy.deepcopy(pcd)
    # print(len(np.asarray(pcd_copy.points)), "点的数量")

    for i in range(len(euler_ang)):
        rot_mat = R.from_euler('ZXY', [euler_ang[i]], degrees=True)
        rotated_data = rot_mat.apply(point_data)
        data_after = rotated_data + cam_loc[i]

        cloud = pv.PolyData(data_after)
        sensor_plane = cloud.delaunay_2d()

        stop = cam_loc[i]

        # Perform ray trace
        point_cam, ind_cam = sensor_plane.ray_trace(start, stop)

        if ind_cam.size:
            # ray = pv.Line(start, stop)
            # intersection = pv.PolyData(point_cam)
            # plotter.add_mesh(ray, color="white", line_width=1, label="Ray Segment", opacity=0.75)
            # plotter.add_mesh(intersection, color="blue",
            #                  point_size=5, label="Intersection Points")
            # plotter.add_mesh(sensor_plane, show_edges=True, opacity=0.75, color="white")

            # 延长射线，确保两端都突出一点，减少计算错误
            ref = (start + stop) / 2
            s_f = np.ones(3) * (1 + 5 / np.linalg.norm(start - stop))
            start_new = scale_with_ref(s_f, ref, start)
            point_main, ind_main = mesh_tower.ray_trace(stop, start_new)

            point_old, ind_old = mesh_tower.ray_trace(stop, start)
            #
            # if ind_main.size and len(ind_main) <= 1:
            #     ray = pv.Line(start_new, stop)
            #     intersection = pv.PolyData(point_cam)
            #     plotter.add_mesh(ray, color="blue", line_width=1, label="Ray Segment", opacity=0.75)
            #     plotter.add_mesh(intersection, color="blue",
            #                      point_size=5, label="Intersection Points")
            #     plotter.add_mesh(sensor_plane, show_edges=True, opacity=0.75, color="blue")
            # #
            if ind_old.size and len(ind_old) <= 1:
                ray = pv.Line(start, stop)
                intersection = pv.PolyData(point_cam)
                plotter.add_mesh(ray, color="r", line_width=1, label="Ray Segment", opacity=1)
                plotter.add_mesh(intersection, color="blue",
                                 point_size=5, label="Intersection Points")
                plotter.add_mesh(sensor_plane, show_edges=False, opacity=1, color="r")

        # # else:
        # #     plotter.add_mesh(sensor_plane, show_edges=True, opacity=0.75, color="white")
        #
            dis_point_to_cam = np.sqrt(np.sum((start - stop)**2))/1000
            print("新的半径", dis_point_to_cam)
            # print("隐点移除原始数据单位", np.asarray(pcd.points))
            _, pt_map = pcd.hidden_point_removal(cam_loc[i] / 1000, 1000*dis_point_to_cam)

            if j in pt_map:
                sphere = pv.Sphere(radius=1000, center=cam_loc[i])
                print(len(pt_map) / len(points_coor), "可视百分比")
                print(cam_loc[i] / 1000, dis_point_to_cam, "重要参数")
                ray = pv.Line(start, stop)
                intersection = pv.PolyData(point_cam)
                plotter.add_mesh(sphere, color="red", opacity=1)
                plotter.add_mesh(ray, color="green", line_width=1, label="Ray Segment", opacity=1)
                plotter.add_mesh(intersection, color="blue",
                                 point_size=15, label="Intersection Points")
                plotter.add_mesh(sensor_plane, show_edges=False, opacity=1, color="green")

    _ = plotter.add_axes(box=True)

    plotter.show()

