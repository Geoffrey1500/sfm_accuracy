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
# print(point_data)
print(len(point_data))

plotter = pv.Plotter()

mesh_tower = pv.read("Low_LoD.ply")
mesh_tower.scale([1000, 1000, 1000])

plotter.add_mesh(mesh_tower, show_edges=True, color="white")

pcd = o3d.io.read_point_cloud("Low_LoD.ply")
diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
radius = diameter * 1000


points_coor = np.asarray(pcd.points)*1000
points_color = np.asarray(pcd.colors)

interested_id = 1000
for j in range(1):
    start = points_coor[interested_id]
    pcd_copy = copy.deepcopy(pcd)
    print(np.asarray(pcd_copy.points))

    for i in range(len(euler_ang)):
        rot_mat = R.from_euler('ZXY', [euler_ang[i]], degrees=True)
        rotated_data = rot_mat.apply(point_data)
        data_after = rotated_data + cam_loc[i]

        cloud = pv.PolyData(data_after)
        sensor_plane = cloud.delaunay_2d()

        stop = cam_loc[i]

        ref = (start + stop) / 2
        s_f = np.ones(3)*(1 + 5/np.linalg.norm(start - stop))
        start_new = scale_with_ref(s_f, ref, start)
        stop_new = scale_with_ref(s_f, ref, stop)

        # Perform ray trace
        point_cam, ind_cam = sensor_plane.ray_trace(start_new, stop)
        point_main, ind_main = mesh_tower.ray_trace(start_new, stop)
        # print(points)

        if point_cam.size and point_main.size and len(point_main) <= 1:
        # if point_cam.size and not point_main.size:
            ray = pv.Line(start, stop)
            intersection = pv.PolyData(point_cam)
            plotter.add_mesh(ray, color="r", line_width=1, label="Ray Segment", opacity=0.75)
            plotter.add_mesh(intersection, color="blue",
                             point_size=15, label="Intersection Points")
            plotter.add_mesh(sensor_plane, show_edges=True, opacity=0.75, color="r")

            _, pt_map = pcd_copy.hidden_point_removal(cam_loc[i] / 1000, radius)
            # # pcd_new = pcd_copy.select_by_index(pt_map)
            # pcd_new = points_coor[pt_map]
            print(len(pt_map) / len(points_coor), "可视百分比")
            # index_help = np.where(np.sum(np.absolute(np.asarray(pcd_new.points)*1000 - start), axis=1) <= 10 ** (-1))[0]
            # print(index_help, "查看索引")

            # if index_help.size:
            # if (pcd_new == start).any():
            if interested_id in pt_map:
                print(cam_loc[i] / 1000, radius, "重要参数")
                ray = pv.Line(start, stop)
                intersection = pv.PolyData(point_cam)
                plotter.add_mesh(ray, color="green", line_width=1, label="Ray Segment", opacity=0.75)
                plotter.add_mesh(intersection, color="blue",
                                 point_size=15, label="Intersection Points")
                plotter.add_mesh(sensor_plane, show_edges=True, opacity=0.75, color="green")


_ = plotter.add_axes(box=True)

plotter.show()


