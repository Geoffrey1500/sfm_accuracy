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

'''
Step 1: visibility check
Step 2: cone generation
Step 3: gaussian weighted euclidean distance calc
'''


def sensor_plane_point(points_per_side_=2, scale_factor_=300):
    # To create sensor plane

    x_ = np.linspace(-w/2*scale_factor_, w/2*scale_factor_, 3*points_per_side_)
    y_ = np.linspace(-h/2*scale_factor_, h/2*scale_factor_, 2*points_per_side_)
    x_, y_ = np.meshgrid(x_, y_)
    z_ = (np.zeros_like(x_) - f)*scale_factor_

    points_in_sensor_ = np.hstack((x_.reshape((-1, 1)), y_.reshape((-1, 1)),  z_.reshape((-1, 1))))

    return points_in_sensor_

# plotter = pv.Plotter()


rot_mat_set = R.from_euler('ZXY', euler_ang, degrees=True)

mesh_tower = pv.read("Low_LoD.ply")
mesh_tower.scale([1000, 1000, 1000])

plotter = pv.Plotter()
plotter.add_mesh(mesh_tower, show_edges=True, color="white")


pcd = o3d.io.read_point_cloud("Low_LoD.ply")

points_coor = np.asarray(pcd.points)*1000
points_color = np.asarray(pcd.colors)

for j in np.arange(1400, 1405):
    start = points_coor[j]

    for i in range(len(euler_ang)):
        rotated_sensor_plane = rot_mat_set[i].apply(sensor_plane_point())
        final_sensor_plane = rotated_sensor_plane + cam_loc[i]

        cloud = pv.PolyData(final_sensor_plane)
        sensor_plane = cloud.delaunay_2d()

        stop = cam_loc[i]

        # Perform ray trace for sensor plane
        point_cam, ind_cam = sensor_plane.ray_trace(start, stop)

        if ind_cam.size:
            # Perform hidden point removal from camera viewpoint
            dis_point_to_cam = np.sqrt(np.sum((start - stop)**2))/1000
            print("新的半径", dis_point_to_cam)
            # print("隐点移除原始数据单位", np.asarray(pcd.points))
            _, pt_map = pcd.hidden_point_removal(cam_loc[i] / 1000, 550*dis_point_to_cam)

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


