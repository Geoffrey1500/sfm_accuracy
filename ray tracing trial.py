import pyvista as pv
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import open3d as o3d
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
print(point_data)
print(len(point_data))

plotter = pv.Plotter()

mesh_tower = pv.read("Model.ply")
mesh_tower.scale([1000, 1000, 1000])

plotter.add_mesh(mesh_tower, show_edges=True, color="white")

pcd = o3d.io.read_point_cloud("Model.ply")


points_coor = np.asarray(pcd.points)*1000
points_color = np.asarray(pcd.colors)

print(mesh_tower.face_normals)

for j in range(1):
    start = points_coor[1000]

    for i in range(len(euler_ang)):
        rot_mat = R.from_euler('ZXY', [euler_ang[i]], degrees=True)
        rotated_data = rot_mat.apply(point_data)
        data_after = rotated_data + cam_loc[i]

        cloud = pv.PolyData(data_after)
        surf = cloud.delaunay_2d()

        stop = cam_loc[i]

        # Perform ray trace
        point_cam, ind_cam = surf.ray_trace(start, stop)
        # print(points)

        point_main, ind_main = mesh_tower.ray_trace(start, stop)

        if point_cam.size and len(point_main) <= 1:
            # print(len(point_main))
            # Create geometry to represent ray trace
            ray = pv.Line(start, stop)
            intersection = pv.PolyData(point_cam)
            plotter.add_mesh(ray, color="r", line_width=1, label="Ray Segment", opacity=0.75)
            plotter.add_mesh(intersection, color="blue",
                             point_size=15, label="Intersection Points")
            plotter.add_mesh(surf, show_edges=True, opacity=0.75)


_ = plotter.add_axes(box=True)

plotter.show()


