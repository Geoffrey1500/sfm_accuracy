import cv2
import pyvista as pv
import numpy as np
import pandas as pd
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
print(np.gcd(resol_x, resol_y))
data = pd.read_csv("internal_and_external_parameters_4.csv")
print(data.head(5))
print(data.index)
cam_loc = data[["x", "y", "z"]].values*1000
euler_ang = data[["pitch", "roll", "heading"]].values - np.array([[180, 180, 0]])


def tran_matrix_builder(rot_, tran_):
    r_ = R.from_euler('ZYX', [rot_], degrees=True)
    r_matrix_ = r_[0].as_matrix()

    tran_ = tran_.reshape((-1, 1))

    tra_mat_ = np.vstack(((np.hstack((r_matrix_, tran_))), np.array([[0., 0., 0., 1.]])))

    return tra_mat_


transform_matrix = tran_matrix_builder(np.array([0, 30, 0]), np.array([0, 0, 0]))
print("output test")
print(transform_matrix)


# To create sensor plane
n_points = 2
scale_factor = 500
X1 = np.linspace(-w/2*scale_factor, w/2*scale_factor, 3*n_points)
Y1 = np.linspace(-h/2*scale_factor, h/2*scale_factor, 2*n_points)
X, Y = np.meshgrid(X1, Y1)
Z = np.zeros_like(X) - 8.8

X_reshaped = np.reshape(X, (-1, 1))
Y_reshaped = np.reshape(Y, (-1, 1))
Z_reshaped = np.reshape(Z, (-1, 1))

point_data = np.hstack((X_reshaped, Y_reshaped, Z_reshaped))
print(point_data)
print(len(point_data))

cloud = pv.PolyData(point_data)
surf = cloud.delaunay_2d()
# surf = surf.transform(transform_matrix)


plotter = pv.Plotter()
for i in range(len(euler_ang)):
    tran_mat = tran_matrix_builder(euler_ang[i], cam_loc[i])
    cam_i = surf.copy()
    cam_i = cam_i.transform(tran_mat)

    plotter.add_mesh(cam_i, show_edges=True)


# sphere = pv.Sphere(radius=1000)

# plotter.add_mesh(sphere, show_edges=True)


mesh_tower = pv.read("Model.ply")
mesh_tower.scale([1000, 1000, 1000])
plotter.add_mesh(mesh_tower, show_edges=True, color="white")

arrow_scale = 500
arrow_x = pv.Arrow(direction=(1., 0., 0.))
arrow_x.scale([arrow_scale, arrow_scale, arrow_scale])
arrow_y = pv.Arrow(direction=(0., 1., 0.))
arrow_y.scale([arrow_scale, arrow_scale, arrow_scale])
arrow_z = pv.Arrow(direction=(0., 0., 1.))
arrow_z.scale([arrow_scale, arrow_scale, arrow_scale])


plotter.add_mesh(arrow_x, show_edges=True, color="r")
plotter.add_mesh(arrow_y, show_edges=True, color="g")
plotter.add_mesh(arrow_z, show_edges=True, color="b")

_ = plotter.add_axes(box=True)
plotter.show()

