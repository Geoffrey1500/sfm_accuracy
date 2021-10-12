import pyvista as pv
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
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
euler_ang = data[["heading", "pitch", "roll"]].values * np.array([[-1, 1, 1]])


def polyhull(data_set):
    hull = ConvexHull(data_set)
    faces = np.column_stack((3*np.ones((len(hull.simplices), 1), dtype=int), hull.simplices)).flatten()
    poly = pv.PolyData(hull.points, faces)
    return poly


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

point_data = np.hstack((X_reshaped, Y_reshaped, Z_reshaped))
point_data = np.vstack((point_data, np.array([0, 0, 0])))
print(point_data)
print(len(point_data))

plotter = pv.Plotter()
cloud1 = pv.PolyData(point_data)
surf1 = cloud1.delaunay_2d()
plotter.add_mesh(surf1, show_edges=True)


for i in range(len(euler_ang)):
    rot_mat = R.from_euler('ZXY', [euler_ang[i]], degrees=True)
    rotated_data = rot_mat.apply(point_data)
    data_after = rotated_data + cam_loc[i]

    # cloud = pv.PolyData(data_after)
    # surf = cloud.delaunay_2d()
    surf = polyhull(data_after)

    plotter.add_mesh(surf, show_edges=True)


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
