import cv2
import pyvista as pv
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# # Define a simple Gaussian surface
# pcd = o3d.io.read_point_cloud("25M30Ds_A.pcd")
# points = np.asarray(pcd.points)
# print(points)
#
# plane = pv.Plane()
#
# # # Get the points as a 2D NumPy array (N by 3)
# # points = a[:, 0:3]
#
# p = pv.Plotter(shape=(1, 2))
#
# # simply pass the numpy points to the PolyData constructor
# cloud = pv.PolyData(points)
# surf = cloud.delaunay_2d()
# p.subplot(0, 0)
# p.add_mesh(surf, color="tan")
#
# mesh = pv.read("25M30Ds_A.ply")
# p.subplot(0, 1)
# p.add_mesh(mesh, color="tan")
# p.show()
# plotter = pv.Plotter(off_screen=True)
# plotter.add_mesh(mesh)
# plotter.show(screenshot="myscreenshot.png")

'''
大疆Phantom 4 Pro
传感器大小：1英寸 13.2 mm x 8.8 mm
分辨率：5472×3648
像元大小：2.4123 um
焦距：8.8 mm
FOV：84°？

'''
print(np.arctan((13.2/2)/8.8)/np.pi*180*2)
data = pd.read_csv("internal_and_external_parameters.csv")
print(data.head(5))
cam_loc = data[["x", "y", "alt"]].values
euler_ang = data[["heading", "pitch", "roll"]].values
distCoeffs = data[["k1", "k2", "t1", "t2", "k3"]].values

cali_file = np.load("Phantom.npz")
mtx, dist = cali_file["mtx"], cali_file["dist"]
rvecs, tvecs = cali_file["rvecs"], cali_file["tvecs"]

# print(euler_ang)
# print(cam_loc)
# print(data.head(5))

X1 = np.arange(-10000, 15000, 5000.5)
Y1 = np.arange(-10000, 15000, 5000.5)
X, Y = np.meshgrid(X1, Y1)
noise = np.random.random(X.shape)
Z = np.zeros_like(X) + noise

X_reshaped = np.reshape(X, (-1, 1))
Y_reshaped = np.reshape(Y, (-1, 1))
Z_reshaped = np.reshape(Z, (-1, 1))

data_original_1 = np.hstack((X_reshaped, Y_reshaped, Z_reshaped))
print(data_original_1)
print(len(data_original_1))

r = R.from_euler('yxz', [[0, 0, 0]], degrees=True)
r_matrix = r[0].as_matrix()
print(r_matrix)

print(cam_loc[0])
print(cam_loc[0].reshape((-1, 1)))

r_rvecs = cv2.Rodrigues(r_matrix)[0]
print(r_rvecs)

imgpoints2, _ = cv2.projectPoints(data_original_1, r_rvecs, np.array([[0.1, 0.1, 100000.1]]).T, mtx, dist)


# print(imgpoints2)
print(imgpoints2.shape)
print(imgpoints2[:, 0, :])
def cam_proj():

    return 0


print(10/np.tan(70/180*np.pi))
