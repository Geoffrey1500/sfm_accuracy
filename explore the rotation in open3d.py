import open3d as o3d
import numpy as np


z_axis = np.array([0, 0, 1])

target_point = np.array([3, 4, 5])

camera_location = np.array([[0, 3, 4],
                            [0, -3, 4],
                            [2, 0, 4],
                            [-1.2, 0, 10]])

R_theta = np.arccos(z_axis.dot(target_point)/(np.linalg.norm(z_axis) * np.linalg.norm(target_point)))
R_axis = np.cross(z_axis, target_point)
R_axis = R_axis/np.linalg.norm(R_axis)

print(R_theta/np.pi*180, R_axis)

qx = R_axis[0] * np.sin(R_theta/2)
qy = R_axis[1] * np.sin(R_theta/2)
qz = R_axis[2] * np.sin(R_theta/2)
qw = np.cos(R_theta/2)

print(qw, qx, qy, qz)

ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
ball.translate(target_point)

ball2 = o3d.geometry.TriangleMesh.create_sphere(radius=1)
ball2.translate((0, 0, -5))

cone_2 = o3d.geometry.TriangleMesh.create_cone(radius=1.0, height=12, resolution=50)
R = cone_2.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
cone_2.translate((0, 0, -5))
# cone_2.rotate(R, center=(0, 0, 0))
cone_2.translate((0, 0, 0))


mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])

o3d.visualization.draw_geometries([cone_2, mesh_frame, ball, ball2])
