import open3d as o3d
import numpy as np

print("Testing mesh in Open3D...")

cone_1 = o3d.geometry.TriangleMesh.create_cone(height=80.0)
print('Vertices:')
print(np.asarray(cone_1.vertices))

a = np.asarray(cone_1.vertices)
b = np.sum(a**2, axis=1)
print(b)

cone_2 = o3d.geometry.TriangleMesh.create_cone(height=80.0)
R = cone_2.get_rotation_matrix_from_xyz((np.pi / 9, 0, np.pi / 4))
cone_2.rotate(R, center=(0, 0, 0))

cone_3 = o3d.geometry.TriangleMesh.create_cone(height=80.0)
R = cone_3.get_rotation_matrix_from_xyz((np.pi / 9, np.pi / 9, np.pi / 4))
cone_3.rotate(R, center=(0, 0, 0))

o3d.visualization.draw_geometries([cone_1, cone_2, cone_3])