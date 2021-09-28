import open3d as o3d
import numpy as np
import pymesh
from copy import deepcopy
import trimesh
import pyvista

print("Testing mesh in Open3D...")

cone_1 = o3d.geometry.TriangleMesh.create_cone(height=80.0, resolution=50)
print('Vertices:')
vertices = np.asarray(cone_1.vertices)
faces = np.asarray(cone_1.triangles)
print(vertices)
print(faces)

cone_2 = o3d.geometry.TriangleMesh.create_cone(height=80.0, resolution=50)
R = cone_2.get_rotation_matrix_from_xyz((np.pi / 9, 0, np.pi / 4))
cone_2.rotate(R, center=(0, 0, 0))
cone_2.translate((0, 0, 0))

cone_3 = o3d.geometry.TriangleMesh.create_cone(height=80.0, resolution=50)
R = cone_3.get_rotation_matrix_from_xyz((-np.pi / 9, 0, np.pi / 4))
cone_3.rotate(R, center=(0, 0, 0))
cone_3.translate((0, 0, 0))

mesh_A = pymesh.form_mesh(vertices, faces)
mesh_B = pymesh.form_mesh(np.asarray(cone_2.vertices), np.asarray(cone_2.triangles))
mesh_C = pymesh.form_mesh(np.asarray(cone_3.vertices), np.asarray(cone_3.triangles))
output_mesh = pymesh.boolean(mesh_A, mesh_B, operation="intersection", engine="igl")
output_mesh = pymesh.boolean(output_mesh, mesh_C, operation="intersection", engine="igl")

points = np.asarray(output_mesh.vertices)
faces = np.asarray(output_mesh.faces)


print(points)
print(faces)

# faces = [[0, 1, 2]]
mesh = pyvista.make_tri_mesh(points, faces)
# mesh = pyvista.wrap(tmesh)
mesh.plot(show_edges=True, line_width=5)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(deepcopy(points))
# o3d.visualization.draw_geometries([pcd])

# o3d.visualization.draw_geometries([cone_1, cone_2, cone_3])

