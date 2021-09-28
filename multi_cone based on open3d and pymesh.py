import open3d as o3d
import numpy as np
import pymesh
from copy import deepcopy
import pyvista


# camera_location = np.array([[0, 3, 4],
#                             [0, -3, 4],
#                             [2, 0, 4],
#                             [-1.2, 0, 10]])
np.random.seed(0)
number_of_points = 10
camera_location = np.random.randint(-10, 10, (number_of_points, 2))
height = np.ones((number_of_points, 1)) * 10
# print(height)
noise = np.random.random((number_of_points, 3))*3
camera_location = np.hstack((camera_location, height))
camera_location += noise
print(camera_location)

point_location = np.array([2.1, 3.1, 1.1])

height_set = np.sqrt(np.sum((camera_location - point_location)**2, axis=1))*2.1
direction_set = camera_location - point_location
print(direction_set)
center_set = (camera_location + point_location)/2

rotation_set = np.random.randn(number_of_points, 3)


for i in range(len(camera_location)):
    if i == 0:
        cone1 = o3d.geometry.TriangleMesh.create_cone(height=height_set[i], resolution=50)
        R = cone1.get_rotation_matrix_from_xyz(rotation_set[i])
        cone1.rotate(R, center=(0, 0, 0))
        cone1.translate((0, 0, 0))
        # cone1.scale(2, center=(0, 0, 0))

        cone2 = o3d.geometry.TriangleMesh.create_cone(height=height_set[i + 1], resolution=50)
        R = cone2.get_rotation_matrix_from_xyz(rotation_set[i + 1])
        cone2.rotate(R, center=(0, 0, 0))
        cone2.translate((0, 0, 0))

        meshA = pymesh.form_mesh(np.asarray(cone1.vertices), np.asarray(cone1.triangles))
        meshB = pymesh.form_mesh(np.asarray(cone2.vertices), np.asarray(cone2.triangles))
        output_mesh = pymesh.boolean(meshA, meshB, operation="intersection", engine="igl")

        # mesh1 = cone1.triangulate()
        # mesh2 = cone2.triangulate()
        #
        # mesh_A = pymesh.form_mesh(np.asarray(mesh1.vectors), np.asarray(mesh1.faces))
        # mesh_B = pymesh.form_mesh(np.asarray(mesh1.vectors), np.asarray(mesh1.faces))
        #
        # initial_core = pymesh.boolean(mesh_A, mesh_B, operation="intersection", engine="igl")
        # core_after = initial_core.copy()
        # # initial_core = points.delaunay_2d()

    if i > 1:
        print("working on " + str(i))
        cone3 = o3d.geometry.TriangleMesh.create_cone(height=height_set[i], resolution=50)
        R = cone3.get_rotation_matrix_from_xyz(rotation_set[i])
        cone3.rotate(R, center=(0, 0, 0))
        cone3.translate((0, 0, 0))

        print("before")
        meshC = pymesh.form_mesh(np.asarray(cone3.vertices), np.asarray(cone3.triangles))
        # print(initial_core.points)

        output_mesh = pymesh.boolean(output_mesh, meshC, operation="intersection", engine="igl")

        # points = initial_core.points
        # faces = initial_core.faces
        # initial_core = pv.PolyData(points, faces)
        # initial_core = initial_core.clean()

        print("after")
        # print(initial_core.is_all_triangles)
        # print(type(initial_core))


points = np.asarray(output_mesh.vertices)
faces = np.asarray(output_mesh.faces)


print(points)
print(faces)

# faces = [[0, 1, 2]]
mesh = pyvista.make_tri_mesh(points, faces)
# mesh = pyvista.wrap(tmesh)
mesh.plot(show_edges=True, line_width=5)

