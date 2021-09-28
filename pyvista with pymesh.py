import pyvista as pv
import numpy as np
import pymesh


# camera_location = np.array([[0, 3, 4],
#                             [0, -3, 4],
#                             [2, 0, 4],
#                             [-1.2, 0, 10]])
np.random.seed(0)
np.random.seed(0)
number_of_points = 2
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


for i in range(len(camera_location)):
    if i == 0:
        cone1 = pv.Cone(center=center_set[i], direction=direction_set[i], height=height_set[i], radius=1,
                        capping=True, angle=None, resolution=50)
        cone2 = pv.Cone(center=center_set[i + 1], direction=direction_set[i + 1], height=height_set[i + 1], radius=1,
                        capping=True, angle=None, resolution=50)

        mesh1 = cone1.triangulate()
        mesh2 = cone2.triangulate()

        mesh_A = pymesh.form_mesh(np.asarray(mesh1.vectors), np.asarray(mesh1.faces))
        mesh_B = pymesh.form_mesh(np.asarray(mesh1.vectors), np.asarray(mesh1.faces))

        initial_core = pymesh.boolean(mesh_A, mesh_B, operation="intersection", engine="igl")
        core_after = initial_core.copy()
        # initial_core = points.delaunay_2d()

    if i > 1:
        print("working on " + str(i))
        cone3 = pv.Cone(center=center_set[i], direction=direction_set[i], height=height_set[i], radius=1,
                        capping=True, angle=None, resolution=50)
        mesh3 = cone3.triangulate()

        print("before")
        print(core_after.is_all_triangles)
        print(mesh3.is_all_triangles)
        print(type(core_after))
        # print(initial_core.points)

        initial_core = core_after.boolean_intersection(mesh3)
        core_after = initial_core.copy()

        # points = initial_core.points
        # faces = initial_core.faces
        # initial_core = pv.PolyData(points, faces)
        # initial_core = initial_core.clean()

        print("after")
        print(initial_core.is_all_triangles)
        print(type(initial_core))


print(core_after.volume)
p = pv.Plotter()
# p.add_mesh(cone1, color="tan", show_edges=True)
# p.add_mesh(cone2, color="tan", show_edges=True)
# p.add_mesh(sphere, color="tan", show_edges=True)
p.add_mesh(initial_core, color="tan")
# initial_core.plot_normals(mag=0.1)
p.show()

