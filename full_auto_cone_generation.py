import open3d as o3d
import numpy as np


camera_location = np.array([[0, 3, 10],
                            [0, -3, 11],
                            [2, 0, 12],
                            [-1.2, 0, 9]])

target_point = np.array([1, 1, 1])

z_axis = np.array([0, 0, 1])

# constant_ = 3*sigma/point_in_camera


def useful_tools(cam_, target_, axis_, scale_=2, cons_=0.02, resolution=50):
    vector_ = cam_ - target_
    r_theta = np.arccos(np.dot(vector_, axis_)/(np.linalg.norm(axis_) * np.linalg.norm(vector_)))
    r_axis = np.cross(axis_, vector_)
    r_axis = r_axis/np.linalg.norm(r_axis)
    # print(R_theta/np.pi*180, R_axis)

    qw_ = np.cos(r_theta / 2)
    qx_ = r_axis[0] * np.sin(r_theta/2)
    qy_ = r_axis[1] * np.sin(r_theta/2)
    qz_ = r_axis[2] * np.sin(r_theta/2)

    height_ = np.linalg.norm(vector_)*scale_

    tran_1_ = [0, 0, -0.5*height_]
    tran_2_ = target_

    radius_ = cons_*0.5*height_

    cone_ = o3d.geometry.TriangleMesh.create_cone(radius=radius_, height=height_, resolution=resolution)
    r_ = cone_.get_rotation_matrix_from_quaternion([qw_, qx_, qy_, qz_])
    cone_.translate(tran_1_)
    cone_.rotate(r_, center=(0, 0, 0))
    cone_.translate(tran_2_)

    return cone_



ball_in_center = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
core_set = [ball_in_center, mesh_frame]
# ball_set = []

for i in range(len(camera_location)):
    cone1 = useful_tools(camera_location[i], target_point, z_axis)
    core_set.append(cone1)

    ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    ball.translate(camera_location[i])
    core_set.append(ball)

o3d.visualization.draw_geometries(core_set)

# ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
# ball.translate(target_point)
#
# ball2 = o3d.geometry.TriangleMesh.create_sphere(radius=1)
# ball2.translate((0, 0, -5))
#
# cone_2 = o3d.geometry.TriangleMesh.create_cone(radius=1.0, height=12, resolution=50)
# R = cone_2.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
# cone_2.translate((0, 0, -5))
# # cone_2.rotate(R, center=(0, 0, 0))
# cone_2.translate((0, 0, 0))
#
#
# mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
#
# o3d.visualization.draw_geometries([cone_2, mesh_frame, ball, ball2])


# for i in range(len(camera_location)):
#     if i == 0:
#         cone1 = o3d.geometry.TriangleMesh.create_cone(height=height_set[i], resolution=50)
#         R = cone1.get_rotation_matrix_from_xyz(rotation_set[i])
#         cone1.rotate(R, center=(0, 0, 0))
#         cone1.translate((0, 0, 0))
#         # cone1.scale(2, center=(0, 0, 0))
#
#         cone2 = o3d.geometry.TriangleMesh.create_cone(height=height_set[i + 1], resolution=50)
#         R = cone2.get_rotation_matrix_from_xyz(rotation_set[i + 1])
#         cone2.rotate(R, center=(0, 0, 0))
#         cone2.translate((0, 0, 0))
#
#         meshA = pymesh.form_mesh(np.asarray(cone1.vertices), np.asarray(cone1.triangles))
#         meshB = pymesh.form_mesh(np.asarray(cone2.vertices), np.asarray(cone2.triangles))
#         output_mesh = pymesh.boolean(meshA, meshB, operation="intersection", engine="igl")
#
#         # mesh1 = cone1.triangulate()
#         # mesh2 = cone2.triangulate()
#         #
#         # mesh_A = pymesh.form_mesh(np.asarray(mesh1.vectors), np.asarray(mesh1.faces))
#         # mesh_B = pymesh.form_mesh(np.asarray(mesh1.vectors), np.asarray(mesh1.faces))
#         #
#         # initial_core = pymesh.boolean(mesh_A, mesh_B, operation="intersection", engine="igl")
#         # core_after = initial_core.copy()
#         # # initial_core = points.delaunay_2d()
#
#     if i > 1:
#         print("working on " + str(i))
#         cone3 = o3d.geometry.TriangleMesh.create_cone(height=height_set[i], resolution=50)
#         R = cone3.get_rotation_matrix_from_xyz(rotation_set[i])
#         cone3.rotate(R, center=(0, 0, 0))
#         cone3.translate((0, 0, 0))
#
#         print("before")
#         meshC = pymesh.form_mesh(np.asarray(cone3.vertices), np.asarray(cone3.triangles))
#         # print(initial_core.points)
#
#         output_mesh = pymesh.boolean(output_mesh, meshC, operation="intersection", engine="igl")
#
#         # points = initial_core.points
#         # faces = initial_core.faces
#         # initial_core = pv.PolyData(points, faces)
#         # initial_core = initial_core.clean()
#
#         print("after")
#         # print(initial_core.is_all_triangles)
#         # print(type(initial_core))


# points = np.asarray(output_mesh.vertices)
# faces = np.asarray(output_mesh.faces)
#
#
# print(points)
# print(faces)
#
# # faces = [[0, 1, 2]]
# mesh = pyvista.make_tri_mesh(points, faces)
# # mesh = pyvista.wrap(tmesh)
# mesh.plot(show_edges=True, line_width=5)
