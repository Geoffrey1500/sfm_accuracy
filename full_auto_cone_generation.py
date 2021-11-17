import open3d as o3d
import numpy as np
import pymesh
import pyvista
import time
from numpy.linalg import norm
import gc


# np.random.seed(0)
number_of_points = 100
camera_location = np.random.randint(-60, 80, (number_of_points, 2))
camera_height = np.ones((number_of_points, 1)) * 50
# print(height)
noise = np.random.random((number_of_points, 3))*3
camera_location = np.hstack((camera_location, camera_height))
camera_location += noise

target_point = np.array([1, 1, 1])

z_axis = np.array([0, 0, 1])

# constant_ = 3*sigma/point_in_camera


def useful_tools(cam_, target_, axis_, scale_=2, cons_=0.02, resolution=6):
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


def fix_mesh(mesh, detail="normal"):
    bbox_min, bbox_max = mesh.bbox
    diag_len = norm(bbox_max - bbox_min)
    if detail == "normal":
        target_len = diag_len * 5e-3
    elif detail == "high":
        target_len = diag_len * 2.5e-3
    elif detail == "low":
        target_len = diag_len * 1e-2
    print("Target resolution: {} mm".format(target_len))

    mesh, __ = pymesh.remove_degenerated_triangles(mesh)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    mesh, __ = pymesh.collapse_short_edges(mesh, target_len)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh)

    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)

    return mesh


ball_in_center = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
core_set = [ball_in_center, mesh_frame]
ball_set = []

for i in range(len(camera_location)):
    cone1 = useful_tools(camera_location[i], target_point, z_axis)
    core_set.append(cone1)

    ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
    ball.translate(camera_location[i])
    core_set.append(ball)

o3d.visualization.draw_geometries(core_set)

cone1 = useful_tools(camera_location[0], target_point, z_axis)

cone2 = useful_tools(camera_location[0 + 1], target_point, z_axis)

meshA = pymesh.form_mesh(np.asarray(cone1.vertices), np.asarray(cone1.triangles))
meshB = pymesh.form_mesh(np.asarray(cone2.vertices), np.asarray(cone2.triangles))
output_mesh = pymesh.boolean(meshA, meshB, operation="intersection", engine="corefinement")
del meshA, meshB
gc.collect()


start = time.process_time()
for i in range(len(camera_location)):
    if i > 1:
        print("working on " + str(i))
        cone3 = useful_tools(camera_location[i], target_point, z_axis)

        print("before")
        meshC = pymesh.form_mesh(np.asarray(cone3.vertices), np.asarray(cone3.triangles))

        if i % 50 == 0 and i != len(camera_location)-1:
            print("I am delete sometion")
            clean_mesh = fix_mesh(output_mesh, detail="low")
            pymesh.save_mesh("output_mesh.obj", clean_mesh)
            pymesh.save_mesh("meshC.obj", meshC)
            del output_mesh, meshC
            gc.collect()
            output_mesh = pymesh.load_mesh("output_mesh.obj")
            meshC = pymesh.load_mesh("meshC.obj")
        # clean_mesh = fix_mesh(output_mesh, detail="low")
        # clean_mesh = output_mesh
        output_mesh = pymesh.boolean(output_mesh, meshC, operation="intersection", engine="corefinement")
        # output_mesh = clean_mesh
        print("after")

        if i == len(camera_location)-1:
            print("I am the last one")
            output_mesh = fix_mesh(output_mesh, detail="low")

end = time.process_time()

points = np.asarray(output_mesh.vertices)
faces = np.asarray(output_mesh.faces)

print(len(points))
print(faces)
print("一共模拟 " + str(number_of_points) + " 个点")
print("相交体积为：" + str(output_mesh.volume) + "mm^3")
print("共运行：" + str(end - start) + "s")

# faces = [[0, 1, 2]]
mesh = pyvista.make_tri_mesh(points, faces)
# mesh = pyvista.wrap(tmesh)
mesh.plot(show_edges=True, line_width=1)
