import pyvista as pv
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation as R
import pymesh
import time
from numpy.linalg import norm
import gc


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
euler_ang = data[["heading", "pitch", "roll"]].values * np.array([[-1, 1, 1]]) + np.array([[0, 0, 0]])
z_axis = np.array([0, 0, 1])

'''
Step 1: visibility check
Step 2: cone generation
Step 3: gaussian weighted euclidean distance calc
'''


def sensor_plane_point(points_per_side_=2, scale_factor_=300):
    # To create sensor plane

    x_ = np.linspace(-w/2*scale_factor_, w/2*scale_factor_, 3*points_per_side_)
    y_ = np.linspace(-h/2*scale_factor_, h/2*scale_factor_, 2*points_per_side_)
    x_, y_ = np.meshgrid(x_, y_)
    z_ = (np.zeros_like(x_) - f)*scale_factor_

    points_in_sensor_ = np.hstack((x_.reshape((-1, 1)), y_.reshape((-1, 1)),  z_.reshape((-1, 1))))

    return points_in_sensor_


def useful_tools(cam_, target_, axis_, scale_=2, cons_=0.0002, resolution=6):
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
    # print("what happend")
    # print(tran_2_)
    cone_.translate(tran_2_)

    return cone_


# def fix_mesh(mesh_input, detail="normal"):
#     bbox_min, bbox_max = mesh_input.bbox
#     diag_len = norm(bbox_max - bbox_min)
#     if detail == "normal":
#         target_len = diag_len * 5e-3
#     elif detail == "high":
#         target_len = diag_len * 2.5e-3
#     elif detail == "low":
#         target_len = diag_len * 1e-2
#     # print("Target resolution: {} mm".format(target_len))
#
#     mesh_input, __ = pymesh.remove_degenerated_triangles(mesh_input)
#     mesh_input, __ = pymesh.remove_duplicated_faces(mesh_input)
#     mesh_input, __ = pymesh.remove_isolated_vertices(mesh_input)
#
#     mesh_input, __ = pymesh.collapse_short_edges(mesh_input, target_len)
#     mesh_input, __ = pymesh.remove_obtuse_triangles(mesh_input)
#
#     mesh_input = pymesh.compute_outer_hull(mesh_input)
#     mesh_input, __ = pymesh.remove_duplicated_faces(mesh_input)
#
#     return mesh_input


def fix_mesh(mesh_input, detail="normal"):
    bbox_min, bbox_max = mesh_input.bbox
    diag_len = norm(bbox_max - bbox_min)
    if detail == "normal":
        target_len = diag_len * 5e-3
    elif detail == "high":
        target_len = diag_len * 2.5e-3
    elif detail == "low":
        target_len = diag_len * 1e-2
    print("Target resolution: {} mm".format(target_len))

    count = 0
    mesh_input, __ = pymesh.remove_degenerated_triangles(mesh_input, 100)
    mesh_input, __ = pymesh.split_long_edges(mesh_input, target_len)
    num_vertices = mesh_input.num_vertices
    while True:
        mesh_input, __ = pymesh.collapse_short_edges(mesh_input, 1e-3)
        mesh_input, __ = pymesh.collapse_short_edges(mesh_input, target_len,
                                               preserve_feature=True)
        mesh_input, __ = pymesh.remove_obtuse_triangles(mesh_input, 150.0, 100)
        if mesh_input.num_vertices == num_vertices:
            break

        num_vertices = mesh_input.num_vertices
        print("#v: {}".format(num_vertices))
        count += 1
        if count > 1: break

    mesh_input = pymesh.resolve_self_intersection(mesh_input)
    mesh_input, __ = pymesh.remove_duplicated_faces(mesh_input)
    mesh_input = pymesh.compute_outer_hull(mesh_input)
    mesh_input, __ = pymesh.remove_duplicated_faces(mesh_input)
    mesh_input, __ = pymesh.remove_obtuse_triangles(mesh_input, 179.0, 5)
    mesh_input, __ = pymesh.remove_isolated_vertices(mesh_input)

    return mesh_input


rot_mat_set = R.from_euler('ZXY', euler_ang, degrees=True)

mesh_tower = pv.read("Low_LoD.ply")
mesh_tower.scale([1000, 1000, 1000])

pcd = o3d.io.read_point_cloud("Low_LoD.ply")

points_coor = np.asarray(pcd.points)*1000
points_color = np.asarray(pcd.colors)

for j in np.arange(500, 1001):

    start = points_coor[j]
    coneA = 0.00001
    v = 0

    for i in range(len(euler_ang)):
        rotated_sensor_plane = rot_mat_set[i].apply(sensor_plane_point())
        final_sensor_plane = rotated_sensor_plane + cam_loc[i]

        cloud = pv.PolyData(final_sensor_plane)
        sensor_plane = cloud.delaunay_2d()

        stop = cam_loc[i]

        # Perform ray trace for sensor plane
        point_cam, ind_cam = sensor_plane.ray_trace(start, stop)

        if ind_cam.size:
            # Perform hidden point removal from camera viewpoint
            dis_point_to_cam = np.sqrt(np.sum((start - stop)**2))/1000
            _, pt_map = pcd.hidden_point_removal(cam_loc[i] / 1000, 550*dis_point_to_cam)

            start_time = time.process_time()
            if j in pt_map:
                sphere = pv.Sphere(radius=1000, center=cam_loc[i])
                ray = pv.Line(start, stop)
                intersection = pv.PolyData(point_cam)

                if coneA == 0.00001:
                    coneA = useful_tools(cam_loc[i], points_coor[j], z_axis)
                    meshA = pymesh.form_mesh(np.asarray(coneA.vertices), np.asarray(coneA.triangles))
                    print("new round started")
                    # v += 1
                    continue

                coneB = useful_tools(cam_loc[i], points_coor[j], z_axis)
                meshB = pymesh.form_mesh(np.asarray(coneB.vertices), np.asarray(coneB.triangles))

                meshA = pymesh.boolean(meshA, meshB, operation="intersection", engine="igl")
                if len(np.asarray(meshA.vertices)) != 0:
                    meshA = pymesh.convex_hull(meshA, engine="auto")

                v += 1

            end_time = time.process_time()

    if v != 0:
        print("Round " + str(j))
        print("一共模拟 " + str(v) + " 个点")
        print("相交体积为：" + str(meshA.volume) + "mm^3")
        print("共运行：" + str(end_time - start_time) + "s")


        del meshA, meshB
        gc.collect()
    else:
        print('no intesection')


