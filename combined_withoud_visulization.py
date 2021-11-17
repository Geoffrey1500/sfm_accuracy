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
    print("what happend")
    print(tran_2_)
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


rot_mat_set = R.from_euler('ZXY', euler_ang, degrees=True)

mesh_tower = pv.read("Low_LoD.ply")
mesh_tower.scale([1000, 1000, 1000])



pcd = o3d.io.read_point_cloud("Low_LoD.ply")

points_coor = np.asarray(pcd.points)*1000
points_color = np.asarray(pcd.colors)

for j in np.arange(900, 920):
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh_tower, show_edges=True, color="white")

    start = points_coor[j]
    cam_index = []

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
            # print("新的半径", dis_point_to_cam)
            # print("隐点移除原始数据单位", np.asarray(pcd.points))
            _, pt_map = pcd.hidden_point_removal(cam_loc[i] / 1000, 550*dis_point_to_cam)

            if j in pt_map:
                # sphere = pv.Sphere(radius=1000, center=cam_loc[i])
                # print(len(pt_map) / len(points_coor), "可视百分比")
                # print(cam_loc[i] / 1000, dis_point_to_cam, "重要参数")
                # ray = pv.Line(start, stop)
                # intersection = pv.PolyData(point_cam)
                # plotter.add_mesh(sphere, color="red", opacity=1)
                # plotter.add_mesh(ray, color="green", line_width=1, label="Ray Segment", opacity=1)
                # plotter.add_mesh(intersection, color="blue",
                #                  point_size=15, label="Intersection Points")
                # plotter.add_mesh(sensor_plane, show_edges=False, opacity=1, color="green")

                cam_index.append(i)

    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    # core_set = [mesh_frame]

    camera_location = cam_loc[cam_index]
    # print("Print Camera location")
    # print(camera_location)
    #
    # for k in range(len(camera_location)):
    #     cone1 = useful_tools(camera_location[k], points_coor[j], z_axis)
    #     core_set.append(cone1)
    #
    #     ball = o3d.geometry.TriangleMesh.create_sphere(radius=1000)
    #     ball.translate(camera_location[k])
    #     core_set.append(ball)
    #
    # o3d.visualization.draw_geometries(core_set)

    # _ = plotter.add_axes(box=True)
    #
    # plotter.show()

    cone1 = useful_tools(camera_location[0], points_coor[j], z_axis)
    meshA = pymesh.form_mesh(np.asarray(cone1.vertices), np.asarray(cone1.triangles))

    start = time.process_time()
    for v in range(len(camera_location) - 1):
        print("working on " + str(v + 1))
        cone2 = useful_tools(camera_location[v + 1], points_coor[j], z_axis)
        meshB = pymesh.form_mesh(np.asarray(cone2.vertices), np.asarray(cone2.triangles))

        if v % 10 == 0 and v != 0:
            # release RAM per 50 round
            # print("I am delete sometion")
            clean_mesh = fix_mesh(meshA, detail="low")
            pymesh.save_mesh("meshA.obj", clean_mesh)
            pymesh.save_mesh("meshB.obj", meshB)
            del meshA, meshB
            gc.collect()
            meshA = pymesh.load_mesh("meshA.obj")
            meshB = pymesh.load_mesh("meshB.obj")
        # clean_mesh = fix_mesh(meshA, detail="low")
        # clean_mesh = meshA

        meshA = pymesh.boolean(meshA, meshB, operation="intersection", engine="corefinement")
        # output_mesh = clean_mesh
        print("after")

        if v == len(camera_location):
            # print("I am the last one")
            output_mesh = fix_mesh(meshA, detail="low")

    end = time.process_time()

    points = np.asarray(meshA.vertices)
    faces = np.asarray(meshA.faces)

    print(len(points))
    print(faces)
    print("一共模拟 " + str(len(cam_index)) + " 个点")
    print("相交体积为：" + str(meshA.volume) + "mm^3")
    print("共运行：" + str(end - start) + "s")

    # # faces = [[0, 1, 2]]
    # mesh = pv.make_tri_mesh(points, faces)
    # # mesh = pyvista.wrap(tmesh)
    # mesh.plot(show_edges=True, line_width=1)


