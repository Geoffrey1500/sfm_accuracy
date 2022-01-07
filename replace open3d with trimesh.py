import pyvista as pv
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import time
from numpy.linalg import norm
import trimesh
import gc


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

    cone_.translate(tran_2_)

    return cone_


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
data = pd.read_csv("data/internal_and_external_parameters_5.csv")
print(data.head(5))
cam_loc = data[["x", "y", "z"]].values*1000
euler_ang = data[["heading", "pitch", "roll"]].values * np.array([[-1, 1, 1]]) + np.array([[0, 0, 0]])
z_axis = np.array([0, 0, 1])

'''
Step 1: visibility check
Step 2: cone generation
Step 3: gaussian weighted euclidean distance calc
'''

rot_mat_set = R.from_euler('ZXY', euler_ang, degrees=True)

mesh_tower = pv.read("data/1_7_2.ply")
mesh_tower.scale([1000, 1000, 1000])

pcd = o3d.io.read_point_cloud("data/1_7_2.ply")

mesh_for_trimesh = trimesh.load("data/1_7_2.glb", force='mesh')
trimesh_points = mesh_for_trimesh.vertices
trimesh_triangle = mesh_for_trimesh.triangles
vertices_normal = mesh_for_trimesh.vertex_normals

model_normal = pd.read_csv('data/1_7_2.xyz', header=None, sep=' ').to_numpy()[:, 3:6]
mesh_for_trimesh.vertex_normals = model_normal

vertices_normal_after = mesh_for_trimesh.vertex_normals

points_coor = np.asarray(pcd.points)*1000
points_color = np.asarray(pcd.colors)

for j in np.arange(1024, 2058):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_tower, show_edges=True, color="white")

    start = points_coor[j]
    coneA = 0.00001
    v = 0
    start_time = time.process_time()

    for i in range(len(euler_ang)):
        rotated_sensor_plane = rot_mat_set[i].apply(sensor_plane_point())
        final_sensor_plane = rotated_sensor_plane + cam_loc[i]

        cloud = pv.PolyData(final_sensor_plane)
        sensor_plane = cloud.delaunay_2d()

        stop = cam_loc[i]

        # Perform ray trace for sensor plane
        point_cam, ind_cam = sensor_plane.ray_trace(start, stop)

        if ind_cam.size:
            ray_dire = (start-stop)/1000

            # 判断 ray 和 mesh 第一个相交点是否为目标点
            index_tri = mesh_for_trimesh.ray.intersects_first(
                ray_origins=[stop/1000],
                ray_directions=[ray_dire])

            tri_queried = trimesh_triangle[index_tri]
            points_wanted = start/1000

            dis_check = tri_queried[0] - points_wanted
            sum_dis_check = np.sum(dis_check**2, axis=1)

            if np.any(sum_dis_check <= 0.0001):
                sphere = pv.Sphere(radius=1000, center=cam_loc[i])
                ray = pv.Line(start, stop)
                intersection = pv.PolyData(point_cam)
                plotter.add_mesh(sphere, color="red", opacity=1)
                plotter.add_mesh(ray, color="green", line_width=1, label="Ray Segment", opacity=1)
                plotter.add_mesh(intersection, color="blue",
                                 point_size=15, label="Intersection Points")
                plotter.add_mesh(sensor_plane, show_edges=False, opacity=1, color="green")

                # 初始化第一个圆锥，并暂时跳出循环
                if coneA == 0.00001:
                    coneA = useful_tools(cam_loc[i], points_coor[j], z_axis)
                    meshA = trimesh.Trimesh(vertices=np.asarray(coneA.vertices), faces=np.asarray(coneA.triangles))
                    print("new round started")

                    continue

                print("working on " + str(v + 1))
                coneB = useful_tools(cam_loc[i], points_coor[j], z_axis)
                meshB = trimesh.Trimesh(vertices=np.asarray(coneB.vertices), faces=np.asarray(coneB.triangles))

                meshA = trimesh.boolean.intersection([meshA, meshB], engine='scad')
                meshA = trimesh.convex.convex_hull(meshA, qhull_options='Qt')

                v += 1

    end_time = time.process_time()

    _ = plotter.add_axes(box=True)

    plotter.show()

    if v != 0:
        points = np.asarray(meshA.vertices)
        faces = np.asarray(meshA.faces)

        print(len(points))
        # print(faces)
        print("一共模拟 " + str(v) + " 个点")
        print("总共有 " + str(v/len(cam_loc)*100) + "% 的相机参与运算")
        print("相交体积为：" + str(meshA.volume) + "mm^3")
        print("共运行：" + str(end_time - start_time) + "s")

        # faces = [[0, 1, 2]]
        mesh = pv.make_tri_mesh(points, faces)
        # mesh = pyvista.wrap(tmesh)
        mesh.plot(show_edges=True, line_width=1)

        del meshA, meshB
        gc.collect()

    else:
        print('no intersection')
