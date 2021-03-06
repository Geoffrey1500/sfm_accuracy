import pyvista as pv
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import time
from numpy.linalg import norm
import trimesh
import gc
from scipy.spatial import distance_matrix
from sklearn.neighbors import KDTree


def sensor_plane_point(points_per_side_=2, scale_factor_=200):
    # To create sensor plane

    x_ = np.linspace(-w/2*scale_factor_, w/2*scale_factor_, 3*points_per_side_)
    y_ = np.linspace(-h/2*scale_factor_, h/2*scale_factor_, 2*points_per_side_)
    x_, y_ = np.meshgrid(x_, y_)
    z_ = (np.zeros_like(x_) - f)*scale_factor_

    points_in_sensor_ = np.hstack((x_.reshape((-1, 1)), y_.reshape((-1, 1)),  z_.reshape((-1, 1))))

    return points_in_sensor_


def useful_tools(cam_, target_, axis_, pix_size_, focal_, scale_=2, repro_err=3, resolution=6):
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

    radius_ = repro_err*pix_size_*(0.5*height_/focal_)
    # print("圆锥的投影半径 ", radius_)

    cone_ = o3d.geometry.TriangleMesh.create_cone(radius=radius_, height=height_, resolution=resolution)
    r_ = cone_.get_rotation_matrix_from_quaternion([qw_, qx_, qy_, qz_])
    cone_.translate(tran_1_)
    cone_.rotate(r_, center=(0, 0, 0))

    cone_.translate(tran_2_)

    return cone_


def gaussian_dis(dist, sigma, mu=0):
    return (1/(sigma*np.sqrt(2*np.pi))) * np.e ** (-0.5*((dist-mu)/sigma)**2)


def vector_length(input_vector):
    return np.sqrt(np.sum((input_vector ** 2)))


def angle_between_vectors(v1, v2):
    # return np.arccos(np.dot(v1, v2) / (vector_length(v1) * vector_length(v2))) * (180 / np.pi)
    return np.dot(v1, v2) / (vector_length(v1) * vector_length(v2))


def grab_tree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

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
pixel_size = np.average([w/resol_x, h/resol_y])
data = pd.read_csv("data/camera_parameters.csv")
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

mesh_tower = pv.read("data/1_8_2.ply")
mesh_tower.scale([1000, 1000, 1000])

pcd = o3d.io.read_point_cloud("data/1_8_2.ply")

mesh_for_trimesh = trimesh.load("data/1_8_2.glb", force='mesh')
trimesh_points = mesh_for_trimesh.vertices*1000
trimesh_triangle = mesh_for_trimesh.triangles
vertices_normal = mesh_for_trimesh.vertex_normals

model_normal = pd.read_csv('data/1_8_2.xyz', header=None, sep=' ').to_numpy()[:, 3:6]
mesh_for_trimesh.vertex_normals = model_normal

start_time = time.process_time()
pcd_ref = o3d.io.read_point_cloud("data/1_8_2_df - point.pcd")
end_time = time.process_time()
print("读取参考点云数据总耗时 ：" + str(end_time - start_time) + "s")
points_in_ref = np.asarray(pcd_ref.points) * 1000

start_time = time.process_time()
kdt = grab_tree("tree.txt")
end_time = time.process_time()
print("读取knn树总耗时 ：" + str(end_time - start_time) + "s")

density_ratio = int(len(points_in_ref)/len(trimesh_points))

vertices_normal_after = mesh_for_trimesh.vertex_normals

points_coor = np.asarray(pcd.points)*1000

for j in np.arange(4068, 4078):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_tower, show_edges=True, color="white")

    start = points_coor[j]
    start_vertex_normals = mesh_for_trimesh.vertex_normals[j]
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

            # 加入射线与目标点法向量的限制，过滤掉夹角大于60度的射线
            if np.any(sum_dis_check <= 0.0001) and angle_between_vectors(start_vertex_normals, (stop - start) / 1000) > 0.866:
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
                    coneA = useful_tools(cam_loc[i], points_coor[j], z_axis, pix_size_=pixel_size, focal_=f)
                    meshA = trimesh.Trimesh(vertices=np.asarray(coneA.vertices), faces=np.asarray(coneA.triangles))
                    print("new round started")

                    continue

                # print("working on " + str(v + 1))
                coneB = useful_tools(cam_loc[i], points_coor[j], z_axis, pix_size_=pixel_size, focal_=f)
                meshB = trimesh.Trimesh(vertices=np.asarray(coneB.vertices), faces=np.asarray(coneB.triangles))

                meshA = trimesh.boolean.intersection([meshA, meshB], engine='scad')
                meshA = trimesh.convex.convex_hull(meshA, qhull_options='Qt')

                v += 1

    end_time = time.process_time()

    _ = plotter.add_axes(box=True)

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
        # mesh.plot(show_edges=True, line_width=1)

        max_distance = np.max(distance_matrix(start.reshape((1, 3)), points))
        dist_set = np.max(np.sqrt(np.sum((start.reshape((1, 3)) - points) ** 2, axis=1)))

        core_point = points_coor[j].reshape((1, 3))/1000
        idx, dis_tree = kdt.query_radius(core_point, r=max_distance/1000, return_distance=True)
        idx = idx[0]
        dis_tree = dis_tree[0]

        neighbor_set = points_in_ref[idx]

        pcd_neighbor_set = o3d.geometry.PointCloud()
        pcd_neighbor_set.points = o3d.utility.Vector3dVector(neighbor_set)

        intersection_mesh = meshA.as_open3d

        core_from_target = o3d.geometry.TriangleMesh.create_sphere(radius=2.0).translate((core_point[0, 0]*1000, core_point[0, 1]*1000, core_point[0, 2]*1000))
        # o3d.visualization.draw_geometries([pcd_neighbor_set, core_from_target, intersection_mesh],
        #                                   mesh_show_wireframe=True,
        #                                   window_name='4 pixel')

        # o3d.visualization.draw_geometries([pcd_neighbor_set, core_from_target],
                                          # window_name='before filtered')

        signed_dis = trimesh.proximity.signed_distance(meshA, points_in_ref[idx])

        idx_inner = np.argwhere(signed_dis > 0).flatten().tolist()
        neighbor_set_inner = points_in_ref[idx[idx_inner]]

        pcd_neighbor_set_inner = o3d.geometry.PointCloud()
        pcd_neighbor_set_inner.points = o3d.utility.Vector3dVector(neighbor_set_inner)

        # o3d.visualization.draw_geometries([pcd_neighbor_set_inner, core_from_target],
                                          # window_name='filtered')

        gaussian_weight = np.array(gaussian_dis(signed_dis[idx_inner], sigma=7, mu=np.min(signed_dis[idx_inner])))
        filtered_dis = np.array(dis_tree[idx_inner])*1000

        average_dis = np.sum(gaussian_weight*filtered_dis)/np.sum(gaussian_weight)

        print("高斯加权平均后误差", average_dis, "mm")
        print("直接平均值误差", np.average(filtered_dis), "mm")
        print("临近点数量", len(dis_tree))

        # plotter.show()
        del meshA, meshB
        gc.collect()

    else:
        print('no intersection')
