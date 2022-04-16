import pyvista as pv
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import time
import trimesh


def sensor_plane_point(points_per_side_=2, scale_factor_=200):
    # To create sensor plane

    x_ = np.linspace(-w/2*scale_factor_, w/2*scale_factor_, 3*points_per_side_)
    y_ = np.linspace(-h/2*scale_factor_, h/2*scale_factor_, 2*points_per_side_)
    x_, y_ = np.meshgrid(x_, y_)
    z_ = (np.zeros_like(x_) - f)*scale_factor_

    points_in_sensor_ = np.hstack((x_.reshape((-1, 1)), y_.reshape((-1, 1)),  z_.reshape((-1, 1))))

    return points_in_sensor_


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

data_path = "data/25m"
filename_list = ["50o", "60o", "70o", "80o", "90o"]

for k in filename_list:
    """
    读取相机位置和朝向
    """
    camera_data = pd.read_csv(data_path + '/cameras/' + k + ".csv")
    cam_loc = camera_data[["x", "y", "z"]].values*1000
    euler_ang = camera_data[["heading", "pitch", "roll"]].values * np.array([[-1, 1, 1]]) + np.array([[0, 0, 0]])
    rot_mat_set = R.from_euler('ZXY', euler_ang, degrees=True)

    '''
    读取目标点云
    '''
    original_data = pd.read_csv(data_path + '/points/' + k + ".txt", header=None, sep=' ').to_numpy()
    points = original_data[:, 0:3]*1000
    colors = original_data[:, 3:6]/255
    normals = original_data[:, 6::]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    '''
    计算平均临近点距离
    '''
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    std_dist = np.std(distances)

    '''
    重建三角面片
    '''
    point_cloud = pv.PolyData(points)
    point_cloud.point_data.active_normals = normals
    mesh_for_pv = point_cloud.delaunay_2d(alpha=avg_dist + std_dist * 3)
    mesh_for_pv.point_data.active_normals = normals
    mesh_for_pv['colors'] = colors

    faces_pv = mesh_for_pv.faces.reshape((-1, 4))[:, 1:4]
    points_pv = mesh_for_pv.points

    # create the triangular mesh with the vertices and faces from pyvista
    mesh_for_trimesh = trimesh.Trimesh(vertices=points, faces=np.asarray(faces_pv), vertex_normals=normals)

    num_img_visible = np.ones(len(points))*-1

    for j in np.arange(len(points)):
        start = points[j]
        start_vertex_normals = normals[j]
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
                ray_dire = (start-stop)

                # 判断 ray 和 mesh 第一个相交点是否为目标点
                index_tri = mesh_for_trimesh.ray.intersects_first(
                    ray_origins=[stop],
                    ray_directions=[ray_dire])

                tri_queried = mesh_for_trimesh.triangles[index_tri]
                points_wanted = start

                dis_check = tri_queried[0] - points_wanted
                sum_dis_check = np.sum(dis_check**2, axis=1)

                # 加入射线与目标点法向量的限制，过滤掉夹角大于60度的射线
                if np.any(sum_dis_check <= 0.0001) and angle_between_vectors(start_vertex_normals, (stop - start) / 1000) > 0.866:

                    v += 1

        end_time = time.process_time()
        print("共运行：" + str(end_time - start_time) + "s")
        print("一共模拟 " + str(v) + " 个点")
        print(v)

        if v == 0:
            print('no intersection')
        num_img_visible[j] = v

    np.savetxt(data_path + '/results/' + k + ".txt", num_img_visible, fmt="%d")
    print(k + " is finished")
