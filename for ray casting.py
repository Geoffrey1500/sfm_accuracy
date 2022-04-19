import pyvista as pv
from pyvista import examples
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import sympy as sp


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
intrinsic_matrix = [[f/pixel_size, 0, resol_x/2],
                    [0, f/pixel_size, resol_y/2],
                    [0, 0, 1]]



def sensor_plane_point(points_per_side_=500, scale_factor_=1.0):
    # To create sensor plane

    x_ = np.linspace((-w/2+pixel_size/2)*scale_factor_, (w/2-pixel_size/2)*scale_factor_, 3*points_per_side_)
    y_ = np.linspace((-h/2+pixel_size/2)*scale_factor_, (h/2-pixel_size/2)*scale_factor_, 2*points_per_side_)
    x_, y_ = np.meshgrid(x_, y_)

    x_ = x_.reshape((-1, 1))
    y_ = y_.reshape((-1, 1))
    z_ = (np.zeros_like(x_) - f)*scale_factor_

    points_in_sensor_ = np.hstack((x_, y_,  z_))

    return points_in_sensor_


def sensor_plane_point_2(points_per_side_=500, scale_factor_=1.0):
    # To create sensor plane

    x_ = np.linspace((-w/2+pixel_size/2)*scale_factor_, (w/2-pixel_size/2)*scale_factor_, 3*points_per_side_)
    y_ = np.linspace((-h/2+pixel_size/2)*scale_factor_, (h/2-pixel_size/2)*scale_factor_, 2*points_per_side_)
    x_, y_ = np.meshgrid(x_, y_)

    x_ = x_.reshape((-1, 1))
    y_ = y_.reshape((-1, 1))
    z_ = (np.zeros_like(x_) - f)

    points_in_sensor_ = np.hstack((x_, y_,  z_))

    return points_in_sensor_


def sen_pts_gen(dist_, cam_id_, points_per_side_=1824):
    # To create sensor plane

    dist_s_ = dist_[cam_id_]

    x_ = np.linspace((-w/2+pixel_size/2)*1.5, (w/2-pixel_size/2)*1.5, 3*points_per_side_)
    y_ = np.linspace((-h/2+pixel_size/2)*1.5, (h/2-pixel_size/2)*1.5, 2*points_per_side_)
    x_, y_ = np.meshgrid(x_, y_)

    x_ = x_.reshape((-1, 1))
    y_ = y_.reshape((-1, 1))
    z_ = (np.zeros_like(x_) - f)

    x_corrt = x_/z_
    y_corrt = y_/z_
    r_ = x_corrt ** 2 + y_corrt ** 2

    x_dist = x_corrt * (1 + dist_s_[0] * r_ + dist_s_[1] * (r_ ** 2) + dist_s_[2] * (r_ ** 3))
    y_dist = y_corrt * (1 + dist_s_[0] * r_ + dist_s_[1] * (r_ ** 2) + dist_s_[2] * (r_ ** 3))

    pts_dist = np.hstack((x_dist, y_dist, np.ones_like(x_dist)))
    pix_dist_ = np.dot(intrinsic_matrix, pts_dist.T).T

    ind_x_ = np.where((0 < pix_dist_[:, 0]) & (pix_dist_[:, 0] < resol_x), True, False)
    ind_y_ = np.where((0 < pix_dist_[:, 1]) & (pix_dist_[:, 1] < resol_y), True, False)

    ind_final = np.logical_and(ind_x_, ind_y_)

    pts_for_view = np.hstack((x_, y_, z_))

    visualize_camera(pts_for_view, pts_for_view[ind_final])

    return pts_for_view[ind_final]


def test2():
    data = pd.read_csv("data/UAV_only4.csv")
    # print(data.head(5))
    cam_loc = data[["x", "y", "alt"]].values
    euler_ang = data[["roll", "pitch", "heading"]].values * np.array([[1, 1, -1]])
    rot_mat_set = R.from_euler('yxz', [[0, 0, 0]], degrees=True)

    dist = data[["k1", "k2", "k3", "k4"]].values

    kkk = 0
    aaa = sen_pts_gen(dist, kkk, points_per_side_=30)
    print(cam_loc[kkk].reshape((-1, 1)))

    rays_direction2 = np.asarray(rot_mat_set[kkk].apply(sensor_plane_point_2(points_per_side_=30, scale_factor_=1.5)))

    pt3, pt4, pix_dist_2 = dist_pts_2(rays_direction2, dist[kkk])

    print(np.min(pix_dist_2, axis=0))
    print(0 < pix_dist_2[:, 0])

    ind_x = np.where((0 < pix_dist_2[:, 0]) & (pix_dist_2[:, 0] < resol_x), True, False)
    ind_y = np.where((0 < pix_dist_2[:, 1]) & (pix_dist_2[:, 1] < resol_y), True, False)

    ind_final = np.logical_and(ind_x, ind_y)

    filtered = pix_dist_2[ind_final]
    print(np.min(filtered, axis=0))

    x_ = np.linspace(0, resol_x, 3*20)
    y_ = np.linspace(0, resol_y, 2*20)
    x_, y_ = np.meshgrid(x_, y_)

    x_ = x_.reshape((-1, 1))
    y_ = y_.reshape((-1, 1))
    z_ = np.ones_like(x_)

    points_in_sensor_ = np.hstack((x_, y_,  z_))

    print(pix_dist_2)
    print(points_in_sensor_)

    visualize_camera(pt4, pt4[ind_final])

    # blue = np.array([0, 0, 1])
    # colors_tar = np.expand_dims(blue, 0).repeat(len(pt4[ind_list>0]), axis=0)
    # pcd_tar = o3d.geometry.PointCloud()
    # pcd_tar.points = o3d.utility.Vector3dVector(pt4[ind_list>0])
    # pcd_tar.colors = o3d.utility.Vector3dVector(colors_tar)
    # o3d.visualization.draw_geometries([pcd_tar])


def dist_pts_2(pts_org_, dist_):
    z_corrt = pts_org_[:, -1].reshape((-1, 1))
    x_corrt = pts_org_[:, 0].reshape((-1, 1))/z_corrt
    y_corrt = pts_org_[:, 1].reshape((-1, 1))/z_corrt
    r_ = x_corrt ** 2 + y_corrt ** 2

    pts_org_new = np.hstack((x_corrt, y_corrt, np.ones_like(x_corrt)))
    pixel_org = np.dot(intrinsic_matrix, pts_org_new.T).T

    x_dist = x_corrt * (1 + dist_[0] * r_ + dist_[1] * (r_ ** 2) + dist_[2] * (r_ ** 3))
    y_dist = y_corrt * (1 + dist_[0] * r_ + dist_[1] * (r_ ** 2) + dist_[2] * (r_ ** 3))

    pts_dist = np.hstack((x_dist, y_dist, np.ones_like(x_dist)))
    pixel_dist = np.dot(intrinsic_matrix, pts_dist.T).T

    # print(pts_dist)
    # print(pixel_org)
    # print(np.min(pixel_dist, axis=0))
    # print(np.min(pixel_org, axis=0))

    return pts_org_new, pts_dist, pixel_dist


def test():
    data = pd.read_csv("data/UAV_only4.csv")
    # print(data.head(5))
    cam_loc = data[["x", "y", "alt"]].values
    euler_ang = data[["roll", "pitch", "heading"]].values * np.array([[1, 1, -1]])
    rot_mat_set = R.from_euler('yxz', [[0, 0, 0]], degrees=True)

    dist = data[["k1", "k2", "k3", "k4"]].values

    kkk = 0
    print(cam_loc[kkk].reshape((-1, 1)))

    rays_direction = np.asarray(rot_mat_set[kkk].apply(sensor_plane_point(points_per_side_=30)))

    pt1, pt2 = dist_pts_2(rays_direction, dist[kkk])

    visualize_camera(pt2, pt1)


def visualize_camera(pts_tar, pts_ref):
    blue = np.array([0, 0, 1])
    colors_tar = np.expand_dims(blue, 0).repeat(len(pts_tar), axis=0)
    pcd_tar = o3d.geometry.PointCloud()
    pcd_tar.points = o3d.utility.Vector3dVector(pts_tar)
    pcd_tar.colors = o3d.utility.Vector3dVector(colors_tar)

    red = np.array([1, 0, 0])
    colors_ref = np.expand_dims(red, 0).repeat(len(pts_ref), axis=0)
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(pts_ref)
    pcd_ref.colors = o3d.utility.Vector3dVector(colors_ref)

    o3d.visualization.draw_geometries([pcd_ref, pcd_tar])


def my_ray_casting3():
    data = pd.read_csv("data/UAV_only.csv")
    print(data.head(5))
    cam_loc = data[["x", "y", "z"]].values
    euler_ang = data[["roll", "pitch", "heading"]].values * np.array([[1, 1, -1]])
    rot_mat_set = R.from_euler('yxz', euler_ang, degrees=True)

    mesh = o3d.io.read_triangle_mesh('data/UAV_only.ply')

    original_data = pd.read_csv('data/UAV_only.xyz', header=None, sep=' ').to_numpy()
    points = original_data[:, 0:3]
    colors = original_data[:, 6::] / 255
    normals = original_data[:, 3:6]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    # o3d.visualization.draw_geometries([pcd, mesh], mesh_show_wireframe=True, point_show_normal=False)

    mesh_for_ray = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_for_ray)

    kkk = 301
    print(cam_loc[kkk].reshape((-1, 1)))

    rays_direction = np.asarray(rot_mat_set[kkk].apply(sensor_plane_point(points_per_side_=1824)))
    print(rays_direction)
    rays_direction = rays_direction / rays_direction.max(axis=1).reshape((-1, 1))

    rays_starts = np.expand_dims(cam_loc[kkk], 0).repeat(len(rays_direction), axis=0)

    rays_sets = np.hstack((rays_starts, rays_direction))
    print(rays_sets)
    rays = o3d.core.Tensor(rays_sets,
                           dtype=o3d.core.Dtype.Float32)

    # We can directly pass the rays tensor to the cast_rays function.
    ans = scene.cast_rays(rays)

    original_triangle = np.asarray(mesh.triangles)
    # print(original_triangle)
    # 在 geometry_ids 和 primitive_ids 中 4294967295 等同于 finite 无效数字

    triangle_index = ans['primitive_ids'][ans['primitive_ids'] != 4294967295].numpy()
    hit_triangles = original_triangle[triangle_index]
    tri_pts_cors_1 = np.asarray(mesh.vertices)[hit_triangles[:, 0]]
    tri_pts_cors_2 = np.asarray(mesh.vertices)[hit_triangles[:, 1]]
    tri_pts_cors_3 = np.asarray(mesh.vertices)[hit_triangles[:, 2]]

    print(tri_pts_cors_1)

    rays_index = ans['t_hit'].isfinite()
    inter_pts_cors = rays[rays_index][:, :3] + rays[rays_index][:, 3:] * ans['t_hit'][rays_index].reshape((-1, 1))
    inter_pts_cors = inter_pts_cors.numpy()
    print(inter_pts_cors)

    print(len(inter_pts_cors), len(triangle_index))

    dis_pts2tri_1 = np.sum(np.abs(inter_pts_cors - tri_pts_cors_1), axis=1).reshape(-1, 1)
    dis_pts2tri_2 = np.sum(np.abs(inter_pts_cors - tri_pts_cors_2), axis=1).reshape(-1, 1)
    dis_pts2tri_3 = np.sum(np.abs(inter_pts_cors - tri_pts_cors_3), axis=1).reshape(-1, 1)

    dis_pts2tri_set = np.hstack((dis_pts2tri_1, dis_pts2tri_2, dis_pts2tri_3))
    print(dis_pts2tri_set)

    index_xx = np.argmin(dis_pts2tri_set, axis=1)
    print(index_xx)

    index_ff = hit_triangles[np.arange(len(hit_triangles)), index_xx]
    print(index_ff)
    print(hit_triangles)

    # print(np.sum(np.abs(inter_pts_cors - tri_pts_cors), axis=1))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[index_ff])
    pcd.colors = o3d.utility.Vector3dVector(colors[index_ff])
    # pcd.normals = o3d.utility.Vector3dVector(normals[ans['primitive_ids'].numpy()])
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    test2()

