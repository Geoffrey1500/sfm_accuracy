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


def sen_pts_gen(dist_, cam_id_, points_per_side_=30, scale_factor_=1.5):
    # To create sensor plane

    dist_s_ = dist_[cam_id_]

    x_ = np.linspace((-w/2+pixel_size/2)*scale_factor_, (w/2-pixel_size/2)*scale_factor_,
                     int(3*points_per_side_*scale_factor_))
    y_ = np.linspace((-h/2+pixel_size/2)*scale_factor_, (h/2-pixel_size/2)*scale_factor_,
                     int(2*points_per_side_*scale_factor_))
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
    pix_dist_pd = pd.DataFrame(np.rint(pix_dist_))

    # ind_x_ = np.where((0 < pix_dist_[:, 0]) & (pix_dist_[:, 0] < resol_x), True, False)
    # ind_y_ = np.where((0 < pix_dist_[:, 1]) & (pix_dist_[:, 1] < resol_y), True, False)

    pix_inside_idx = ((0 <= pix_dist_pd[0]) & (pix_dist_pd[0] < resol_x) & (0 <= pix_dist_pd[1]) & (pix_dist_pd[1] < resol_y)).values

    pix_du_idx = pix_dist_pd.duplicated(keep='last').values

    # ind_final = np.logical_and(np.logical_and(ind_x_, ind_y_), pix_du_idx)
    ind_final = np.logical_and(pix_inside_idx, ~pix_du_idx)
    pts_for_view = np.hstack((x_, y_, z_))

    filtered_pts_ = pts_for_view[ind_final]
    print(len(pts_dist), len(filtered_pts_), resol_x*resol_y)

    # visualize_camera(pts_for_view, pts_for_view[ind_final])

    return filtered_pts_


def find_nearest_hit_pts(cam_loc_, cam_pos_, org_rays_, mesh_, scene_):
    '''
    :param cam_loc_: 相机位置
    :param cam_pos_: 相机姿态
    :param org_rays_: 用于批量光追的光线的初始位置
    :param mesh_: o3d 格式的 mesh 模型
    :param scene_: 光追场景
    :return: 和光线相交的符合要求的点的索引
    '''

    '''
    构建用于光追的光线集合"rays_"，并将其投射至场景"scene_"之中，返回的结果存为"ans_"
    '''
    rays_direction = np.asarray(cam_pos_.apply(org_rays_))
    rays_direction = rays_direction / rays_direction.max(axis=1).reshape((-1, 1))

    rays_starts = np.expand_dims(cam_loc_, 0).repeat(len(rays_direction), axis=0)
    rays_sets = np.hstack((rays_starts, rays_direction))
    # print(rays_sets)
    rays_ = o3d.core.Tensor(rays_sets, dtype=o3d.core.Dtype.Float32)

    orig_tri_ = np.asarray(mesh_.triangles)
    ans_ = scene_.cast_rays(rays_)

    '''
    找到首次相交的三角面片，并且将组成三角面片的三个顶点的坐标分别存到 tri_pts_cors_1, tri_pts_cors_2, tri_pts_cors_3
    注意：在 geometry_ids 和 primitive_ids 中 4294967295 等同于 finite 无效数字
    '''
    triangle_index = ans_['primitive_ids'][ans_['primitive_ids'] != 4294967295].numpy()
    hit_triangles = orig_tri_[triangle_index]
    tri_pts_cors_1 = np.asarray(mesh_.vertices)[hit_triangles[:, 0]]
    tri_pts_cors_2 = np.asarray(mesh_.vertices)[hit_triangles[:, 1]]
    tri_pts_cors_3 = np.asarray(mesh_.vertices)[hit_triangles[:, 2]]
    # print(tri_pts_cors_1)

    '''
    计算rays和mesh相交的坐标
    '''
    rays_index = ans_['t_hit'].isfinite()
    inter_pts_cors = rays_[rays_index][:, :3] + rays_[rays_index][:, 3:] * ans_['t_hit'][rays_index].reshape((-1, 1))
    inter_pts_cors = inter_pts_cors.numpy()
    # print(inter_pts_cors)
    # print(len(inter_pts_cors), len(triangle_index))

    '''
    找到距离三角面片的交点与最近的顶点，这里用的距离是 Manhattan Distance，用于加速计算
    '''
    dis_pts2tri_1 = np.sum(np.abs(inter_pts_cors - tri_pts_cors_1), axis=1).reshape(-1, 1)
    dis_pts2tri_2 = np.sum(np.abs(inter_pts_cors - tri_pts_cors_2), axis=1).reshape(-1, 1)
    dis_pts2tri_3 = np.sum(np.abs(inter_pts_cors - tri_pts_cors_3), axis=1).reshape(-1, 1)

    dis_pts2tri_set = np.hstack((dis_pts2tri_1, dis_pts2tri_2, dis_pts2tri_3))
    # print(dis_pts2tri_set)

    '''
    找到距离三角面片的交点与最近的顶点的索引
    '''
    idx_col = np.argmin(dis_pts2tri_set, axis=1)

    '''
    找到对应顶点的索引
    这里用的是一个组合索引，np.arange(len(hit_triangles))代表索引所有的行，idx_col是对应的列索引
    得到的是：距离三角面片的交点与最近的顶点的索引
    '''
    idx_pts = hit_triangles[np.arange(len(hit_triangles)), idx_col]
    # print(index_ff)
    # print(hit_triangles)

    # print(np.sum(np.abs(inter_pts_cors - tri_pts_cors), axis=1))
    return idx_pts


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

    data = pd.read_csv("data/UAV_only4.csv")
    # print(data.head(5))
    cam_loc = data[["x", "y", "alt"]].values
    euler_ang = data[["roll", "pitch", "heading"]].values * np.array([[1, 1, -1]])
    rot_mat_set = R.from_euler('yxz', euler_ang, degrees=True)
    dist = data[["k1", "k2", "k3", "k4"]].values

    mesh = o3d.io.read_triangle_mesh('data/UAV_only.ply')
    original_triangle = np.asarray(mesh.triangles)

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

    original_sen_pts = sen_pts_gen(dist, int(len(cam_loc)/2))

    kkk = 197
    print(cam_loc[kkk].reshape((-1, 1)))

    rays_direction = np.asarray(rot_mat_set[kkk].apply(original_sen_pts))
    rays_direction = rays_direction / rays_direction.max(axis=1).reshape((-1, 1))

    rays_starts = np.expand_dims(cam_loc[kkk], 0).repeat(len(rays_direction), axis=0)

    rays_sets = np.hstack((rays_starts, rays_direction))
    # print(rays_sets)
    rays = o3d.core.Tensor(rays_sets, dtype=o3d.core.Dtype.Float32)

    # We can directly pass the rays tensor to the cast_rays function.
    ans = scene.cast_rays(rays)

    # print(original_triangle)
    # 在 geometry_ids 和 primitive_ids 中 4294967295 等同于 finite 无效数字
    triangle_index = ans['primitive_ids'][ans['primitive_ids'] != 4294967295].numpy()
    hit_triangles = original_triangle[triangle_index]
    tri_pts_cors_1 = np.asarray(mesh.vertices)[hit_triangles[:, 0]]
    tri_pts_cors_2 = np.asarray(mesh.vertices)[hit_triangles[:, 1]]
    tri_pts_cors_3 = np.asarray(mesh.vertices)[hit_triangles[:, 2]]

    # print(tri_pts_cors_1)

    rays_index = ans['t_hit'].isfinite()
    inter_pts_cors = rays[rays_index][:, :3] + rays[rays_index][:, 3:] * ans['t_hit'][rays_index].reshape((-1, 1))
    inter_pts_cors = inter_pts_cors.numpy()
    # print(inter_pts_cors)

    # print(len(inter_pts_cors), len(triangle_index))

    dis_pts2tri_1 = np.sum(np.abs(inter_pts_cors - tri_pts_cors_1), axis=1).reshape(-1, 1)
    dis_pts2tri_2 = np.sum(np.abs(inter_pts_cors - tri_pts_cors_2), axis=1).reshape(-1, 1)
    dis_pts2tri_3 = np.sum(np.abs(inter_pts_cors - tri_pts_cors_3), axis=1).reshape(-1, 1)

    dis_pts2tri_set = np.hstack((dis_pts2tri_1, dis_pts2tri_2, dis_pts2tri_3))
    # print(dis_pts2tri_set)

    index_xx = np.argmin(dis_pts2tri_set, axis=1)
    # print(index_xx)

    index_ff = hit_triangles[np.arange(len(hit_triangles)), index_xx]
    # print(index_ff)
    # print(hit_triangles)

    # print(np.sum(np.abs(inter_pts_cors - tri_pts_cors), axis=1))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[index_ff])
    pcd.colors = o3d.utility.Vector3dVector(colors[index_ff])
    # pcd.normals = o3d.utility.Vector3dVector(normals[ans['primitive_ids'].numpy()])
    o3d.visualization.draw_geometries([pcd])


def my_ray_casting4():

    data = pd.read_csv("data/UAV_only4.csv")
    # print(data.head(5))
    cam_loc = data[["x", "y", "alt"]].values
    euler_ang = data[["roll", "pitch", "heading"]].values * np.array([[1, 1, -1]])
    rot_mat_set = R.from_euler('yxz', euler_ang, degrees=True)
    dist = data[["k1", "k2", "k3", "k4"]].values

    mesh = o3d.io.read_triangle_mesh('data/UAV_only.ply')
    original_triangle = np.asarray(mesh.triangles)

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

    original_sen_pts = sen_pts_gen(dist, int(len(cam_loc)/2))

    kkk = 197
    print(cam_loc[kkk].reshape((-1, 1)))

    index_ff = find_nearest_hit_pts(cam_loc[kkk], rot_mat_set[kkk], original_sen_pts, mesh, scene)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[index_ff])
    pcd.colors = o3d.utility.Vector3dVector(colors[index_ff])
    # pcd.normals = o3d.utility.Vector3dVector(normals[ans['primitive_ids'].numpy()])
    o3d.visualization.draw_geometries([pcd])


def test_duplicate():
    df = pd.DataFrame(np.array([[1, 2, 3, 3, 4],
                                [22, 33, 22, 44, 66]]).T)
    print(df)
    a = ((df[0] <= 3) & (df[1] <= 24) & (0 < df[0]) & (22 <= df[1])).values
    print(a)


if __name__ == '__main__':
    my_ray_casting4()
    # test_duplicate()
