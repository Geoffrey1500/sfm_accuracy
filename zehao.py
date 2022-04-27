import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import time


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
intrinsic_matrix = [[3685.25307322617, 0, resol_x / 2 - 26.1377554238884],
                    [0, 3685.25307322617, resol_y / 2 - 14.8594719360401],
                    [0, 0, 1]]
# dist: 畸变参数 [k1, k2, k3, k4, p1, p2]
dist = np.array([-0.288928920598278, 0.145903038241546, -0.0664869742590238, 0.0155044924834934, -0.000606112069582838, 0.000146688084883612])


def sen_pts_gen(pts_, cam_loc_, cam_pos_, dist_s_):
    '''
    :param pts_: 点云坐标 [N x 3]
    :param cam_loc_: 相机坐标 [1 x 3]
    :param cam_pos_: 相机姿态，由scipy.spatial.transform模块生成
    :param dist_s_: 畸变参数 [k1, k2, k3, k4, p1, p2]
    :return:
    '''

    # 注意：一般情况下，由三维重建或者SLAM所获取的相机姿态和位置，都是以世界坐标系建立的，都是相机相对世界坐标系的位置和姿态
    # 而在相机的外参矩阵中，应当使用的是上述矩阵的逆矩阵，很关键
    rot_ext = np.vstack((np.hstack((cam_pos_.as_matrix(), cam_loc_.reshape((-1, 1)))),
                         np.array([[0, 0, 0, 1]])))
    rot_ext_2 = np.linalg.inv(rot_ext)

    pts_cam_ = np.dot(rot_ext_2, np.hstack((pts_, np.ones((len(pts_), 1)))).T).T
    pts_cam_new = pts_cam_/(pts_cam_[:, -2].reshape((-1, 1)))

    x_corrt = pts_cam_new[:, 0].reshape((-1, 1))
    y_corrt = pts_cam_new[:, 1].reshape((-1, 1))
    r_ = x_corrt ** 2 + y_corrt ** 2

    x_dist = x_corrt * ((1 + dist_s_[0] * r_ + dist_s_[1] * (r_ ** 2) + dist_s_[2] * (r_ ** 3))/(1 + dist_s_[3] * r_)) \
             + 2*dist_s_[4]*x_corrt*y_corrt + dist_s_[5]*(r_ + 2*x_corrt**2)
    y_dist = y_corrt * ((1 + dist_s_[0] * r_ + dist_s_[1] * (r_ ** 2) + dist_s_[2] * (r_ ** 3))/(1 + dist_s_[3] * r_)) \
             + dist_s_[4]*(r_ + 2*y_corrt**2) + 2*dist_s_[5]*x_corrt*y_corrt

    pts_dist = np.hstack((x_dist, y_dist, np.ones_like(x_dist)))
    pix_dist_ = np.dot(intrinsic_matrix, pts_dist.T).T

    green = np.array([0, 1, 0])
    colors_tar = np.expand_dims(green, 0).repeat(len(pix_dist_), axis=0)
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(pix_dist_)
    pcd_ref.colors = o3d.utility.Vector3dVector(colors_tar)

    pix_dist_pd = pd.DataFrame(pix_dist_)
    pix_inside_idx = ((0 <= pix_dist_pd[0]) & (pix_dist_pd[0] < resol_x) & (0 <= pix_dist_pd[1]) & (pix_dist_pd[1] < resol_y)).values
    #
    # pix_dist_pd = pd.DataFrame(np.rint(pix_dist_))
    #
    # pix_du_idx = pix_dist_pd.duplicated(keep='last').values
    #
    # ind_final = np.logical_and(pix_inside_idx, ~pix_du_idx)

    ray_starts_ = np.dot(np.linalg.inv(rot_ext_2), np.hstack((pts_dist, np.ones((len(pts_dist), 1)))).T).T
    ray_starts_filtered = ray_starts_[pix_inside_idx][:, :-1]

    rays_dir_ = pts_[pix_inside_idx] - ray_starts_filtered
    rays_all_of_them_ = np.hstack((ray_starts_filtered, rays_dir_))

    return pix_inside_idx, rays_all_of_them_


def find_nearest_hit_pts(org_rays_, mesh_, scene_):
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
    # print(rays_sets)
    rays_ = o3d.core.Tensor(org_rays_, dtype=o3d.core.Dtype.Float32)

    orig_tri_ = np.asarray(mesh_.triangles)
    ans_ = scene_.cast_rays(rays_)

    '''
    找到首次相交的三角面片
    将组成三角面片的三个顶点的坐标分别存到 tri_pts_cors_1, tri_pts_cors_2, tri_pts_cors_3
    注意：
    1. 在 geometry_ids 和 primitive_ids 中 4294967295 等同于 finite 无效数字
    2. 同一个三角面片只能被相交一次
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
    注意：同一个三角面片只能被相交一次
    '''
    idx_col = np.argmin(dis_pts2tri_set, axis=1)
    tri_pd = pd.DataFrame(hit_triangles)
    tri_pd_2 = tri_pd.duplicated(keep='last').values

    filtered_tris = hit_triangles[~tri_pd_2]
    idx_col_2 = idx_col.reshape((-1, 1))[~tri_pd_2]
    '''
    找到对应顶点的索引
    这里用的是一个组合索引，np.arange(len(hit_triangles))代表索引所有的行，idx_col是对应的列索引
    得到的是：距离三角面片的交点与最近的顶点的索引
    可能出现由相邻面片和ray的交点所求得的最近的顶点是同一点的情况，因此需要用np.unique()来进一步过滤
    '''
    idx_pts = filtered_tris[np.arange(len(filtered_tris)), idx_col_2.flatten()]
    idx_pts_2 = np.unique(idx_pts)
    # print(index_ff)
    # print(hit_triangles)

    # print(np.sum(np.abs(inter_pts_cors - tri_pts_cors), axis=1))
    return idx_pts_2


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


def my_ray_casting():
    start = time.perf_counter()
    data = pd.read_csv("data/zehao/cameras/25m30d90o.csv", encoding = "utf-8")
    # print(data.head(5))
    cam_loc = data[["X", "Y", "Z"]].values
    euler_ang = data[["R", "P", "Y"]].values * np.array([[1, 1, -1]]) + np.array([[0, 180, 0]])
    rot_mat_set = R.from_euler('yxz', euler_ang, degrees=True)

    mesh = o3d.io.read_triangle_mesh('data/zehao/plys/1.ply')
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)

    print(mesh.has_adjacency_list)
    print(mesh.has_textures)
    print(mesh.has_triangle_normals)
    print(mesh.has_triangles)
    print(mesh.has_vertex_colors)
    print(mesh.has_vertex_normals)
    print(mesh.has_vertices)
    # o3d.visualization.draw_geometries([pcd, mesh], mesh_show_wireframe=True, point_show_normal=False)

    mesh_for_ray = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_for_ray)

    end = time.perf_counter()
    print('原始数据载入和预处理的时长为:', end-start)

    ini_count = np.empty(0)

    # for i in np.arange(len(cam_loc)):
    # for test, i should be set as 120, 197, 220
    for i in np.arange(120, 131):
        start = time.perf_counter()
        idx_inte_pts, rays_sets_2 = sen_pts_gen(points, cam_loc[i], rot_mat_set[i], dist)
        #
        # rays_direction = points[idx_inte_pts]-cam_loc[i].reshape((1, -1))
        # rays_direction = rays_direction / rays_direction.max(axis=1).reshape((-1, 1))
        #
        # rays_starts = np.expand_dims(cam_loc[i], 0).repeat(len(rays_direction), axis=0)
        # rays_sets = np.hstack((rays_starts, rays_direction))
        idx_inte_pts = find_nearest_hit_pts(rays_sets_2, mesh, scene)
        end = time.perf_counter()
        print(i)
        print('计算可视相机数量时长:', end - start)

        print(i)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[idx_inte_pts])
        pcd.colors = o3d.utility.Vector3dVector(colors[idx_inte_pts])
        # pcd.normals = o3d.utility.Vector3dVector(normals[ans['primitive_ids'].numpy()])
        o3d.visualization.draw_geometries([pcd])

        ini_count = np.append(ini_count, idx_inte_pts)

    start = time.perf_counter()
    unique, counts = np.unique(ini_count, return_counts=True)
    end = time.perf_counter()
    print('去重统计所需时长:', end - start)

    print(counts, max(counts), min(counts))


if __name__ == '__main__':
    my_ray_casting()
    # test_duplicate()
