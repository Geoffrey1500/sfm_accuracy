import open3d as o3d
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import trimesh
import time


def angle_between_vectors(v1_, v2_):
    dot_pr = np.sum(v1_*v2_, axis=1).reshape((-1, 1))
    norms_ = np.linalg.norm(v1_, axis=1, keepdims=True) * np.linalg.norm(v2_, axis=1, keepdims=True)

    return np.rad2deg(np.arccos(dot_pr / norms_))


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
    angles = angle_between_vectors(pts_cam_[:, :3], np.array([[0, 0, 1]]))
    angle_idx = angles.flatten() <= fov/2*1.25

    pts_cam_new = pts_cam_[:, :3]/(pts_cam_[:, -2].reshape((-1, 1)))

    x_corrt = pts_cam_new[:, 0].reshape((-1, 1))
    y_corrt = pts_cam_new[:, 1].reshape((-1, 1))
    r_ = x_corrt ** 2 + y_corrt ** 2

    x_prime_ = x_corrt * (1 + dist_s_[0] * r_ + dist_s_[1] * (r_ ** 2) + dist_s_[2] * (r_ ** 3) + dist_s_[3] * (r_**4)) \
              + dist_s_[4]*(r_ + 2*x_corrt**2) + 2*dist_s_[5]*x_corrt*y_corrt
    y_prime_ = y_corrt * (1 + dist_s_[0] * r_ + dist_s_[1] * (r_ ** 2) + dist_s_[2] * (r_ ** 3) + dist_s_[3] * (r_**4)) \
             + dist_s_[5]*(r_ + 2*y_corrt**2) + 2*dist_s_[4]*x_corrt*y_corrt

    pts_dist = np.hstack((x_prime_, y_prime_, np.ones_like(x_prime_)))
    pix_dist_ = np.dot(intrinsic_matrix, pts_dist.T).T

    pix_dist_pd = pd.DataFrame(pix_dist_)
    pix_inside_idx = ((0 < pix_dist_pd[0]) & (pix_dist_pd[0] < resol_x) & (0 < pix_dist_pd[1]) & (pix_dist_pd[1] < resol_y)).values

    ind_final = np.logical_and(pix_inside_idx, angle_idx)

    ray_starts_ = np.dot(rot_ext, np.hstack((pts_dist, np.ones((len(pts_dist), 1)))).T).T
    ray_starts_filtered = ray_starts_[ind_final][:, :-1]

    rays_dir_ = pts_[ind_final] - ray_starts_filtered
    rays_all_of_them_ = np.hstack((ray_starts_filtered, rays_dir_))

    # visualize_camera(pts_[pix_inside_idx], pts_[ind_final])

    return rays_all_of_them_


def find_nearest_hit_pts(org_rays_, mesh_, scene_):
    '''
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
    找到距离三角面片的交点与最近的顶点的索引，这里用的距离是 Manhattan Distance，用于加速计算
    '''
    dis_pts2tri_1 = np.sum(np.abs(inter_pts_cors - tri_pts_cors_1), axis=1).reshape(-1, 1)
    dis_pts2tri_2 = np.sum(np.abs(inter_pts_cors - tri_pts_cors_2), axis=1).reshape(-1, 1)
    dis_pts2tri_3 = np.sum(np.abs(inter_pts_cors - tri_pts_cors_3), axis=1).reshape(-1, 1)

    dis_pts2tri_set = np.hstack((dis_pts2tri_1, dis_pts2tri_2, dis_pts2tri_3))
    idx_col = np.argmin(dis_pts2tri_set, axis=1)
    # print(dis_pts2tri_set)

    '''
    找到距离三角面片的交点与最近的顶点的索引
    注意：同一个三角面片只能被相交一次
    '''
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


def my_ray_casting(cam_loc_, rot_mat_, mesh_input_):
    """
    :param rot_mat_: 旋转矩阵
    :param cam_loc_: 相机外参
    :param mesh_input_: mesh文件路径，格式建议为 .ply
    :return:
    """

    # o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, point_show_normal=False)
    mesh_for_ray = o3d.t.geometry.TriangleMesh.from_legacy(mesh_input_)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_for_ray)

    points = np.asarray(mesh_input_.vertices)
    big_idx_map = np.zeros((len(points), len(cam_loc_)))

    # for i in np.arange(len(cam_loc)):
    # for test, i should be set as 120, 197, 220
    for i in np.arange(len(cam_loc_)):
        start = time.perf_counter()
        print("第n个相机", i)
        print(cam_loc_[i])
        rays_sets_2 = sen_pts_gen(points, cam_loc_[i], rot_mat_[i], dist)

        idx_inte_pts = find_nearest_hit_pts(rays_sets_2, mesh_input_, scene)
        end = time.perf_counter()
        print(i)
        print('计算可视相机数量时长:', end - start)

        # print(i)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points[idx_inte_pts])
        # pcd.colors = o3d.utility.Vector3dVector(colors[idx_inte_pts])
        # # pcd.normals = o3d.utility.Vector3dVector(normals[ans['primitive_ids'].numpy()])
        # o3d.visualization.draw_geometries([pcd])

        big_idx_map[idx_inte_pts, i] = 1

    return big_idx_map


def pts_cam_ang(cam_locs_, pts_locs_, pts_norms_, thre_=30):
    """
    :param idx_map_: 符合可视性检查的相机和三维点的对应索引表 NxM
    :param cam_locs_: 相机坐标 Mx3
    :param pts_locs_: 点坐标 Nx3
    :param pts_norms_: 点的法向量 Nx3
    :param thre_: 相机与局部法向量夹角限制
    :return: 相机和空间点的夹角的索引表

    注意：
    在计算相机 ray 和 local 法向量的夹角过程中，暂时并未考虑相机畸变模型，后期可以考虑加上
    """

    cam_x_, pts_x_ = np.meshgrid(cam_locs_[:, 0], pts_locs_[:, 0])
    cam_y_, pts_y_ = np.meshgrid(cam_locs_[:, 1], pts_locs_[:, 1])
    cam_z_, pts_z_ = np.meshgrid(cam_locs_[:, 2], pts_locs_[:, 2])

    cam_big_mat_ = np.dstack((cam_x_, np.dstack((cam_y_, cam_z_)))).transpose(2, 0, 1)
    pts_big_mat_ = np.dstack((pts_x_, np.dstack((pts_y_, pts_z_)))).transpose(2, 0, 1)

    cam_x_, norms_x_ = np.meshgrid(cam_locs_[:, 0], pts_norms_[:, 0])
    cam_y_, norms_y_ = np.meshgrid(cam_locs_[:, 1], pts_norms_[:, 1])
    cam_z_, norms_z_ = np.meshgrid(cam_locs_[:, 2], pts_norms_[:, 2])

    norms_big_mat_ = np.dstack((norms_x_, np.dstack((norms_y_, norms_z_)))).transpose(2, 0, 1)

    rays_ = cam_big_mat_ - pts_big_mat_

    dot_pr = np.sum(np.multiply(rays_, norms_big_mat_), axis=0)
    norms_cal_ = np.linalg.norm(rays_, axis=0, keepdims=True) * np.linalg.norm(norms_big_mat_, axis=0, keepdims=True)

    # print(cam_big_mat_)
    # print(pts_big_mat_)

    print(dot_pr)
    print(norms_cal_)
    # print(np.linalg.norm(cam_big_mat_, axis=2, keepdims=True))

    # print(np.multiply(cam_big_mat_, pts_big_mat_))
    angs_ = np.rad2deg(np.arccos(dot_pr / norms_cal_[0]))
    print(np.min(angs_, axis=1))
    print(angs_.shape)

    angs_idx = angs_ < thre_
    print(angs_idx)

    return angs_idx


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


def mvs_error_scores(pts_locs_, cam_locs_, scale_fac_):
    mat_mask_ = np.triu(np.ones((len(cam_locs_), len(cam_locs_))))

    vets_ = cam_locs_ - pts_locs_
    dist_to_pts_ = np.linalg.norm(vets_, axis=1, keepdims=True)
    samp_rate_ = dist_to_pts_*scale_fac_

    cam_1_, cam_2_ = np.meshgrid(samp_rate_, samp_rate_)
    min_cam_samp_rate_ = np.maximum(cam_1_, cam_2_)
    min_cam_samp_rate_[np.eye(len(min_cam_samp_rate_), dtype=np.bool)] = 0
    cam_samp_new = mat_mask_*min_cam_samp_rate_

    cam_locs_1_ = np.expand_dims(cam_locs_, 0).repeat(len(cam_locs_), axis=0)
    cam_locs_1_after = cam_locs_1_.transpose(2, 0, 1)

    cam_locs_2_ = np.expand_dims(cam_locs_.T, 0).repeat(len(cam_locs_), axis=0)
    cam_locs_2_after = cam_locs_2_.transpose(1, 2, 0)

    print(cam_locs_1_after)
    print(cam_locs_2_after)

    dot_pr = np.sum(np.multiply(cam_locs_1_after, cam_locs_2_after), axis=0)
    norms_cal_ = np.linalg.norm(cam_locs_1_after, axis=0, keepdims=True) * np.linalg.norm(cam_locs_2_after, axis=0, keepdims=True)

    ang_ = np.rad2deg(np.arccos(dot_pr / norms_cal_[0]))
    # print(ang_)

    utility_ = np.where(ang_ < 20, 5, 15)

    factor_ = -1*((ang_-20)**2)/(2*utility_**2)
    factor_[np.eye(len(factor_), dtype=np.bool)] = 0
    factor_new = mat_mask_ * factor_

    weighted_ang = np.exp(factor_new)
    weighted_ang[np.eye(len(weighted_ang), dtype=np.bool)] = 0
    weighted_ang_new = mat_mask_ * weighted_ang
    print(weighted_ang)

    mvs_error_ = weighted_ang_new*cam_samp_new
    mvs_error_[mvs_error_ >= np.max(samp_rate_)] = 0
    # mvs_radius_ = 1/mvs_error_
    print(mvs_error_)

    return mvs_error_


def boolean_of_cone(cam_a_loc_, cam_b_loc_, pts_loc_):
    cone_a_ = useful_tools(cam_a_loc_, pts_loc_, np.array([0, 0, 1]), pix_size_=pixel_size, focal_=f)
    mesh_a_ = trimesh.Trimesh(vertices=np.asarray(cone_a_.vertices), faces=np.asarray(cone_a_.triangles))

    cone_b_ = useful_tools(cam_b_loc_, pts_loc_, np.array([0, 0, 1]), pix_size_=pixel_size, focal_=f)
    mesh_b_ = trimesh.Trimesh(vertices=np.asarray(cone_b_.vertices), faces=np.asarray(cone_b_.triangles))

    mesh_intersect_ = trimesh.boolean.intersection([mesh_a_, mesh_b_], engine='scad')
    mesh_intersect_renew_ = trimesh.convex.convex_hull(mesh_intersect_, qhull_options='Qt')

    return mesh_intersect_renew_


def cam_boolean(pts_locs_, cam_locs_, mvs_error_):
    sorted_ = np.argsort(mvs_error_.flatten())
    sorted_idx_ = np.array([divmod(sorted_[0], mvs_error_.shape[1]),
                            divmod(sorted_[1], mvs_error_.shape[1]),
                            divmod(sorted_[2], mvs_error_.shape[1])])

    print(sorted_idx_)

    int_1_ = boolean_of_cone(pts_locs_, cam_locs_[0, 0], cam_locs_[0, 1])
    int_2_ = boolean_of_cone(pts_locs_, cam_locs_[1, 0], cam_locs_[1, 1])
    int_3_ = boolean_of_cone(pts_locs_, cam_locs_[2, 0], cam_locs_[2, 1])

    int_1_2_ = trimesh.boolean.intersection([int_1_, int_2_], engine='scad')
    int_1_2_renew_ = trimesh.convex.convex_hull(int_1_2_, qhull_options='Qt')
    int_2_3_ = trimesh.boolean.intersection([int_2_, int_3_], engine='scad')
    int_2_3_renew_ = trimesh.convex.convex_hull(int_2_3_, qhull_options='Qt')

    int_final_ = trimesh.boolean.intersection([int_1_2_renew_, int_2_3_renew_], engine='scad')
    int_final_renew_ = trimesh.convex.convex_hull(int_final_, qhull_options='Qt')

    return int_final_renew_


if __name__ == '__main__':
    '''
    大疆Phantom 4 Pro
    传感器大小：1英寸 13.2 mm x 8.8 mm
    分辨率：5472×3648
    像元大小：2.4123 um
    焦距：8.8 mm
    FOV：84° 对角线分辨率
    '''

    w, h = 13.2, 8.8
    f = 8.8
    fov = 84
    fov_w = np.arctan(w / 2 / f) / np.pi * 180 * 2
    fov_h = np.arctan(h / 2 / f) / np.pi * 180 * 2

    resol_x, resol_y = 5472, 3648
    cx, cy = -26.1377554238884, -14.8594719360401
    f_xy = 3685.25307322617
    pixel_size = np.average([w / resol_x, h / resol_y])

    # dist: 畸变参数 [k1, k2, k3, k4, p1, p2, b1, b2]
    # b1, b2 是 affinity and non-orthogonality (skew) coefficients
    dist = np.array(
        [-0.288928920598278, 0.145903038241546, -0.0664869742590238, 0.0155044924834934, -0.000606112069582838,
         0.000146688084883612, 0.238532277878522, -0.464831768588501])

    bx, by = dist[-2], dist[-1]

    intrinsic_matrix = [[f_xy + bx, by, resol_x / 2 + cx],
                        [0, f_xy, resol_y / 2 + cy],
                        [0, 0, 1]]
    print(intrinsic_matrix)

    mesh_test = o3d.io.read_triangle_mesh("../data/zehao/plys/2.ply")
    mesh_vertices = np.asarray(mesh_test.vertices)
    norms = np.asarray(mesh_test.vertex_normals)
    print(norms)
    # mesh_test = o3d.io.read_triangle_mesh("../data/zehao/plys/UAV_only_B_zone.glb")
    # print("Try to render a mesh with normals (exist: " +
    #       str(mesh_test.has_vertex_normals()) + ") and colors (exist: " +
    #       str(mesh_test.has_vertex_colors()) + ")")
    # o3d.visualization.draw_geometries([mesh_test])
    # print("A mesh with no normals and no colors does not look good.")

    cam_data = pd.read_csv("../data/zehao/cameras/25m30d90o.csv", encoding="utf-8")
    # print(data.head(5))
    cam_loc = cam_data[["X", "Y", "Z"]].values
    euler_ang = cam_data[["R", "P", "H"]].values * np.array([[1, 1, -1]]) + np.array([[0, 180, 0]])
    print(euler_ang[0])
    rot_mat_set = R.from_euler('yxz', euler_ang, degrees=True)

    idx_of_visi_ = my_ray_casting(cam_loc, rot_mat_set, mesh_test)
    ang_idx = pts_cam_ang(cam_loc, mesh_vertices, norms)
    print(ang_idx.shape, idx_of_visi_.shape)
    final_idx = ang_idx.astype(int) & idx_of_visi_.astype(int)
    # print(np.sum(final_idx, axis=1))

    # aaa = np.sum(final_idx, axis=1)
    # np.savetxt('test.tx2', aaa)

    new_data = np.zeros((len(mesh_vertices), 7))

    new_data[:, :3] = mesh_vertices
    new_data[:, 3:6] = np.asarray(mesh_test.vertex_colors)
    new_data[:, -1] = np.sum(idx_of_visi_, axis=1)

    np.savetxt('directly_.txt', new_data)
    # test_duplicate()


# if __name__ == '__main__':
#     '''
#     大疆Phantom 4 Pro
#     传感器大小：1英寸 13.2 mm x 8.8 mm
#     分辨率：5472×3648
#     像元大小：2.4123 um
#     焦距：8.8 mm
#     FOV：84° 对角线分辨率
#     '''
#     w, h = 13.2, 8.8
#     f = 8.8
#     fov = 84
#     fov_w = np.arctan(w / 2 / f) / np.pi * 180 * 2
#     fov_h = np.arctan(h / 2 / f) / np.pi * 180 * 2
#
#     resol_x, resol_y = 5472, 3648
#     cx, cy = -26.1377554238884, -14.8594719360401
#     f_xy = 3685.25307322617
#     pixel_size = np.average([w / resol_x, h / resol_y])
#
#     # dist: 畸变参数 [k1, k2, k3, k4, p1, p2, b1, b2]
#     # b1, b2 是 affinity and non-orthogonality (skew) coefficients
#     dist = np.array(
#         [-0.288928920598278, 0.145903038241546, -0.0664869742590238, 0.0155044924834934, -0.000606112069582838,
#          0.000146688084883612, 0.238532277878522, -0.464831768588501])
#
#     bx, by = dist[-2], dist[-1]
#
#     intrinsic_matrix = [[f_xy + bx, by, resol_x / 2 + cx],
#                         [0, f_xy, resol_y / 2 + cy],
#                         [0, 0, 1]]
#     print(intrinsic_matrix)
#
#     nx, ny = (3, 3)
#     x = np.linspace(-10, 10, nx)
#     # [0. 1. 2.]
#     y = np.linspace(-10, 10, ny)
#     # [0. 1. 2.]
#     xv, yv = np.meshgrid(x, y)
#     print(xv.ravel())
#     #[ 0.  1.  2.  0.  1.  2.  0.  1.  2.]
#     print(yv.ravel())
#     #[ 0.  0.  0.  1.  1.  1.  2.  2.  2.]
#
#     cam_locs = np.hstack((np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1))), np.ones(xv.size).reshape(-1, 1)*10))
#
#     print(cam_locs)
#
#     pts_loc = np.array([[0, 0, 0]])
#
#     pts_norm = np.array([[0, 0, 1]])
#
#     cam_loc = np.array([[0, 0, 5],
#                         [1, 1, 5],
#                         [2, 2, 4],
#                         [3, 3, 5],
#                         [-1, -1, 3],
#                         [-2, -2, 5],
#                         [-3, -3, 4]])
#
#     mvs_error = mvs_error_scores(pts_loc, cam_loc, 1)
#     #
#     # pts_loc = np.array([[0, 0, 0],
#     #                     [1, 1, 0],
#     #                     [2, 2, 0],
#     #                     [3, 3, 0],
#     #                     [-1, -1, 0],
#     #                     [-2, -2, 0],
#     #                     [-3, -3, 0]])
#     #
#     # pts_norm = np.array([[0, 0, 1],
#     #                     [0, 0, 1],
#     #                     [0, 0, 1],
#     #                     [0, 0, 1],
#     #                     [0, 0, 1],
#     #                     [0, 0, 1],
#     #                     [0, 0, 1]])
#
#     # pts_cam_ang(cam_locs, pts_loc, pts_norm)
#     #
#     # mesh_test = o3d.io.read_triangle_mesh("../data/zehao/plys/2.ply")
#     #
#     # pts = np.asarray(mesh_test.vertices)
#     # norms = np.asarray(mesh_test.vertex_normals)
#     #
#     # data = pd.read_csv("../data/zehao/cameras/25m30d90o.csv", encoding="utf-8")
#     # # print(data.head(5))
#     # cam_locs = data[["X", "Y", "Z"]].values
#     #
#     # print(norms)
#     #
#     # out_ang = pts_cam_ang(cam_locs, pts, norms)
#     #
#     # np.savetxt('ang_mat_ind.txt', out_ang)
#
#     # mesh_test = o3d.io.read_triangle_mesh("../data/zehao/plys/UAV_only_B_zone.glb")
#     # print("Try to render a mesh with normals (exist: " +
#     #       str(mesh_test.has_vertex_normals()) + ") and colors (exist: " +
#     #       str(mesh_test.has_vertex_colors()) + ")")
#     # o3d.visualization.draw_geometries([mesh_test])
#     # print("A mesh with no normals and no colors does not look good.")
