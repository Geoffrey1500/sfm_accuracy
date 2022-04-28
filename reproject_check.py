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

# print(np.arctan((13.2/2)/8.8)/np.pi*180*2)

w, h = 13.2, 8.8
f = 8.8
fov = 84
fov_w = np.arctan(w/2/f)/np.pi*180*2
fov_h = np.arctan(h/2/f)/np.pi*180*2
print(fov_w)
print(fov_h)
resol_x, resol_y = 5472, 3648
pixel_size = np.average([w/resol_x, h/resol_y])
intrinsic_matrix = [[3685.25307322617, 0, resol_x / 2 - 26.1377554238884],
                    [0, 3685.25307322617, resol_y / 2 - 14.8594719360401],
                    [0, 0, 1]]
print(intrinsic_matrix)
# dist: 畸变参数 [k1, k2, k3, k4, p1, p2]
dist = np.array([-0.288928920598278, 0.145903038241546, -0.0664869742590238, 0.0155044924834934, -0.000606112069582838, 0.000146688084883612])


def angle_between_vectors(v1_, v2_):
    dot_pr = np.sum(v1_*v2_, axis=1).reshape((-1, 1))
    norms = np.linalg.norm(v1_, axis=1, keepdims=True) * np.linalg.norm(v2_, axis=1, keepdims=True)

    return np.rad2deg(np.arccos(dot_pr / norms))


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
    angle_idx = angles.flatten() <= fov/2

    pts_cam_new = pts_cam_[:, :3]/(pts_cam_[:, -2].reshape((-1, 1)))

    x_corrt = pts_cam_new[:, 0].reshape((-1, 1))
    y_corrt = pts_cam_new[:, 1].reshape((-1, 1))
    r_ = x_corrt ** 2 + y_corrt ** 2

    # x_dist = x_corrt * ((1 + dist_s_[0] * r_ + dist_s_[1] * (r_ ** 2) + dist_s_[2] * (r_ ** 3))/(1 + dist_s_[3] * r_)) \
    #          + 2*dist_s_[4]*x_corrt*y_corrt + dist_s_[5]*(r_ + 2*x_corrt**2)
    # y_dist = y_corrt * ((1 + dist_s_[0] * r_ + dist_s_[1] * (r_ ** 2) + dist_s_[2] * (r_ ** 3))/(1 + dist_s_[3] * r_)) \
    #          + dist_s_[4]*(r_ + 2*y_corrt**2) + 2*dist_s_[5]*x_corrt*y_corrt

    x_dist = x_corrt * (1 + dist_s_[0] * r_ + dist_s_[1] * (r_ ** 2) + dist_s_[2] * (r_ ** 3)) \
             + 2*dist_s_[4]*x_corrt*y_corrt + dist_s_[5]*(r_ + 2*x_corrt**2)
    y_dist = y_corrt * (1 + dist_s_[0] * r_ + dist_s_[1] * (r_ ** 2) + dist_s_[2] * (r_ ** 3)) \
             + dist_s_[4]*(r_ + 2*y_corrt**2) + 2*dist_s_[5]*x_corrt*y_corrt

    pts_dist = np.hstack((x_dist, y_dist, np.ones_like(x_dist)))
    pix_dist_ = np.dot(intrinsic_matrix, pts_dist.T).T

    # pix_dist_pd = pd.DataFrame(pix_dist_)
    # pix_inside_idx = ((0 <= pix_dist_pd[0]) & (pix_dist_pd[0] < resol_x) & (0 <= pix_dist_pd[1]) & (pix_dist_pd[1] < resol_y)).values

    pix_inside_idx = np.where((0 <= pix_dist_[:, 0]) & (pix_dist_[:, 0] < resol_x) & (0 <= pix_dist_[:, 1]) & (pix_dist_[:, 1] < resol_y), True, False)
    # pix_dist_pd = pd.DataFrame(np.rint(pix_dist_))
    # pix_du_idx = pix_dist_pd.duplicated(keep='last').values
    #
    # ind_final = np.logical_and(pix_inside_idx, ~pix_du_idx)

    ind_final = np.logical_and(pix_inside_idx, angle_idx)

    ray_starts_ = np.dot(np.linalg.inv(rot_ext_2), np.hstack((pts_dist, np.ones((len(pts_dist), 1)))).T).T
    ray_starts_filtered = ray_starts_[ind_final][:, :-1]

    rays_dir_ = pts_[ind_final] - ray_starts_filtered
    rays_all_of_them_ = np.hstack((ray_starts_filtered, rays_dir_))

    visualize_camera(pts_[pix_inside_idx], pts_[ind_final])

    return rays_all_of_them_


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
    scale_factor_ = 3
    points_per_side_ = 30
    x_ = np.linspace((-w / 2 + pixel_size / 2) * scale_factor_, (w / 2 - pixel_size / 2) * scale_factor_,
                     int(3 * points_per_side_ * scale_factor_))
    y_ = np.linspace((-h / 2 + pixel_size / 2) * scale_factor_, (h / 2 - pixel_size / 2) * scale_factor_,
                     int(2 * points_per_side_ * scale_factor_))
    x_, y_ = np.meshgrid(x_, y_)

    x_ = x_.reshape((-1, 1))
    y_ = y_.reshape((-1, 1))
    z_ = (np.zeros_like(x_) - f)

    points_co = np.hstack((x_, y_, z_))
    rot_mat_set = R.from_euler('yxz', np.array([0, 180, -220]), degrees=True)
    cam_location = np.array([0, 0, 0])

    sen_pts_gen(points_co, cam_location, rot_mat_set, dist)



if __name__ == '__main__':
    my_ray_casting()
    # test_duplicate()
