import pyvista as pv
from pyvista import examples
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


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



def sensor_plane_point(points_per_side_=500, scale_factor_=1):
    # To create sensor plane

    x_ = np.linspace((-w/2+pixel_size/2)*scale_factor_, (w/2-pixel_size/2)*scale_factor_, 3*points_per_side_)
    y_ = np.linspace((-h/2+pixel_size/2)*scale_factor_, (h/2-pixel_size/2)*scale_factor_, 2*points_per_side_)
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


def show_pv():
    # Get a sample file
    mesh = pv.read("data/UAV_only.ply")
    # cpos = mesh.plot()

    # print(mesh.points)
    # print(mesh.faces)
    #
    #
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, point_size=1, rgb=True)
    plotter.show()


def show_trimesh():
    mesh_B = trimesh.load('data/UAV_only.ply', force='mesh')
    mesh_B.show()

    print(mesh_B.vertex_normals)


def ray_casting():
    # Load mesh and convert to open3d.t.geometry.TriangleMesh
    cube = o3d.geometry.TriangleMesh.create_box().translate([0, 0, 0])
    cube = o3d.t.geometry.TriangleMesh.from_legacy(cube)

    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    cube_id = scene.add_triangles(cube)
    print(cube_id)

    # We create two rays:
    # The first ray starts at (0.5,0.5,10) and has direction (0,0,-1).
    # The second ray start at (-1,-1,-1) and has direction (0,0,-1).
    rays = o3d.core.Tensor([[0.5, 0.5, 10, 0, 0, -1], [-1, -1, -1, 0, 0, -1]],
                           dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)
    print(ans.keys())
    print(ans['t_hit'].numpy(), ans['geometry_ids'].numpy())

    # Create meshes and convert to open3d.t.geometry.TriangleMesh
    cube = o3d.geometry.TriangleMesh.create_box().translate([0, 0, 0])
    cube = o3d.t.geometry.TriangleMesh.from_legacy(cube)
    torus = o3d.geometry.TriangleMesh.create_torus().translate([0, 0, 2])
    torus = o3d.t.geometry.TriangleMesh.from_legacy(torus)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5).translate(
        [1, 2, 3])
    sphere = o3d.t.geometry.TriangleMesh.from_legacy(sphere)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)
    scene.add_triangles(torus)
    _ = scene.add_triangles(sphere)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=90,
        center=[0, 0, 2],
        eye=[2, 3, 0],
        up=[0, 1, 0],
        width_px=640,
        height_px=480,
    )
    # We can directly pass the rays tensor to the cast_rays function.
    ans = scene.cast_rays(rays)

    plt.imshow(ans['t_hit'].numpy())

    # use abs to avoid negative values
    plt.imshow(np.abs(ans['primitive_normals'].numpy()))

    plt.imshow(ans['geometry_ids'].numpy(), vmax=3)

    hit = ans['t_hit'].isfinite()
    points = rays[hit][:, :3] + rays[hit][:, 3:] * ans['t_hit'][hit].reshape((-1, 1))
    pcd = o3d.t.geometry.PointCloud(points)
    # Press Ctrl/Cmd-C in the visualization window to copy the current viewpoint
    o3d.visualization.draw_geometries([pcd.to_legacy()],
                                      front=[0.5, 0.86, 0.125],
                                      lookat=[0.23, 0.5, 2],
                                      up=[-0.63, 0.45, -0.63],
                                      zoom=0.7)
    # o3d.visualization.draw([pcd]) # new API


def my_ray_casting():
    data = pd.read_csv("data/UAV_only.csv")
    print(data.head(5))
    cam_loc = data[["x", "y", "z"]].values
    euler_ang = data[["heading", "pitch", "roll"]].values * np.array([[-1, 1, 1]]) + np.array([[0, 0, 0]])
    rot_mat_set = R.from_euler('ZXY', euler_ang, degrees=True)

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

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    cube_id = scene.add_triangles(mesh)
    print(cube_id)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=90,
        center=points[200],
        eye=cam_loc[70],
        up=[0, 1, 0],
        width_px=resol_x,
        height_px=resol_y,
    )
    # We can directly pass the rays tensor to the cast_rays function.
    ans = scene.cast_rays(rays)

    hit = ans['t_hit'].isfinite()
    points = rays[hit][:, :3] + rays[hit][:, 3:] * ans['t_hit'][hit].reshape((-1, 1))
    pcd = o3d.t.geometry.PointCloud(points)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5, origin=[0, 0, 0])
    # Press Ctrl/Cmd-C in the visualization window to copy the current viewpoint
    o3d.visualization.draw_geometries([pcd.to_legacy()])
    # o3d.visualization.draw([pcd]) # new API


def my_ray_casting2():
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

    kkk = 220
    print(cam_loc[kkk].reshape((-1, 1)))
    ex_mat = np.vstack((np.hstack((rot_mat_set.as_matrix()[kkk], cam_loc[kkk].reshape((-1, 1)))), [[0, 0, 0, 1]]))
    # ex_mat = np.vstack((np.hstack(([[1, 0, 0],
    #                                 [0, 1, 0],
    #                                 [0, 0, -1]], cam_loc[kkk].reshape((-1, 1)))), [[0, 0, 0, 1]]))
    # print(ex_mat)
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=ex_mat,
        width_px=int(resol_x),
        height_px=int(resol_y),
    )

    # rays_direction = np.asarray(rot_mat_set[kkk].apply(sensor_plane_point(points_per_side_=1824)))
    # print(rays_direction)
    # rays_direction = rays_direction / rays_direction.max(axis=1).reshape((-1, 1))
    #
    # rays_starts = np.expand_dims(cam_loc[kkk], 0).repeat(len(rays_direction), axis=0)
    #
    # rays_sets = np.hstack((rays_starts, rays_direction))
    # print(rays_sets)
    # rays = o3d.core.Tensor(rays_sets,
    #                        dtype=o3d.core.Dtype.Float32)

    # We can directly pass the rays tensor to the cast_rays function.
    ans = scene.cast_rays(rays)

    # hit = ans['t_hit'].isfinite()
    # points = rays[hit][:, :3] + rays[hit][:, 3:] * ans['t_hit'][hit].reshape((-1, 1))
    # pcd = o3d.t.geometry.PointCloud(points)
    # Press Ctrl/Cmd-C in the visualization window to copy the current viewpoint
    # o3d.visualization.draw_geometries([pcd.to_legacy()])
    # o3d.visualization.draw([pcd]) # new API

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


def visualize_camera():
    data = pd.read_csv("data/UAV_only.csv")
    print(data.head(5))
    cam_loc = data[["x", "y", "z"]].values
    euler_ang = data[["roll", "pitch", "heading"]].values * np.array([[1, 1, -1]])
    rot_mat_set = R.from_euler('yxz', euler_ang, degrees=True)

    kkk = 252
    print(euler_ang[kkk])
    print(rot_mat_set[kkk].as_matrix())
    blue = np.array([0, 0, 1])
    points_tar = np.asarray(rot_mat_set[kkk].apply(sensor_plane_point(points_per_side_=30)))
    colors_tar = np.expand_dims(blue, 0).repeat(len(points_tar), axis=0)
    pcd_tar = o3d.geometry.PointCloud()
    pcd_tar.points = o3d.utility.Vector3dVector(points_tar)
    pcd_tar.colors = o3d.utility.Vector3dVector(colors_tar)

    red = np.array([1, 0, 0])
    rot_mat_ref = R.from_euler('yxz', [[0, 0, euler_ang[kkk, -1]]], degrees=True)
    points_ref = np.asarray(rot_mat_ref[0].apply(sensor_plane_point(points_per_side_=30)))
    colors_ref = np.expand_dims(red, 0).repeat(len(points_ref), axis=0)
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(points_ref)
    pcd_ref.colors = o3d.utility.Vector3dVector(colors_ref)

    green = np.array([0, 1, 0])
    rot_mat_ref_2 = R.from_euler('yxz', [[0, 0, 0]], degrees=True)
    points_ref_2 = np.asarray(rot_mat_ref_2[0].apply(sensor_plane_point(points_per_side_=30)))
    colors_ref_2 = np.expand_dims(green, 0).repeat(len(points_ref_2), axis=0)
    pcd_ref_2 = o3d.geometry.PointCloud()
    pcd_ref_2.points = o3d.utility.Vector3dVector(points_ref_2)
    pcd_ref_2.colors = o3d.utility.Vector3dVector(colors_ref_2)

    o3d.visualization.draw_geometries([pcd_ref, pcd_tar, pcd_ref_2])


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

    kkk = 220
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
    my_ray_casting3()
