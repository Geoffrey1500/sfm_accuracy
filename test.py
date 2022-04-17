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
    # Press Ctrl/Cmd-C in the visualization window to copy the current viewpoint
    o3d.visualization.draw_geometries([pcd.to_legacy()])
    # o3d.visualization.draw([pcd]) # new API


def my_ray_casting2():
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

    mesh_for_ray = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    cube_id = scene.add_triangles(mesh_for_ray)
    print(cube_id)

    rays_set = original_data[0:3, :6]

    rays = o3d.core.Tensor(rays_set,
                           dtype=o3d.core.Dtype.Float32)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=90,
        center=points[200],
        eye=cam_loc[200],
        up=[0, 1, 0],
        width_px=50,
        height_px=50,
    )
    # We can directly pass the rays tensor to the cast_rays function.
    ans = scene.cast_rays(rays)

    # hit = ans['t_hit'].isfinite()
    # points = rays[hit][:, :3] + rays[hit][:, 3:] * ans['t_hit'][hit].reshape((-1, 1))
    # pcd = o3d.t.geometry.PointCloud(points)
    # Press Ctrl/Cmd-C in the visualization window to copy the current viewpoint
    # o3d.visualization.draw_geometries([pcd.to_legacy()])
    # o3d.visualization.draw([pcd]) # new API

    original_triangle = np.asarray(mesh.triangles)
    print(original_triangle)
    hit = ans['primitive_ids'] != 4294967295
    index = ans['primitive_ids'][hit].numpy()
    # print(index)
    # print(hit)
    # print(original_triangle[index])

    hit_triangles = original_triangle[index]
    points2 = rays[hit][:, :3] + rays[hit][:, 3:] * ans['t_hit'][hit].reshape((-1, 1))
    print(points2)

    print(np.asarray(mesh.vertices)[hit_triangles[0]])

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points[ans['primitive_ids'].numpy()])
    # pcd.colors = o3d.utility.Vector3dVector(colors[ans['primitive_ids'].numpy()])
    # pcd.normals = o3d.utility.Vector3dVector(normals[ans['primitive_ids'].numpy()])
    # o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    my_ray_casting2()
