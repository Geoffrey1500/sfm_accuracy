import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("Sparse.ply")
diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=5, origin=[0, 0, 0])

camera = [-21.90558665, -13.96051597,  31.58213869] # 定义用于隐藏点删除的参数，获取从给定视图中可见的所有点，可视化结果
radius = 52078

_, pt_map = pcd.hidden_point_removal(camera, radius)

pcd_new = pcd.select_by_index(pt_map)

view_point = o3d.geometry.TriangleMesh.create_sphere()
view_point.translate(np.array([camera]).T)

o3d.visualization.draw_geometries([pcd_new, mesh_frame, view_point])
