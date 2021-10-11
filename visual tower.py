import pyvista as pv

mesh_tower = pv.read("Model.ply")
mesh_tower.scale([1000, 1000, 1000])
plotter = pv.Plotter()
plotter.add_mesh(mesh_tower, show_edges=True, color="white")
plotter.show()
