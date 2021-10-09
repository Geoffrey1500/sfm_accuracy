import pyvista as pv
import numpy as np

# Define a simple Gaussian surface
n = 20
x = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
y = np.linspace(-200, 200, num=n) + np.random.uniform(-5, 5, size=n)
xx, yy = np.meshgrid(x, y)
A, b = 100, 100
zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

# Get the points as a 2D NumPy array (N by 3)
points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
points[0:5, :]
print(points)

# simply pass the numpy points to the PolyData constructor
cloud = pv.PolyData(points)

surf = cloud.delaunay_2d()
surf.plot(show_edges=True)

start = [0, 0, 80]
stop = [0.25, 1, 110]

# Perform ray trace
points, ind = surf.ray_trace(start, stop)
print(points)

# Create geometry to represent ray trace
ray = pv.Line(start, stop)
intersection = pv.PolyData(points)

p = pv.Plotter()
p.add_mesh(surf,
           show_edges=True, opacity=1, color="w",
           lighting=False, label="Test Mesh")
p.add_mesh(ray, color="blue", line_width=5, label="Ray Segment")
p.add_mesh(intersection, color="maroon",
           point_size=25, label="Intersection Points")
p.add_legend()
p.show()