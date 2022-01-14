import pyvista as pv
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import time
from numpy.linalg import norm
import trimesh
import gc
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


def gaussian_mvs(data_):
    sigma_1_ = 5
    sigma_2_ = 15
    result_set = []
    for i in data_:
        if i <= 20:
            res_tmp = np.exp(-(i-20)**2/(2*sigma_1_**2))
            result_set.append(res_tmp)
        else:
            res_tmp = np.exp(-(i-20)**2/(2*sigma_2_**2))
            result_set.append(res_tmp)

    return result_set


print(np.exp(-20**2/(2*5**2)), np.exp(-20**2/(2*15**2)))

x = np.linspace(0, 80, num=91*4)
y = gaussian_mvs(x)

plt.style.use('seaborn-dark-palette')
plt.tight_layout()
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=3, c="black")

# ax.set_title('Gaussian weight to the angle', fontsize=16)
ax.set_xlabel('Angle between two viewing rays (deg)', fontsize=14)
ax.set_ylabel('Gaussian weight', fontsize=14)
ax.grid()

# fig.savefig("Gaussian weight to the angle.png", dpi=600)
plt.show()


