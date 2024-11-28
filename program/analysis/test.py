import numpy as np

import utils as ut
from voro_visualize import VoroVisualize
from voronoi import Voronoi

# 示例输入数据
num_points = 100
raw_input_points = ut.CArray((np.random.rand(num_points, 2) - 0.5) * 15, dtype=np.float32)
raw_input_tu = np.random.rand(num_points, 2) * 2 * np.pi
xytu = np.hstack((raw_input_points.data, raw_input_tu))
voro = Voronoi(1.3, 10, 10, xytu)

d = voro.true_delaunay()

# 使用matplotlib进行可视化
VoroVisualize(voro)
