import numpy as np
from matplotlib import pyplot as plt

import analysis.utils as ut
from analysis.voronoi import Voronoi


def VoroVisualize(voro: Voronoi):
    output = voro.true_voronoi()
    plt.scatter(voro.disk_map[:, 0], voro.disk_map[:, 1])
    for edge in output:
        plt.plot([edge['x1'], edge['x2']], [edge['y1'], edge['y2']], 'b-')
        x1, y1 = voro.disk_map[edge['id1']]
        x2, y2 = voro.disk_map[edge['id2']]
        plt.plot([x1, x2], [y1, y2], 'r-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Voronoi Edges')
    plt.show()


# 示例输入数据
num_points = 100
raw_input_points = ut.CArray((np.random.rand(num_points, 2) - 0.5) * 15, dtype=np.float32)
raw_input_tu = np.random.rand(num_points, 2) * 2 * np.pi
xytu = np.hstack((raw_input_points.data, raw_input_tu))
voro = Voronoi(1.3, 10, 10, xytu)

d = voro.true_delaunay()

# 使用matplotlib进行可视化
VoroVisualize(voro)
