import matplotlib.pyplot as plt

from voronoi import Voronoi


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
