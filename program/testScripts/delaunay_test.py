import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from analysis.voronoi import Voronoi
from art.art import Figure
from art.delaunay_art import showTypedDelaunay
from art.viewer import RenderState, RenderSetup

gamma = 2.5
A = 32
B = 16
use_segment_delaunay = True
use_modulo = True
show_types = True
show_hull = True

if __name__ == '__main__':
    xyt = np.array(pd.read_csv('testScripts/example_data.csv', header=None))
    with Figure() as f:
        RenderState(f).drawParticles(xyt, {'gamma': gamma, 'A': A, 'B': B},
                                     RenderSetup('convex_hull'), with_label=False)
        if use_segment_delaunay:
            voro = Voronoi(gamma, np.max(np.abs(xyt[:, 0])), np.max(np.abs(xyt[:, 1])), xyt)
            if use_modulo:
                delaunay = voro.delaunay()
                if show_types:
                    showTypedDelaunay(f, delaunay, xyt)
                else:
                    for i, j, _ in delaunay.iter_edges():
                        x1, y1 = xyt[i, :2]
                        x2, y2 = xyt[j, :2]
                        f.ax.plot([x1, x2], [y1, y2], color='blue')
            else:
                delaunay = Delaunay(voro.disk_map)
                xy = voro.disk_map
                f.ax.triplot(xy[:, 0], xy[:, 1], delaunay.simplices.copy())
        else:
            delaunay = Delaunay(xyt[:, :2])
            f.ax.triplot(xyt[:, 0], xyt[:, 1], delaunay.simplices.copy())
