import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

import analysis.utils as ut
import default
from analysis.voronoi import Voronoi, DelaunayBase
from art.art import Figure
from art.viewer import RenderState

gamma = 2.5
use_segment_delaunay = True
use_python = True

if __name__ == '__main__':
    xyt = np.array(pd.read_csv('testScripts/example_data.csv', header=None))
    with Figure() as f:
        RenderState(f).drawParticles(xyt, {'gamma': gamma}, with_label=False)
        if use_segment_delaunay:
            if use_python:
                voro = Voronoi(gamma, np.max(np.abs(xyt[:, 0])), np.max(np.abs(xyt[:, 1])), xyt)
                delaunay = Delaunay(voro.disk_map.data)
                xy = voro.disk_map.data
                f.ax.triplot(xy[:, 0], xy[:, 1], delaunay.simplices.copy())
            # if default.if_using_legacy_delaunay:
            #     delaunay = DelaunayBase.legacy(xyt.shape[0], gamma, ut.CArray(xyt))
            # else:
            #     delaunay = Voronoi(gamma, np.max(np.abs(xyt[:, 0])), np.max(np.abs(xyt[:, 1])), xyt).delaunay(False)
            # for i, j in delaunay.iter_edges():
            #     x1, y1 = xyt[i, :2]
            #     x2, y2 = xyt[j, :2]
            #     f.ax.plot([x1, x2], [y1, y2], color='blue')
        else:
            delaunay = Delaunay(xyt[:, :2])
            f.ax.triplot(xyt[:, 0], xyt[:, 1], delaunay.simplices.copy())
