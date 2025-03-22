import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from analysis.voronoi import Voronoi, EllipsePoints
from art.art import Figure
from art.viewer import RenderState


def convertXY(edge_type, r, x1, y1, t1, x2, y2, t2):
    if edge_type == 0:  # head-to-head
        dx1, dy1 = r * np.cos(t1), r * np.sin(t1)
        x1l, x1r = x1 - dx1, x1 + dx1
        y1l, y1r = y1 - dy1, y1 + dy1
        dx2, dy2 = r * np.cos(t2), r * np.sin(t2)
        x2l, x2r = x2 - dx2, x2 + dx2
        y2l, y2r = y2 - dy2, y2 + dy2
        d_ll = (x1l - x2l) ** 2 + (y1l - y2l) ** 2
        d_lr = (x1l - x2r) ** 2 + (y1l - y2r) ** 2
        d_rl = (x1r - x2l) ** 2 + (y1r - y2l) ** 2
        d_rr = (x1r - x2r) ** 2 + (y1r - y2r) ** 2
        idx = np.argmin([d_ll, d_lr, d_rl, d_rr])
        if idx == 0:
            x1, y1, x2, y2 = x1l, y1l, x2l, y2l
        elif idx == 1:
            x1, y1, x2, y2 = x1l, y1l, x2r, y2r
        elif idx == 2:
            x1, y1, x2, y2 = x1r, y1r, x2l, y2l
        elif idx == 3:
            x1, y1, x2, y2 = x1r, y1r, x2r, y2r
    elif edge_type == 1:  # head-to-side
        dx1, dy1 = r * np.cos(t1), r * np.sin(t1)
        x1l, x1r = x1 - dx1, x1 + dx1
        y1l, y1r = y1 - dy1, y1 + dy1
        dx2, dy2 = r * np.cos(t2), r * np.sin(t2)
        x2l, x2r = x2 - dx2, x2 + dx2
        y2l, y2r = y2 - dy2, y2 + dy2
        d_lc = (x1l - x2) ** 2 + (y1l - y2) ** 2
        d_rc = (x1r - x2) ** 2 + (y1r - y2) ** 2
        d_cl = (x1 - x2l) ** 2 + (y1 - y2l) ** 2
        d_cr = (x1 - x2r) ** 2 + (y1 - y2r) ** 2
        idx = np.argmin([d_lc, d_rc, d_cl, d_cr])
        if idx == 0:
            x1, y1, x2, y2 = x1l, y1l, x2, y2
        elif idx == 1:
            x1, y1, x2, y2 = x1r, y1r, x2, y2
        elif idx == 2:
            x1, y1, x2, y2 = x1, y1, x2l, y2l
        elif idx == 3:
            x1, y1, x2, y2 = x1, y1, x2r, y2r
    return x1, y1, x2, y2


gamma = 2.5
use_segment_delaunay = True
use_modulo = True
show_types = True

EllipsePoints(30, 15, 0.1)

if __name__ == '__main__':
    xyt = np.array(pd.read_csv('testScripts/example_data.csv', header=None))
    with Figure() as f:
        RenderState(f).drawParticles(xyt, {'gamma': gamma}, with_label=False)
        if use_segment_delaunay:
            voro = Voronoi(gamma, np.max(np.abs(xyt[:, 0])), np.max(np.abs(xyt[:, 1])), xyt)
            if use_modulo:
                delaunay = voro.delaunay()
                if show_types:
                    colors = ['red', 'magenta', 'blue']
                    r = 1 - 1 / gamma
                    for i, j, edge_type in delaunay.iter_edges():
                        x1, y1, t1 = xyt[i, :3]
                        x2, y2, t2 = xyt[j, :3]
                        x1, y1, x2, y2 = convertXY(edge_type, r, x1, y1, t1, x2, y2, t2)
                        f.ax.plot([x1, x2], [y1, y2], color=colors[edge_type])
                    for i in range(delaunay.num_rods):
                        x1, y1, t1 = xyt[i, :3]
                        dx1, dy1 = r * np.cos(t1), r * np.sin(t1)
                        x1l, x1r = x1 - dx1, x1 + dx1
                        y1l, y1r = y1 - dy1, y1 + dy1
                        f.ax.plot([x1l, x1r], [y1l, y1r], color='green', alpha=0.5)
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
