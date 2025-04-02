import numpy as np

import analysis.utils as ut
from analysis.kernel import ker
from art.art import Figure


def convertXY(edge_type, r, x1, y1, t1, x2, y2, t2) -> np.ndarray:
    arr = ut.CArrayF(np.array([x1, y1, x2, y2]))
    ker.dll.convertXY(edge_type, r, t1, t2, arr.ptr)
    return arr.data


def showTypedDelaunay(f: Figure, delaunay, xyt: np.ndarray):
    colors = ['red', 'magenta', 'blue']
    r = 1 - 1 / delaunay.gamma
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
