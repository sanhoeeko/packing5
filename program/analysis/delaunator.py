import numpy as np

from . import utils as ut
from .kernel import ker

max_neighbors = 32


class Delaunay:
    def __init__(self, points: np.ndarray):
        self.points = points.astype(np.float32)
        self._points_c = ut.CArray(self.points)
        self.n_points = self.points.shape[0]
        self.indices = ut.CArray(np.zeros((self.n_points + 1,), dtype=np.int32))
        self.edges = ut.CArray(np.zeros((self.n_points * max_neighbors * 2,), dtype=np.int32))
        self.convex_hull = None  # we don't use it
        self.num_edges_2 = ker.dll.CreateDelaunay(self.n_points, self._points_c.ptr, self.indices.ptr,
                                                  self.edges.ptr, 0)

    @property
    def vertex_neighbor_vertices(self) -> (np.ndarray[np.int32], np.ndarray[np.int32]):
        return self.indices.data, self.edges.data[:self.num_edges_2]
