from functools import lru_cache

import numpy as np

import utils as ut
from kernel import ker


class Voronoi:
    def __init__(self, gamma: float, A: float, B: float, configuration: np.ndarray):
        self.gamma = gamma
        self.A, self.B = A, B
        self.configuration = configuration
        self.num_rods = configuration.shape[0]
        self.disks_per_rod = 3
        self.disk_map = ut.CArray(self.getDiskMap(configuration), dtype=np.float32)

    def getDiskMap(self, xytu: np.ndarray):
        xy = xytu[:, :2]
        t = xytu[:, 2]
        a = self.gamma - 1
        v = np.vstack([np.cos(t), np.sin(t)]).T
        left_xy = xy - a * v
        right_xy = xy + a * v
        return np.vstack([left_xy, xy, right_xy])

    def true_voronoi(self) -> np.ndarray:
        output = ut.CArray(np.zeros(self.num_rods * self.disks_per_rod * 8, dtype=[
            ('id1', np.int32), ('id2', np.int32),
            ('x1', np.float32), ('y1', np.float32), ('x2', np.float32), ('y2', np.float32)]))
        n_edges = ker.dll.disksToVoronoiEdges(self.num_rods, self.disks_per_rod,
                                              self.disk_map.ptr, output.ptr, self.A, self.B)
        return output.data[:n_edges]

    def delaunay_template(self, kernel_function) -> (ut.CArray, np.ndarray):
        """
        :param kernel_function: ker.dll.trueDelaunay | ker.dll.weightedDelaunay
        """
        output = ut.CArray(np.zeros(self.num_rods * 8, dtype=[
            ('id2', np.int32), ('weight', np.float32)]))
        indices = ut.CArray(np.zeros((self.num_rods,), dtype=np.int32))
        n_edges = kernel_function(self.num_rods, self.disks_per_rod,
                                  self.disk_map.ptr, output.ptr, indices.ptr, self.A, self.B)
        return indices, output.data[:n_edges]

    def true_delaunay(self):
        return Delaunay(False, *self.delaunay_template(ker.dll.trueDelaunay))

    def weighted_delaunay(self):
        return Delaunay(True, *self.delaunay_template(ker.dll.weightedDelaunay))


class Delaunay:
    def __init__(self, weighted: bool, indices: ut.CArray, weighted_edges: np.ndarray):
        self.weighted = weighted
        self.indices = indices
        self.num_rods = indices.data.shape[0]
        self.weight_sums = ut.CArrayFZeros((self.num_rods,))
        self.edges = ut.CArray(weighted_edges['id2'], dtype=np.int32)
        self.num_edges = self.edges.data.shape[0]
        self.weights = ut.CArray(weighted_edges['weight'], dtype=np.float32)
        ker.dll.sumOverWeights(self.num_edges, self.num_rods, self.indices.ptr,
                               self.edges.ptr, self.weights.ptr, self.weight_sums.ptr)

    def z_number(self):
        if self.weighted:
            result = ut.CArrayFZeros((self.num_rods,))
            one_weights = ut.CArray(np.ones((self.num_edges,), dtype=np.float32))
            ker.dll.sumOverWeights(
                self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr, one_weights.ptr, result.ptr
            )
            return result
        else:
            return self.weight_sums

    @lru_cache(maxsize=None)
    def theta_ij(self, xyt: ut.CArray):
        output = ut.CArrayFZeros((self.num_edges,))
        ker.dll.theta_ij(self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr, xyt.ptr, output.ptr)
        return output

    def phi_p(self, p: int, xyt: ut.CArray) -> np.ndarray:
        u = ut.CArray(np.exp(1j * p * self.theta_ij(xyt)) * self.weights)
        Phi = ut.CArrayFZeros((self.num_rods,))
        ker.dll.sumAntisym(self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr, u.ptr, Phi.ptr)
        return Phi.data / self.weight_sums

    def S_center(self, xyt: ut.CArray) -> np.ndarray:
        """
        Use the orientation of the centering particle, θ_i, as director.
        """
        ti_tj = ut.CArrayFZeros((self.num_edges,))
        ker.dll.orientation_diff_ij(self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr, xyt.ptr, ti_tj.ptr)
        c = np.cos(2 * ti_tj.data)
        S = ut.CArrayFZeros((self.num_rods,))
        ker.dll.sumOverWeights(self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr, c.ptr, S.ptr)
        return S.data / self.weight_sums

    def S_local(self, xyt: ut.CArray) -> np.ndarray:
        """
        Calculate the director as the eigenvector of Q-tensor.
        """
        pass