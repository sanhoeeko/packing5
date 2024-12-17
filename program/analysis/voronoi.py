import numpy as np

from . import utils as ut
from .kernel import ker


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

    def delaunay(self, weighted: bool):
        if weighted:
            return self.weighted_delaunay()
        else:
            return self.true_delaunay()


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

    def z_number(self, arg=None):
        if self.weighted:
            result = ut.CArrayFZeros((self.num_rods,))
            one_weights = ut.CArray(np.ones((self.num_edges,), dtype=np.float32))
            ker.dll.sumOverWeights(
                self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr, one_weights.ptr, result.ptr
            )
            return result
        else:
            return self.weight_sums

    def phi_p(self, p: int, xyt: ut.CArray) -> np.ndarray[np.complex64]:
        """
        Assume that p is an even number. Because we use (-z)^4 = z^4, (-z)^6 = z^6
        """
        z_p = ut.CArray(np.zeros((self.num_edges,), dtype=np.complex64))
        Phi = ut.CArray(np.zeros((self.num_rods,), dtype=np.complex64))
        ker.dll.z_ij_power_p(self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr,
                             xyt.ptr, z_p.ptr, np.float32(p))
        z_p.data *= self.weights.data
        ker.dll.sumComplex(self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr, z_p.ptr, Phi.ptr)
        return Phi.data / self.weight_sums.data

    def Phi6Complex(self, xyt: ut.CArray) -> np.ndarray[np.complex64]:
        return self.phi_p(6, xyt)

    def Phi4Complex(self, xyt: ut.CArray) -> np.ndarray[np.complex64]:
        return self.phi_p(4, xyt)

    def Phi6(self, xyt: ut.CArray) -> np.ndarray:
        return np.abs(self.phi_p(6, xyt))

    def Phi4(self, xyt: ut.CArray) -> np.ndarray:
        return np.abs(self.phi_p(4, xyt))

    def phi_p_ellipse(self, p: int, gamma: float, xyt: ut.CArray) -> np.ndarray[np.complex64]:
        pass

    def S_center(self, xyt: ut.CArray) -> np.ndarray:
        """
        Use the orientation of the centering particle, θ_i, as director.
        """
        ti_tj = ut.CArrayFZeros((self.num_edges,))
        ker.dll.orientation_diff_ij(self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr, xyt.ptr, ti_tj.ptr)
        c = ut.CArray(np.cos(2 * ti_tj.data))
        S = ut.CArrayFZeros((self.num_rods,))
        ker.dll.sumOverWeights(self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr, c.ptr, S.ptr)
        return S.data / self.weight_sums.data

    def Q_tensor(self, xyt: ut.CArray) -> (ut.CArray, ut.CArray):
        """
        Q = ∑ 2(u @ u.T - 1) = [ ∑ cos 2t, ∑ sin 2t; ∑ sin 2t, -∑ cos 2t]
        The sum (can be weighted) is taken over neighbors.
        Here we define:
        sum_ux[i] = ∑[j] cos 2t[i,j]
        sum_uy[i] = ∑[j] sin 2t[i,j]
        """
        t_mul_2 = 2 * xyt.data[:, 2]
        ux = ut.CArray(np.cos(t_mul_2))
        uy = ut.CArray(np.sin(t_mul_2))
        sum_ux = ux.copy()
        sum_uy = uy.copy()
        ker.dll.sumOverNeighbors(self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr, ux.ptr, sum_ux.ptr)
        ker.dll.sumOverNeighbors(self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr, uy.ptr, sum_uy.ptr)
        if self.weighted:
            # TODO: bug
            sum_ux.data *= self.weights.data
            sum_uy.data *= self.weights.data
        return sum_ux, sum_uy

    def S_local(self, xyt: ut.CArray) -> np.ndarray:
        """
        Use the director as the eigenvector of Q-tensor, then S_local is the eigenvalue.
        """
        sum_ux, sum_uy = self.Q_tensor(xyt)
        S = np.sqrt(sum_ux.data ** 2 + sum_uy.data ** 2)
        return S.data / (self.weight_sums.data + 1)

    def director_angle(self, xyt: ut.CArray) -> np.ndarray:
        """
        Calculate the director as the eigenvector of Q-tensor.
        """
        sum_ux, sum_uy = self.Q_tensor(xyt)
        S = np.sqrt(sum_ux.data ** 2 + sum_uy.data ** 2)
        return np.atan2(sum_uy, sum_ux + S)
