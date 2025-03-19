import numpy as np

from . import utils as ut
from .kernel import ker


class Voronoi:
    d = 0.05

    def __init__(self, gamma: float, A: float, B: float, configuration: np.ndarray):
        self.gamma = gamma
        self.A, self.B = A, B
        self.configuration = configuration
        self.num_rods = configuration.shape[0]
        self.disks_per_rod = int(1 + 2 * (gamma - 1) / Voronoi.d)
        self.disk_map = ut.CArray(self.getDiskMap(configuration), dtype=np.float32)

    @classmethod
    def fromStateDict(cls, dic: dict):
        return cls(dic['metadata']['gamma'], dic['metadata']['A'], dic['metadata']['B'], dic['xyt'])

    def getDiskMap(self, xytu: np.ndarray):  # TODO: rewrite it
        xy = xytu[:, :2]
        t = xytu[:, 2]
        a = -self.disks_per_rod * Voronoi.d / self.gamma / 2
        v = np.vstack([np.cos(t), np.sin(t)]).T
        pts = [xy + (a + j * Voronoi.d / self.gamma) * v for j in range(self.disks_per_rod)]
        return np.vstack(pts)

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
        output = ut.CArray(np.zeros(self.num_rods * 8, dtype=[('id2', np.int32), ('weight', np.float32)]))
        indices = ut.CArray(np.zeros((self.num_rods,), dtype=np.int32))
        n_edges = kernel_function(self.num_rods, self.disks_per_rod,
                                  self.disk_map.ptr, output.ptr, indices.ptr, self.A, self.B)
        return indices, output.data[:n_edges]

    def true_delaunay(self):
        from .orders import Delaunay
        # if isParticleTooClose(ut.CArrayF(self.configuration)):
        #     return None
        return Delaunay(False, *self.delaunay_template(ker.dll.trueDelaunay), self.gamma)

    def weighted_delaunay(self):
        from .orders import Delaunay
        # if isParticleTooClose(ut.CArrayF(self.configuration)):
        #     return None
        return Delaunay(True, *self.delaunay_template(ker.dll.weightedDelaunay), self.gamma)

    def delaunay(self, weighted: bool):
        if weighted:
            return self.weighted_delaunay()
        else:
            return self.true_delaunay()


class DelaunayBase:
    """
    Inherited by orders.Delaunay. This class provides interaction with cpp.
    weighted_edges: dtype=[('id2', np.int32), ('weight', np.float32)]
    """

    def __init__(self, weighted: bool, indices: ut.CArray, weighted_edges: np.ndarray, gamma: float):
        self.gamma = gamma
        self.weighted = weighted
        self.indices = indices
        self.num_rods = indices.data.shape[0]
        self.weight_sums = ut.CArrayFZeros((self.num_rods,))
        self.edges = ut.CArray(weighted_edges['id2'], dtype=np.int32)
        self.num_edges = self.edges.data.shape[0]
        self.weights = ut.CArray(weighted_edges['weight'], dtype=np.float32)
        ker.dll.sumOverWeights(self.num_edges, self.num_rods, self.indices.ptr,
                               self.edges.ptr, self.weights.ptr, self.weight_sums.ptr)

    @classmethod
    def legacy(cls, num_rods: int, gamma: float, xyt: ut.CArray):
        disks_per_rod = int(1 + 2 * (gamma - 1) / Voronoi.d)
        output = ut.CArray(np.zeros(num_rods * 10, dtype=[('id2', np.int32), ('weight', np.float32)]))
        indices = ut.CArray(np.zeros((num_rods,), dtype=np.int32))
        try:
            n_edges = ker.dll.legacyDelaunay(num_rods, disks_per_rod, gamma, xyt.ptr, output.ptr, indices.ptr)
        except OSError:
            return None
        obj = cls(False, indices, output.data, gamma)
        return obj

    @property
    def params(self):
        return self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr

    def iter_edges(self):
        i = 0
        for k in range(self.num_edges):
            j = self.edges.data[k]
            if k >= self.indices[i + 1]: i += 1
            yield i, j

    def z_number(self, arg=None) -> np.ndarray:
        if self.weighted:
            result = ut.CArrayFZeros((self.num_rods,))
            one_weights = ut.CArray(np.ones((self.num_edges,), dtype=np.float32))
            ker.dll.sumOverWeights(
                *self.params, one_weights.ptr, result.ptr
            )
            return result.data.copy()
        else:
            return self.weight_sums.data.copy()

    def phi_p(self, p: int, xyt: ut.CArray) -> np.ndarray[np.complex64]:
        """
        Assume that p is an even number. Because we use (-z)^4 = z^4, (-z)^6 = z^6
        """
        z_p = ut.CArray(np.zeros((self.num_edges,), dtype=np.complex64))
        Phi = ut.CArray(np.zeros((self.num_rods,), dtype=np.complex64))
        ker.dll.z_ij_power_p(*self.params, xyt.ptr, z_p.ptr, np.float32(p))
        z_p.data *= self.weights.data
        ker.dll.sumComplex(*self.params, z_p.ptr, Phi.ptr)
        return Phi.data / self.weight_sums.data

    def phi_p_ellipse_template(self, function_providing_angles):
        """
        :param function_providing_angles: (xyt) -> np.ndarray: determines the method to calculate phi angle.
        """

        def inner(p: int, gamma: float, xyt: ut.CArray) -> np.ndarray[np.complex64]:
            angle = ut.CArray(function_providing_angles(xyt))
            z_p = ut.CArray(np.zeros((self.num_edges * 2,), dtype=np.complex64))
            Phi = ut.CArray(np.zeros((self.num_rods,), dtype=np.complex64))
            ker.dll.anisotropic_z_ij_power_p(*self.params, xyt.ptr, angle.ptr, z_p.ptr,
                                             np.float32(gamma), np.float32(p))
            z_p.data *= np.repeat(self.weights.data, 2)
            ker.dll.sumAnisotropicComplex(*self.params, z_p.ptr, Phi.ptr)
            return Phi.data / self.weight_sums.data

        return inner

    def pure_rotation_phi(self, xyt: ut.CArray) -> np.ndarray:
        phi_angle = ut.CArrayFZeros((self.num_rods,))
        ker.dll.pure_rotation_direction_phi(*self.params, xyt.ptr, phi_angle.ptr)
        return phi_angle.data % np.pi

    def S_center(self, xyt: ut.CArray) -> np.ndarray:
        """
        Use the orientation of the centering particle, θ_i, as director.
        """
        ti_tj = ut.CArrayFZeros((self.num_edges,))
        ker.dll.orientation_diff_ij(*self.params, xyt.ptr, ti_tj.ptr)
        c = ut.CArray(np.cos(2 * ti_tj.data))
        S = ut.CArrayFZeros((self.num_rods,))
        ker.dll.sumOverWeights(*self.params, c.ptr, S.ptr)
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
        ker.dll.sumOverNeighbors(*self.params, ux.ptr, sum_ux.ptr)
        ker.dll.sumOverNeighbors(*self.params, uy.ptr, sum_uy.ptr)
        if self.weighted:
            # TODO: bug
            sum_ux.data *= self.weights.data
            sum_uy.data *= self.weights.data
        return sum_ux, sum_uy

    def mean_rij(self, xyt: ut.CArray) -> np.float32:
        return ker.dll.mean_r_ij(*self.params, xyt.ptr)

    def segment_dist_moment(self, xyt: ut.CArray, moment: int) -> np.float32:
        return ker.dll.segment_dist_moment(*self.params, xyt.ptr, self.gamma, moment)
