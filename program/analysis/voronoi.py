import numpy as np
import scipy.spatial as sp

from . import utils as ut
from .kernel import ker


def DelaunayModulo(delaunay: sp.Delaunay, N: int) -> (ut.CArray, ut.CArray, ut.CArray):
    indices_in = ut.CArray(delaunay.vertex_neighbor_vertices[0])
    edges_in = ut.CArray(delaunay.vertex_neighbor_vertices[1])
    n = indices_in.data.shape[0]
    m = edges_in.data.shape[0]
    mask = DelaunayClip(delaunay, indices_in, edges_in)
    indices_out = ut.CArray(np.zeros((N,), np.int32))
    edges_out = ut.CArray(np.zeros((m // 2,), np.int32))
    weights_out = ut.CArray(np.zeros((m // 2,), np.int32))
    n_edges = ker.dll.DelaunayModulo(n, m, N, indices_in.ptr, edges_in.ptr, mask.ptr,
                                     indices_out.ptr, edges_out.ptr, weights_out.ptr)
    # clip edges data
    edges_out = ut.CArray(edges_out.data[:n_edges])
    weights_out = ut.CArray(weights_out.data[:n_edges])
    return indices_out, edges_out, weights_out


def DelaunayClip(delaunay: sp.Delaunay, indices_in, edges_in) -> ut.CArray:
    cos_threshold = -0.9
    u_tri = delaunay.simplices
    u_tri = np.sort(u_tri, axis=1)
    tri = u_tri[np.lexsort((u_tri[:, 2], u_tri[:, 1], u_tri[:, 0]))]
    tri = ut.CArray(np.concatenate([tri.astype(np.int32), np.ones((tri.shape[0], 1), dtype=np.int32)], axis=1))
    mask = ut.CArray(np.ones((delaunay.vertex_neighbor_vertices[1].shape[0],), dtype=np.int32))
    points = ut.CArray(delaunay.points, dtype=np.float32)
    convex_hull = ut.CArray(np.sort(delaunay.convex_hull, axis=1))
    ker.dll.RemoveBadBoundaryEdges(points.ptr, convex_hull.ptr, tri.ptr, indices_in.ptr, edges_in.ptr, mask.ptr,
                                   convex_hull.data.shape[0], cos_threshold)
    return mask

class Voronoi:
    d = 0.05

    def __init__(self, gamma: float, A: float, B: float, configuration: np.ndarray):
        self.gamma = gamma
        self.A, self.B = A, B
        self.configuration = configuration
        self.num_rods = configuration.shape[0]
        self.disks_per_rod = int(1 + 2 * (gamma - 1) / Voronoi.d)
        self.disk_map = self.getDiskMap(configuration)

    @classmethod
    def fromStateDict(cls, dic: dict):
        return cls(dic['metadata']['gamma'], dic['metadata']['A'], dic['metadata']['B'], dic['xyt'])

    def getDiskMap(self, xytu: np.ndarray) -> np.ndarray:
        xy = xytu[:, :2]
        t = xytu[:, 2]
        a = -self.disks_per_rod * Voronoi.d / self.gamma / 2
        v = np.vstack([np.cos(t), np.sin(t)]).T
        pts = [xy + (a + j * Voronoi.d / self.gamma) * v for j in range(self.disks_per_rod)]
        return np.vstack(pts)

    def delaunay(self) -> 'Delaunay':
        """
        :param kernel_function: ker.dll.trueDelaunay | ker.dll.weightedDelaunay
        """
        from .orders import Delaunay
        delaunay = sp.Delaunay(self.disk_map)
        indices, edges, weights = DelaunayModulo(delaunay, self.num_rods)
        return Delaunay(indices, edges, weights, self.gamma)


class DelaunayBase:
    """
    Inherited by orders.Delaunay. This class provides interaction with cpp.
    weighted_edges: dtype=[('id2', np.int32), ('weight', np.float32)]
    """

    def __init__(self, indices: ut.CArray, edges: ut.CArray, weights: ut.CArray, gamma: float):
        self.gamma = gamma
        self.indices = indices
        self.edges = edges
        self.weights = weights
        self.num_rods = indices.data.shape[0]
        self.num_edges = self.edges.data.shape[0]

    @property
    def params(self):
        return self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr

    @property
    def edge_types(self) -> np.ndarray[np.int32]:
        """
        :return: array, 0 for head-to-head, 1 for head-to-side, 2 for side-to-side
        """
        res = np.full((self.num_edges,), 1, dtype=np.int32)
        max_weight = np.max(self.weights.data)
        # res[self.weights.data == 1] = 0
        # res[self.weights.data == max_weight] = 2
        res[self.weights.data <= 60] = 0
        res[self.weights.data >= 120] = 2
        return res

    def iter_edges(self):
        ty = self.edge_types
        i = 0
        for k in range(self.num_edges):
            j = self.edges.data[k]
            while k >= self.indices[i] and i < self.num_rods:
                i += 1
            yield i - 1, j, ty[k]

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
