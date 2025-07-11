import numpy as np
import scipy.spatial as sp
from scipy.special import ellipe as EllipticE

from . import utils as ut
from .bitmatrix import BitMatrix, detectEvents
from .kernel import ker


def DelaunayModulo(delaunay: sp.Delaunay, N: int, disks_per_rod: int) -> (ut.CArray, ut.CArray, ut.CArray):
    indices_in = ut.CArray(delaunay.vertex_neighbor_vertices[0][1:])
    edges_in = ut.CArray(delaunay.vertex_neighbor_vertices[1])
    n = N * disks_per_rod
    m = edges_in.data.shape[0]
    indices_out = ut.CArray(np.zeros((N,), np.int32))
    edges_out = ut.CArray(np.zeros((m // 2,), np.int32))
    weights_out = ut.CArray(np.zeros((m // 2,), np.int32))
    n_edges = ker.dll.DelaunayModulo(n, m, N, indices_in.ptr, edges_in.ptr, 0,
                                     indices_out.ptr, edges_out.ptr, weights_out.ptr)
    # clip edges data
    edges_out = ut.CArray(edges_out.data[:n_edges])
    weights_out = ut.CArray(weights_out.data[:n_edges])
    return indices_out, edges_out, weights_out


def EllipsePoints(a: float, b: float, d: float) -> np.ndarray:
    """
    :return: (2, n) matrix: [[x], [y]]
    """

    def X(t):
        return np.array([a * np.cos(t), b * np.sin(t)])

    def NextT(s, t):
        if np.abs(t - np.round(t / (np.pi / 2)) * (np.pi / 2)) < 1e-6: t += 1e-6
        r = np.sqrt(1 - e ** 2 * np.cos(t) ** 2)
        sqr = np.sqrt(a * r * (a * r ** 3 - 2 * (-1 + r ** 2) * s * np.tan(t)))
        return (a * r ** 2 - sqr) / (a * np.tan(t) * (-1 + r ** 2))

    e = np.sqrt(1 - b ** 2 / a ** 2)
    C = 4 * a * EllipticE(e ** 2)  # Note python's definition of elliptic integrals!
    n = int(np.round(C / d))
    s = C / n
    ts = np.zeros((n,))
    for i in range(1, n):
        ts[i] = ts[i - 1] + NextT(s, ts[i - 1])
    pts = X(ts)  # (2, n) matrix
    # Evaluation of error (typically < 0.002)
    # dx = np.diff(pts[0], prepend=pts[0, -1])
    # dy = np.diff(pts[1], prepend=pts[1, -1])
    # dr = np.sqrt(dx ** 2 + dy ** 2)
    # error = (np.max(dr) - np.min(dr)) / s
    # print(error)
    return pts


def CirclePoints(R: float, d: float) -> np.ndarray:
    def X(t):
        return np.array([R * np.cos(t), R * np.sin(t)])

    C = 2 * np.pi * R
    n = int(np.round(C / d))
    d_theta = 2 * np.pi / n
    ts = np.arange(0, 2 * np.pi, d_theta)
    return X(ts)


def DelaunayModuloClip(delaunay: sp.Delaunay, N: int, disks_per_rod: int) -> (ut.CArray, ut.CArray, ut.CArray):
    indices_in = ut.CArray(delaunay.vertex_neighbor_vertices[0])
    edges_in = ut.CArray(delaunay.vertex_neighbor_vertices[1])
    n = N * disks_per_rod
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
    d = 0.1

    def __init__(self, gamma: float, A: float, B: float, configuration: np.ndarray):
        self.gamma = gamma
        self.A, self.B = A, B
        self.configuration = configuration
        self.num_rods = configuration.shape[0]
        self.disks_per_rod = int(1 + 2 * (gamma - 1) / Voronoi.d)
        self.disk_map = np.vstack([self.getDiskMap(configuration), self.getEllipsePoints()])

    @classmethod
    def fromStateDict(cls, dic: dict):
        return cls(dic['metadata']['gamma'], dic['metadata']['A'], dic['metadata']['B'], dic['xyt'])

    def getDiskMap(self, xytu: np.ndarray) -> np.ndarray:
        xy = xytu[:, :2]
        t = xytu[:, 2]
        a = -(self.disks_per_rod - 1) * Voronoi.d / self.gamma / 2
        v = np.vstack([np.cos(t), np.sin(t)]).T
        pts = [xy + (a + j * Voronoi.d / self.gamma) * v for j in range(self.disks_per_rod)]
        return np.vstack(pts)

    def getEllipsePoints(self):
        if self.A == self.B:
            return CirclePoints(self.A + 1 / self.gamma, Voronoi.d).T
        else:
            return EllipsePoints(self.A + 1 / self.gamma, self.B + 1 / self.gamma, Voronoi.d).T

    def delaunay(self):
        """
        :param kernel_function: ker.dll.trueDelaunay | ker.dll.weightedDelaunay
        """
        from .orders import Delaunay
        delaunay = sp.Delaunay(self.disk_map)
        indices, edges, weights = DelaunayModulo(delaunay, self.num_rods, self.disks_per_rod)
        return Delaunay(indices, edges, weights, self.gamma, self.A, self.B, self.disks_per_rod)


class DelaunayBase:
    """
    Inherited by orders.Delaunay. This class provides interaction with cpp.
    weighted_edges: dtype=[('id2', np.int32), ('weight', np.float32)]
    """

    def __init__(self, indices: ut.CArray, edges: ut.CArray, weights: ut.CArray, gamma: float,
                 A: float, B: float, disks_per_rod: int):
        self.gamma = gamma
        self.A, self.B = A, B
        self.indices = indices
        self.edges = edges
        self.weights = weights
        self.num_rods = indices.data.shape[0]
        self.num_edges = self.edges.data.shape[0]
        self.disks_per_rod = disks_per_rod

    @property
    def params(self):
        return self.num_edges, self.num_rods, self.indices.ptr, self.edges.ptr

    def check(self):
        # to fix some bugs in multiprocess (Maybe in which `CArray`s are incorrectly copied)
        self.indices.check()
        self.edges.check()
        return self

    def adjacency_matrix(self) -> BitMatrix:
        bm = BitMatrix(self.num_rods)
        ker.dll.bitmap_from_delaunay(*self.params, bm.arr.ptr)
        return bm

    def difference(self, o: 'DelaunayBase'):
        """
        :return: (n_edges, 2) matrix.
        Each pair of integers represents a new edge which does not present in [o: 'DelaunayBase'].
        Examples:
            V[t+1].difference(V[t]) -> created edges
            V[t].difference(V[t+1]) -> destroyed edges
        """
        return self.adjacency_matrix() - o.adjacency_matrix()

    def events_compared_with(self, o: 'DelaunayBase', xyt_self: ut.CArray, xyt_o: ut.CArray) -> ut.CArray:
        z1_c = self.z_number_c()
        z1_c.data[self.dist_hull(xyt_self) == 1] = 6
        z0_c = o.z_number_c()
        z0_c.data[self.dist_hull(xyt_o) == 1] = 6
        b1 = self.adjacency_matrix()
        b0 = o.adjacency_matrix()
        return detectEvents(b0, b1, z0_c, z1_c)

    @property
    def edge_types(self) -> np.ndarray[np.int32]:
        """
        :return: array, 0 for head-to-head, 1 for head-to-side, 2 for side-to-side
        """
        res = np.full((self.num_edges,), 1, dtype=np.int32)
        res[self.weights.data < 0.5 * self.disks_per_rod] = 0
        res[self.weights.data >= 1.5 * self.disks_per_rod] = 2
        return res

    def iter_edges(self):
        ty = self.edge_types
        i = 0
        for k in range(self.num_edges):
            j = self.edges.data[k]
            while k >= self.indices[i] and i < self.num_rods:
                i += 1
            yield i, j, ty[k]

    def z_number_c(self, args=None) -> ut.CArray:
        z = ut.CArray(np.zeros((self.num_rods,), dtype=np.int32))
        ker.dll.neighbors(*self.params, z.ptr)
        return z

    def z_number(self, args=None) -> np.ndarray[np.int32]:
        return self.z_number_c(args).data

    def raw_total_topological_charge(self, args=None) -> int:
        return 2 * self.num_edges - 6 * self.num_rods

    def total_topological_charge(self, xyt: ut.CArray) -> int:
        """
        `2 * self.num_edges - 6 * self.num_rods` will result in a large negative number because boundary effect.
        So this function only count total topological charge of internal particles.
        """
        internal = 1 - self.dist_hull(xyt)
        return np.dot(self.z_number(), internal) - 6 * np.sum(internal)

    def dist_hull(self, xyt: ut.CArray) -> np.ndarray[np.int32]:
        """
        :return: binary array: 0 for internal, 1 for marginal
        """
        min_dist = 2
        return (self.dist_to_ellipse(xyt) < min_dist).astype(np.int32)

    def dist_to_ellipse(self, xyt: ut.CArray) -> np.ndarray:
        dists = ut.CArrayFZeros((self.num_rods,))
        xy = ut.CArray(xyt.data[:, 0:2], dtype=np.float32)
        ker.dll.DistToEllipse(self.A, self.B, xy.ptr, dists.ptr, self.num_rods)
        return dists.data

    def phi_p(self, p: int, xyt: ut.CArray) -> np.ndarray[np.complex64]:
        """
        Assume that p is an even number. Because we use (-z)^4 = z^4, (-z)^6 = z^6, where z=exp(i*theta)
        For an edge (i,j), z(j,i)=exp(i*(theta+pi))=-z(i,j)
        """
        z_p = ut.CArray(np.zeros((self.num_edges,), dtype=np.complex64))
        Phi = ut.CArray(np.zeros((self.num_rods,), dtype=np.complex64))
        ker.dll.z_ij_power_p(*self.params, xyt.ptr, z_p.ptr, np.float32(p))
        ker.dll.complexSum(*self.params, z_p.ptr, Phi.ptr)
        return Phi.data / self.z_number()

    def phi_p_ellipse_template(self, function_providing_angles):
        """
        :param function_providing_angles: (xyt) -> np.ndarray: determines the method to calculate phi angle.
        """

        def inner(p: int, gamma: float, xyt: ut.CArray) -> np.ndarray[np.complex64]:
            angle = ut.CArray(function_providing_angles(xyt))
            z_p = ut.CArray(np.zeros((self.num_edges * 2,), dtype=np.complex64))
            Phi = ut.CArray(np.zeros((self.num_rods,), dtype=np.complex64))
            ker.dll.anisotropic_z_ij_power_p(*self.params, xyt.ptr, angle.ptr, z_p.ptr, gamma, p)
            ker.dll.sumAnisotropicComplex(*self.params, z_p.ptr, Phi.ptr)
            return Phi.data / self.z_number()

        return inner

    def phi_p_ellipse_fitted(self, p: int, xyt: ut.CArray) -> dict:
        z_p = ut.CArray(np.zeros((self.num_edges * 2,), dtype=np.complex64))
        Phi = ut.CArray(np.zeros((self.num_rods,), dtype=np.complex64))
        gammas = ut.CArrayFZeros((self.num_rods,))
        thetas = ut.CArrayFZeros((self.num_rods,))
        ker.dll.FittedEllipticPhi_p(*self.params, xyt.ptr, z_p.ptr, gammas.ptr, thetas.ptr, p)
        ker.dll.sumAnisotropicComplex(*self.params, z_p.ptr, Phi.ptr)
        phi = Phi.data.copy()
        phi[np.bitwise_or(gammas.data < 0, gammas.data > self.gamma)] = 0
        return {
            'Phi': phi / self.z_number(),
            'gammas': gammas.data,
            'thetas': thetas.data,
        }

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
        ker.dll.symmetricSum(*self.params, c.ptr, S.ptr)
        return S.data / self.z_number()

    def Q_tensor(self, angles: np.ndarray) -> (ut.CArray, ut.CArray):
        """
        Q = ∑ 2(u @ u.T - 1) = [ ∑ cos 2t, ∑ sin 2t; ∑ sin 2t, -∑ cos 2t]
        The sum (can be weighted) is taken over neighbors.
        Here we define:
        sum_ux[i] = ∑[j] cos 2t[i,j]
        sum_uy[i] = ∑[j] sin 2t[i,j]
        """
        t_mul_2 = 2 * angles
        ux = ut.CArray(np.cos(t_mul_2))
        uy = ut.CArray(np.sin(t_mul_2))
        sum_ux = ux.copy()  # Add the centering rod itself
        sum_uy = uy.copy()
        ker.dll.symmetricSum(*self.params, ux.ptr, sum_ux.ptr)  # Add neighbors
        ker.dll.symmetricSum(*self.params, uy.ptr, sum_uy.ptr)
        return sum_ux, sum_uy

    def mean_rij(self, xyt: ut.CArray) -> np.float32:
        return ker.dll.mean_r_ij(*self.params, xyt.ptr)

    def segment_dist_moment(self, xyt: ut.CArray, moment: int) -> np.float32:
        return ker.dll.segment_dist_moment(*self.params, xyt.ptr, self.gamma, moment)

    def is_isolated_defect(self) -> np.ndarray:
        z = ut.CArray(np.zeros((self.num_rods,), dtype=np.int32))
        ker.dll.is_isolated_defect(*self.params, z.ptr)
        return z.data

    def winding_angle(self, xyt: ut.CArray, angles: ut.CArray) -> np.ndarray:
        theta = ut.CArrayFZeros((self.num_rods,))
        ker.dll.windingAngle(*self.params, xyt.ptr, angles.ptr, theta.ptr)
        return theta.data

    def winding_angle_next_nearest(self, xyt: ut.CArray, angles: ut.CArray) -> np.ndarray:
        theta = ut.CArrayFZeros((self.num_rods,))
        ker.dll.windingAngleNextNearest(*self.params, xyt.ptr, angles.ptr, theta.ptr)
        return theta.data

    def farthest_segment_dist(self, xyt: ut.CArray) -> np.ndarray:
        seg_dist = ut.CArrayFZeros((self.num_edges,))
        max_dist = ut.CArrayFZeros((self.num_rods,))
        ker.dll.SegmentDistForBonds(*self.params, xyt.ptr, seg_dist.ptr, self.gamma)
        ker.dll.symmetricMax(*self.params, seg_dist.ptr, max_dist.ptr)
        return max_dist.data

    def vote(self, raw_res: ut.CArray) -> ut.CArray:
        res = ut.CArray(np.zeros((self.num_rods,), dtype=np.int32))
        ker.dll.vote(*self.params, raw_res.ptr, res.ptr, 2)
        return res

    def VoteN(self, n: int, x: np.ndarray[np.int32]) -> np.ndarray:
        y = ut.CArray(x)
        for i in range(n):
            y = self.vote(y)
        return y.data

    def segment_dist_rank_mask(self, xyt: ut.CArray, ratio: float) -> np.ndarray:
        if ratio == 1:
            return np.ones((self.num_rods,), dtype=bool)
        dist = self.farthest_segment_dist(xyt)
        sorted_dist = np.sort(dist)
        idx = min(int(round(self.num_rods * ratio)), self.num_rods - 1)
        critical_value = sorted_dist[idx]
        return dist <= critical_value
