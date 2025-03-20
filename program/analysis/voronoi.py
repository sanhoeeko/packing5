import numpy as np
import scipy.spatial as sp

from . import utils as ut
from .kernel import ker
from numba import njit


def DelaunayModulo(delaunay: sp.Delaunay, N: int) -> (ut.CArray, ut.CArray, ut.CArray):
    indices_in = ut.CArray(delaunay.vertex_neighbor_vertices[0])
    edges_in = ut.CArray(delaunay.vertex_neighbor_vertices[1])
    mask = ut.CArray(DelaunayClip(delaunay))
    n = indices_in.data.shape[0]
    m = edges_in.data.shape[0]
    indices_out = ut.CArray(np.zeros((N,), np.int32))
    edges_out = ut.CArray(np.zeros((m // 2,), np.int32))
    weights_out = ut.CArray(np.zeros((m // 2,), np.int32))
    n_edges = ker.dll.DelaunayModulo(n, m, N, indices_in.ptr, edges_in.ptr, mask.ptr,
                                     indices_out.ptr, edges_out.ptr, weights_out.ptr)
    # clip edges data
    edges_out = ut.CArray(edges_out.data[:n_edges])
    weights_out = ut.CArray(weights_out.data[:n_edges])
    return indices_out, edges_out, weights_out


# 使用 njit 加速的核心函数
@njit
def _DelaunayClip_core(indptr, indices, convex_hull):
    # 初始化 mask，初始值为 1
    mask = np.ones(len(indices), dtype=np.int32)

    # 将凸包边转化为稠密的标记矩阵
    max_index = np.max(indices)
    convex_hull_mask = np.zeros((max_index + 1, max_index + 1), dtype=np.uint8)
    for edge in convex_hull:
        convex_hull_mask[edge[0], edge[1]] = 1
        convex_hull_mask[edge[1], edge[0]] = 1  # 双向标记

    # 遍历每个顶点的邻接点，检查是否为凸包边
    for i in range(len(indptr) - 1):
        neighbors = indices[indptr[i]:indptr[i + 1]]
        for neighbor in neighbors:
            if convex_hull_mask[i, neighbor] == 1:
                for k in range(indptr[i], indptr[i + 1]):  # 手动匹配位置
                    if indices[k] == neighbor:
                        mask[k] = 0
                        break

    return mask


# 封装为用户友好的接口
def DelaunayClip(delaunay):
    """
    输入 Delaunay 剖分对象，返回一个 mask 数组，用于标记
    vertex_neighbor_vertices[1] 中是否包含凸包边。

    参数:
        delaunay: scipy.spatial.Delaunay 对象

    返回:
        mask: np.ndarray, int32 类型，0 表示对应边是凸包边，1 表示不是。
    """
    # 提取 vertex_neighbor_vertices 和凸包数据
    indptr, indices = delaunay.vertex_neighbor_vertices
    convex_hull = np.array([tuple(sorted(edge)) for edge in delaunay.convex_hull])

    # 调用加速的核心函数
    return _DelaunayClip_core(indptr, indices, convex_hull)


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
