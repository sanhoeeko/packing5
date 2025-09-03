# import tracemalloc

# tracemalloc.start()
import numpy as np

import default
from . import utils as ut
from .voronoi import Voronoi, DelaunayBase


class StaticOrders:
    @staticmethod
    def S_global(xyt: np.ndarray):
        # calculate the eigenvector of total Q-tensor
        t = xyt[:, 2] % np.pi
        c = np.mean(np.cos(2 * t))
        s = np.mean(np.sin(2 * t))
        S_g = np.sqrt(c ** 2 + s ** 2)
        director_angle = np.arctan2(s, c + S_g)
        # calculate 2*cos(θ-α)^2-1
        return np.cos(2 * (t - director_angle))

    @staticmethod
    def S_x(xyt: np.ndarray):
        t = xyt[:, 2] % np.pi
        return np.cos(2 * t)

    @staticmethod
    def Angle(xyt: np.ndarray):
        return xyt[:, 2] % np.pi

    @staticmethod
    def AngleDist(xyt: np.ndarray):
        n_angles = 180
        t = xyt[:, 2] % np.pi
        hist, bins = np.histogram(t, bins=n_angles, range=(0, np.pi))
        return hist  # bins are easy to calculate

    @staticmethod
    def YRank(xyt: np.ndarray, abg: tuple):
        A, B, gamma = abg
        N = xyt.shape[0]
        phi = ut.phi(N, gamma, A, B)
        return ut.y_rank(N, phi, xyt, B)


def general_order_parameter(name: str, xyt: np.ndarray, delaunay: 'Delaunay' = None, abg: tuple = None) -> np.ndarray:
    """
    :return: a numpy array of shape (N,), N = particle number.
    parameter `name` and `xyt` are necessary.
    """
    if name in ['S_global', 'S_x', 'Angle', 'AngleDist']:
        return getattr(StaticOrders, name)(xyt)
    elif name in ['YRank']:
        return getattr(StaticOrders, name)(xyt, abg)
    elif name.startswith('Elliptic'):
        return getattr(delaunay, name)(ut.CArray(xyt), abg[2])
    elif name.startswith('director-'):
        order = int(name.split('-')[1])
        return delaunay.n_order_director(order)(ut.CArray(xyt))
    else:
        return getattr(delaunay, name)(ut.CArray(xyt))


def OrderParameterList(order_parameter_names: list[str]):
    """
    Common order parameter interface. For both voronoi and non-voronoi order parameters.
    Inner function returns a structured array.
    """
    dtype = list(zip(order_parameter_names, ['f4'] * len(order_parameter_names)))

    def inner(xyt, abg) -> np.ndarray:
        xyt_c = ut.CArrayF(xyt)
        n = xyt.shape[-2]
        voro = Voronoi(abg[2], abg[0], abg[1], xyt_c.data).delaunay()
        result = np.full((n,), np.nan, dtype=dtype)
        if voro is not None:
            for name in order_parameter_names:
                result[name] = general_order_parameter(name, xyt, voro, abg)
        return result

    return inner


class Delaunay(DelaunayBase):
    """
    All order parameters that requires Delaunay triangulation are here.
    """

    def __init__(self, indices: ut.CArray, edges: ut.CArray, weights: ut.CArray, gamma: float,
                 A: float, B: float, disks_per_rod: int):
        super().__init__(indices, edges, weights, gamma, A, B, disks_per_rod)

    def C6(self, xyt: ut.CArray) -> np.ndarray:
        """
        :return: 1 - [number of defects] / [number of internal particles]
        """
        z = self.z_number()
        body = ~self.dist_hull(xyt).astype(bool)
        body_rods = np.sum(body)
        d = np.bitwise_and(body, z == 6)
        return d / (body_rods / self.num_rods)

    def C6_raw(self, xyt: ut.CArray) -> np.ndarray:
        """
        This is used for combining multiple masks for C6.
        It MUST be used with a mask: `body`
        """
        return self.z_number() == 6

    def defect_number(self, xyt: ut.CArray) -> int:
        z = self.z_number()
        body = ~self.dist_hull(xyt).astype(bool)
        return np.sum(np.bitwise_and(body, z != 6))

    def Phi6Complex(self, xyt: ut.CArray) -> np.ndarray[np.complex64]:
        return self.phi_p(6, xyt)

    def Phi4Complex(self, xyt: ut.CArray) -> np.ndarray[np.complex64]:
        return self.phi_p(4, xyt)

    def Phi6(self, xyt: ut.CArray) -> np.ndarray:
        return np.abs(self.phi_p(6, xyt))

    def Phi4(self, xyt: ut.CArray) -> np.ndarray:
        return np.abs(self.phi_p(4, xyt))

    def EllipticPhi6(self, xyt: ut.CArray, gamma: float) -> np.ndarray:
        # return np.abs(self.phi_p_ellipse_template(self.pure_rotation_phi)(6, gamma, xyt))
        return np.abs(self.phi_p_ellipse_template(self.DirectorAngle)(6, gamma, xyt))
        # return np.abs(self.phi_p_ellipse_template(StaticOrders.Angle)(6, gamma, xyt))

    def PureRotationAngle(self, xyt: ut.CArray) -> np.ndarray:
        return super().pure_rotation_phi(xyt)

    def S_center(self, xyt: ut.CArray) -> np.ndarray:
        return super().S_center(xyt)

    def S_local(self, xyt: ut.CArray) -> np.ndarray:
        """
        Use the director as the eigenvector of Q-tensor, then S_local is the eigenvalue.
        """
        sum_ux, sum_uy = self.Q_tensor(xyt.data[:, 2])
        S = np.sqrt(sum_ux.data ** 2 + sum_uy.data ** 2)
        return S.data / (self.z_number() + 1)

    def director_raw(self, angles: np.ndarray) -> np.ndarray:
        """
        Calculate the director as the eigenvector of Q-tensor.
        """
        sum_ux, sum_uy = self.Q_tensor(angles)
        S = np.sqrt(sum_ux.data ** 2 + sum_uy.data ** 2)
        return np.arctan2(sum_uy.data, sum_ux.data + S)

    def DirectorAngle(self, xyt: ut.CArray) -> np.ndarray:
        """
        Order parameter corresponding to the eigenvector of Q-tensor.
        """
        return self.director_raw(xyt.data[:, 2]) % np.pi

    def CrystalNematicAngle(self, xyt: ut.CArray) -> np.ndarray:
        phi = self.PureRotationAngle(xyt)
        theta = self.DirectorAngle(xyt)
        return np.cos(2 * (theta - phi))

    def n_order_director(self, order: int):
        """
        Call this by "director-x", where x is an integer.
        """
        if order == 0:
            def inner(xyt: ut.CArray) -> np.ndarray:
                return xyt.data[:, 2]
        elif order == 1:
            def inner(xyt: ut.CArray) -> np.ndarray:
                return self.DirectorAngle(xyt)
        else:
            def inner(xyt: ut.CArray) -> np.ndarray:
                n = self.DirectorAngle(xyt)
                for i in range(1, order):
                    n = self.director_raw(n)
                return n % np.pi

        return inner

    def GlobalMeanSegmentDist(self, xyt: ut.CArray) -> float:
        """
        :return: mean segment distance normalized in [0, 2]
        """
        return super().segment_dist_moment(xyt, 1) * self.gamma

    def StdSegmentDist(self, xyt: ut.CArray) -> float:
        """
        :return: standard deviation of segment distance, normalized in [0, 2]
        """
        m1 = super().segment_dist_moment(xyt, 1)
        m2 = super().segment_dist_moment(xyt, 2)
        std = np.sqrt(m2 - m1 * m1)
        return std * self.gamma

    def FittedEllipticPhi6(self, xyt: ut.CArray) -> np.ndarray:
        dic = self.phi_p_ellipse_fitted(6, xyt)
        print(dic['gammas'])
        return np.abs(dic['Phi'])

    def isolatedDefect(self, xyt: ut.CArray) -> np.ndarray:
        """
        :return: N(isolated defects) / N(internal)
        """
        z = self.z_number()
        body = 1 - self.dist_hull(xyt)
        defect = np.bitwise_and(body.astype(bool), z != 6)
        isolated_defect = np.bitwise_and(self.is_isolated_defect().astype(bool), defect)
        return isolated_defect / np.sum(body) * self.num_rods

    def isolatedDefectRatio(self, xyt: ut.CArray) -> np.ndarray:
        """
        :return: N(isolated defects) / N(defects)
        """
        z = self.z_number()
        body = 1 - self.dist_hull(xyt)
        defect = np.bitwise_and(body.astype(bool), z != 6)
        isolated_defect = np.bitwise_and(self.is_isolated_defect().astype(bool), defect)
        return isolated_defect / np.sum(defect) * self.num_rods

    def winding2(self, xyt: ut.CArray) -> np.ndarray[np.int32]:
        angles = ut.CArray(self.n_order_director(2)(xyt))
        wd2 = super().winding_angle(xyt, angles) / np.pi
        return np.round(wd2).astype(np.int32)

    def FarthestSegmentDist(self, xyt: ut.CArray) -> np.ndarray:
        return self.max_segment_dist(xyt) * self.gamma

    def MeanSegmentDist(self, xyt: ut.CArray) -> np.ndarray:
        return self.mean_segment_dist(xyt) * self.gamma

    def raw_dense(self, xyt: ut.CArray) -> np.ndarray[np.int32]:
        dist = self.FarthestSegmentDist(xyt)
        res = np.zeros((self.num_rods,), dtype=np.int32)
        super_dense = dist < default.segdist_for_dense
        dense = np.bitwise_and(dist <= default.segdist_for_sparse, dist >= default.segdist_for_dense)
        res[super_dense] = 2
        res[dense] = 1
        return res

    def raw_dense_1(self, xyt: ut.CArray) -> np.ndarray[np.int32]:
        dist = self.FarthestSegmentDist(xyt)
        res = np.zeros((self.num_rods,), dtype=np.int32)
        res[dist <= default.segdist_for_sparse] = 1
        return res

    def dense(self, xyt: ut.CArray) -> np.ndarray[np.int32]:
        return self.VoteN(1, self.raw_dense(xyt))

    def dense_1(self, xyt: ut.CArray) -> np.ndarray[np.int32]:
        return self.VoteN(3, self.raw_dense_1(xyt))

    def SegmentDistRankMask(self, xyt: ut.CArray) -> np.ndarray[np.int32]:
        phi = ut.phi(self.num_rods, self.gamma, self.A, self.B)
        ratio = ut.r_phi(phi) * 1.0

        if ratio == 1:
            return np.ones((self.num_rods,), dtype=np.int32)

        dist = self.max_segment_dist(xyt)
        sorted_dist = np.sort(dist)
        idx = min(int(round(self.num_rods * ratio)), self.num_rods - 1)
        critical_value = sorted_dist[idx]
        mask = (dist <= critical_value).astype(np.int32)
        return self.VoteN(4, mask)
        # return mask
