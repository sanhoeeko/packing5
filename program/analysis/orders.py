import tracemalloc

tracemalloc.start()
import numpy as np

from . import utils as ut
from .voronoi import Voronoi, DelaunayBase


def S_global(xyt: np.ndarray):
    # calculate the eigenvector of total Q-tensor
    t = xyt[:, 2]
    c = np.mean(np.cos(2 * t))
    s = np.mean(np.sin(2 * t))
    S_g = np.sqrt(c ** 2 + s ** 2)
    director_angle = np.arctan2(s, c + S_g)
    # calculate 2*cos(θ-α)^2-1
    return np.cos(2 * (t - director_angle))


def Angle(xyt: np.ndarray):
    return xyt[:, 2] % np.pi


def general_order_parameter(name: str, xyt: np.ndarray, voro: Voronoi = None, abg: tuple = None):
    """
    :return: a numpy array of shape (N,), N = particle number.
    parameter `name` and `xyt` are necessary.
    """
    if name == 'S_global':
        return S_global(xyt)
    elif name == 'Angle':
        return Angle(xyt)
    elif name.startswith('Elliptic'):
        return getattr(voro, name)(ut.CArray(xyt), abg[2])
    else:
        return getattr(voro, name)(ut.CArray(xyt))


def OrderParameterList(order_parameter_names: list[str]):
    """
    Common order parameter interface. For both voronoi and non-voronoi order parameters.
    Inner function returns a structured array.
    """
    dtype = list(zip(order_parameter_names, ['f4'] * len(order_parameter_names)))

    def inner(xyt, abg, weighted) -> np.ndarray:
        xyt_c = ut.CArray(xyt)
        n = xyt.shape[-2]
        voro = Voronoi(abg[2], abg[0], abg[1], xyt_c.data).delaunay(weighted)
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

    def __init__(self, weighted: bool, indices: ut.CArray, weighted_edges: np.ndarray):
        super().__init__(weighted, indices, weighted_edges)

    def z_number(self, arg=None) -> np.ndarray:
        return super().z_number(arg)

    def Phi6Complex(self, xyt: ut.CArray) -> np.ndarray[np.complex64]:
        return self.phi_p(6, xyt)

    def Phi4Complex(self, xyt: ut.CArray) -> np.ndarray[np.complex64]:
        return self.phi_p(4, xyt)

    def Phi6(self, xyt: ut.CArray) -> np.ndarray:
        return np.abs(self.phi_p(6, xyt))

    def Phi4(self, xyt: ut.CArray) -> np.ndarray:
        return np.abs(self.phi_p(4, xyt))

    def EllipticPhi6(self, xyt: ut.CArray, gamma: float) -> np.ndarray:
        return np.abs(self.phi_p_ellipse_template(self.pure_rotation_phi)(6, gamma, xyt))

    def PureRotationAngle(self, xyt: ut.CArray) -> np.ndarray:
        return super().pure_rotation_phi(xyt)

    def S_center(self, xyt: ut.CArray) -> np.ndarray:
        return super().S_center(xyt)

    def S_local(self, xyt: ut.CArray) -> np.ndarray:
        """
        Use the director as the eigenvector of Q-tensor, then S_local is the eigenvalue.
        """
        sum_ux, sum_uy = self.Q_tensor(xyt)
        S = np.sqrt(sum_ux.data ** 2 + sum_uy.data ** 2)
        return S.data / (self.weight_sums.data + 1)

    def DirectorAngle(self, xyt: ut.CArray) -> np.ndarray:
        """
        Calculate the director as the eigenvector of Q-tensor.
        """
        sum_ux, sum_uy = self.Q_tensor(xyt)
        S = np.sqrt(sum_ux.data ** 2 + sum_uy.data ** 2)
        return np.atan2(sum_uy.data, sum_ux.data + S)

    def CrystalNematicAngle(self, xyt: ut.CArray) -> np.ndarray:
        phi = self.PureRotationAngle(xyt)
        theta = self.DirectorAngle(xyt)
        return np.cos(2 * (theta - phi))
