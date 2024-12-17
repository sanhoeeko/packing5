import numpy as np

from . import utils as ut
from .voronoi import Voronoi


def S_global(xyt: np.ndarray):
    t = xyt[:, 2]
    return np.cos(2 * t)


def OrderParameter(name: str):
    """
    Common order parameter interface. For both voronoi and non-voronoi order parameters.
    """
    if name == 'S_global':
        def inner(xyt, abg, weighted):
            return S_global(xyt)
    else:
        def inner(xyt, abg, weighted):
            xyt_c = ut.CArray(xyt)
            return getattr(Voronoi(abg[2], abg[0], abg[1], xyt_c.data).delaunay(weighted), name)(xyt_c)
    return inner
