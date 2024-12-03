import numpy as np

import utils as ut
from voronoi import Voronoi


def orderParameterForTensor(A_upper_bound: float, B_upper_bound: float, gamma: float,
                            data: np.ndarray, order_parameter_name: str, weighted: bool):
    batch_size = data.shape[0]
    result = np.zeros((batch_size,))

    def inner(i):
        xyt = ut.CArray(data[i, :, :])
        result[i] = np.mean(getattr(Voronoi(gamma, A_upper_bound, B_upper_bound, xyt.data).delaunay(weighted),
                                    order_parameter_name)(xyt))

    ut.Map(4)(inner, range(batch_size))
    return result
