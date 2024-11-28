import numpy as np

from . import utils as ut
from .kernel import ker


class RadialFunc:
    def __init__(self, Vr_data: ut.CArray, dVr_data: ut.CArray):
        self.Vr_data = Vr_data
        self.dVr_data = dVr_data


class PowerFunc(RadialFunc):
    def __init__(self, alpha: float):
        self.power = alpha
        x2 = np.linspace(0, 4, num=ut.sz1d, endpoint=True, dtype=np.float32)
        x = np.sqrt(x2)
        arr = ut.CArray(alpha * np.power(2 - x, alpha - 1) / x, np.float32)
        arr.data[0] = 0
        super().__init__(Vr_data=ut.CArray(np.power(2 - x, alpha), np.float32),
                         dVr_data=arr)


class Potential:
    def __init__(self, n: int, d: float, vr: RadialFunc, threads: int):
        self.table = ut.CArrayFZeros(ut.potential_table_shape)
        self.radial_func = vr
        self.n = np.int32(n)
        self.d = np.float32(d)
        self.shape_ptr = ker.dll.addParticleShape(threads, self.n, self.d, self.table.ptr, vr.Vr_data.ptr)
