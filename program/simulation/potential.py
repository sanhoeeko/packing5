import numpy as np

from . import utils as ut
from .kernel import ker


class RadialFunc:
    def __init__(self, Vr_data: ut.CArray, dVr_data: ut.CArray):
        self.Vr_data = Vr_data
        self.dVr_data = dVr_data

    def name(self):
        # @property
        raise NotImplementedError  # to be inherited


class PowerFunc(RadialFunc):
    def __init__(self, alpha: float):
        self.power = alpha
        x2 = np.linspace(0, 4, num=ut.sz1d, endpoint=True, dtype=np.float32)
        x = np.sqrt(x2)
        arr = ut.CArray(alpha * np.power(2 - x, alpha - 1) / x, np.float32)
        arr.data[0] = 0
        super().__init__(Vr_data=ut.CArray(np.power(2 - x, alpha), np.float32),
                         dVr_data=arr)

    @property
    def name(self):
        return f"power({'%.1f' % self.power})"


class ScreenedCoulomb(RadialFunc):
    def __init__(self, r0: float):
        self.r0 = r0
        V0 = 1e-1
        x2 = np.linspace(0, 4, num=ut.sz1d, endpoint=True, dtype=np.float32)
        x = np.sqrt(x2)
        r = x / r0
        vr = ut.CArray(V0 * (np.exp(-r) / r - np.exp(-r0) / r0), np.float32)
        vr.data[0] = 0
        dvr = ut.CArray(-V0 * np.exp(-r) * (x + r0) / x ** 2, np.float32)
        dvr.data[0] = 0
        super().__init__(vr, dvr)

    @property
    def name(self):
        return f"screened coulomb({'%.1f' % self.r0})"


class ModifiedPower(RadialFunc):
    def __init__(self, alpha: float, x0: float):
        """
        :param alpha: f(r)=(2-r)^α for x0<r<2; f(r)=a/r^k+b for 0<r<x0.
        :param x0: f(r) is C-2 continuous at x0.
        """
        self.alpha = alpha
        self.x0 = x0
        a = ((2 - x0) ** alpha * x0 ** ((x0 - x0 * alpha) / (-2 + x0)) * alpha) / (-2 + x0 * alpha)
        b = (2 * (2 - x0) ** alpha) / (2 - x0 * alpha)
        k = (2 - x0 * alpha) / (-2 + x0)

        def radial_func(r):
            if r == 0: return 1e12
            return a / r ** k + b if r < x0 else (2 - r) ** alpha

        def d_radial_func(r):
            if r == 0: return 0
            return -a * k * r ** (-1 - k) if r < x0 else -(2 - r) ** (-1 + alpha) * alpha

        x = np.sqrt(np.linspace(0, 4, num=ut.sz1d, endpoint=True, dtype=np.float32))
        vr = ut.CArray(np.vectorize(radial_func)(x), np.float32)
        dvr = ut.CArray(np.vectorize(d_radial_func)(x), np.float32)
        super().__init__(vr, dvr)

    @property
    def name(self):
        return f"modified power({'%.1f' % self.alpha}, {'%.2f' % self.x0})"


class PotentialBase:
    def __init__(self, type_id: int, vr: RadialFunc):
        self.radial_func = vr
        self.shape_ptr = 0
        self.particle_shape_type = type_id

    def __del__(self):
        ker.dll.delParticleShape(self.shape_ptr, self.particle_shape_type)

    def cal_potential(self, threads: int):
        self.table = ut.CArrayFZeros(ut.potential_table_shape)
        self.shape_ptr = self._get_shape_ptr(threads)
        print(f"Successfully load potential, {self.tag}")
        return self

    def _get_shape_ptr(self, threads: int) -> int:
        raise NotImplementedError  # to be inherited

    def tag(self) -> dict:
        # @property
        raise NotImplementedError  # to be inherited


class RodPotential(PotentialBase):
    def __init__(self, n: int, d: float, vr: RadialFunc):
        super().__init__(0, vr)
        self.n = np.int32(n)
        self.d = np.float32(d)

    def _get_shape_ptr(self, threads: int) -> int:
        return ker.dll.addRodShape(threads, self.n, self.d, self.table.ptr, self.radial_func.Vr_data.ptr)

    @property
    def tag(self) -> dict:
        return {
            'n': self.n,
            'd': self.d,
            'scalar': self.radial_func.name,
            'shape': 'rod'
        }


class SegmentPotential(PotentialBase):
    def __init__(self, gamma: float, vr: RadialFunc):
        super().__init__(1, vr)
        self.gamma = gamma

    def _get_shape_ptr(self, threads: int) -> int:
        return ker.dll.addSegmentShape(threads, self.gamma, self.table.ptr, self.radial_func.Vr_data.ptr)

    @property
    def tag(self) -> dict:
        return {
            'gamma': self.gamma,
            'scalar': self.radial_func.name,
            'shape': 'segment'
        }
