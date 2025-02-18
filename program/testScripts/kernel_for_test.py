def setWorkingDirectory():
    import os
    working_dir = "D:/py/packing5/program"
    os.chdir(working_dir)


setWorkingDirectory()

import ctypes as ct

import numpy as np

import simulation.utils as ut
from simulation.kernel import Kernel
from simulation.potential import RodPotential


class TestKernel(Kernel):
    def __init__(self):
        super().__init__()
        self.setTypes(
            ('interpolateGE', [ct.c_void_p] * 2 + [ct.c_float] * 4, None),
            ('preciseGE', [ct.c_void_p] * 4 + [ct.c_float] * 4, None),
        )
        # self.dll.getMirrorOf.argtypes = [ct.c_float] * 5
        # self.getMirrorOf = self.returnFixedArray(self.dll.getMirrorOf, 3)


ker = TestKernel()


class TestPotential:
    def __init__(self, potential: RodPotential):
        self.potential = potential
        self.cache = ut.CArrayFZeros((8,))

    def preciseGE(self, x: float, y: float, t1: float, t2: float) -> np.ndarray:
        ker.dll.preciseGE(self.potential.shape_ptr, self.potential.radial_func.Vr_data.ptr,
                          self.potential.radial_func.dVr_data.ptr, self.cache.ptr, x, y, t1, t2)
        return self.cache.data.copy()

    def interpolateGE(self, x: float, y: float, t1: float, t2: float) -> np.ndarray:
        if x * x + y * y >= 4:
            return np.zeros((8,), dtype=np.float32)
        ker.dll.interpolateGE(self.potential.shape_ptr, self.cache.ptr, x, y, t1, t2)
        return self.cache.data.copy()

    def precisePotential(self, x: float, y: float, t1: float, t2: float) -> np.float32:
        return self.preciseGE(x, y, t1, t2)[3]

    def interpolatePotential(self, x: float, y: float, t1: float, t2: float) -> np.float32:
        return self.interpolateGE(x, y, t1, t2)[3]

    def preciseGradient(self, x: float, y: float, t1: float, t2: float) -> np.ndarray:
        """
        :return: float32 array of shape (6,).
        """
        arr = self.preciseGE(x, y, t1, t2)
        return np.hstack([arr[0:3], arr[4:7]])

    def interpolateGradient(self, x: float, y: float, t1: float, t2: float) -> np.ndarray:
        """
        :return: float32 array of shape (6,).
        """
        arr = self.interpolateGE(x, y, t1, t2)
        return np.hstack([arr[0:3], arr[4:7]])
