import ctypes as ct

from . import utils as ut

ut.setWorkingDirectory()


class Kernel:
    def __init__(self):
        self.dll = ct.CDLL('./x64/Release/packing5Cpp.dll')
        self.dll.init()
        self.setTypes(
            ('addEllipticBoundary', [ct.c_float] * 2, ct.c_void_p),
            ('delEllipticBoundary', [ct.c_void_p], None),
            ('setEllipticBoundary', [ct.c_void_p] + [ct.c_float] * 2, None),
            ('addParticleShape', [ct.c_int] * 2 + [ct.c_float] + [ct.c_void_p] * 2, ct.c_void_p),
            ('delParticleShape', [ct.c_void_p], None),
            ('GridLocate', [ct.c_void_p] * 2 + [ct.c_int] * 4, None),
            ('GridTransform', [ct.c_void_p] * 2 + [ct.c_int], None),
            ('CalGradient', [ct.c_void_p] * 6 + [ct.c_int] * 3, None),
            ('CalGradientAsDisks', [ct.c_void_p] * 6 + [ct.c_int] * 3, None),
            ('StochasticCalGradient', [ct.c_float] + [ct.c_void_p] * 6 + [ct.c_int] * 3, None),
            ('StochasticCalGradientAsDisks', [ct.c_float] + [ct.c_void_p] * 6 + [ct.c_int] * 3, None),
            ('CalGradientAndEnergy', [ct.c_void_p] * 6 + [ct.c_int] * 3, None),
            ('SumTensor4', [ct.c_void_p] * 3 + [ct.c_int], None),
            ('AddVector4', [ct.c_void_p] * 3 + [ct.c_int, ct.c_float], None),
            ('PerturbVector4', [ct.c_void_p, ct.c_int, ct.c_float], None),
            ('FastClear', [ct.c_void_p, ct.c_int], None),
            ('HollowClear', [ct.c_void_p, ct.c_int, ct.c_int], None),
            ('FastNorm', [ct.c_void_p, ct.c_int], ct.c_float),
            ('CwiseMulVector4', [ct.c_void_p, ct.c_int, ct.c_float], None),
            ('CreateLBFGS', [ct.c_int, ct.c_void_p, ct.c_void_p], ct.c_void_p),
            ('DeleteLBFGS', [ct.c_void_p], None),
            ('LbfgsInit', [ct.c_void_p, ct.c_float], None),
            ('LbfgsUpdate', [ct.c_void_p], None),
            ('LbfgsDirection', [ct.c_void_p] * 2, None),
            ('AverageState', [ct.c_float] + [ct.c_void_p] * 3 + [ct.c_int] * 2, None),
            ('AverageStateZeroTemperature', [ct.c_void_p] * 3 + [ct.c_int] * 2, None),
        )

    def setTypes(self, *tup):
        for func_name, argtypes, restype in tup:
            getattr(self.dll, func_name).argtypes = argtypes
            getattr(self.dll, func_name).restype = restype


ker = Kernel()
