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
            ('addRodShape', [ct.c_int] * 2 + [ct.c_float] + [ct.c_void_p] * 2, ct.c_void_p),
            ('addSegmentShape', [ct.c_int, ct.c_float] + [ct.c_void_p] * 2, ct.c_void_p),
            ('delParticleShape', [ct.c_void_p], None),
            ('GridLocate', [ct.c_void_p] * 2 + [ct.c_int] * 4, None),
            ('GridTransform', [ct.c_void_p] * 2 + [ct.c_int], None),
            ('CalGradient', [ct.c_void_p] * 5 + [ct.c_int] * 3, None),
            ('CalGradientAsDisks', [ct.c_void_p] * 5 + [ct.c_int] * 3, None),
            ('CalGradientAndEnergy', [ct.c_void_p] * 5 + [ct.c_int] * 3, None),
            ('MinDistanceRij', [ct.c_void_p] * 2 + [ct.c_int] * 3, ct.c_float),
            ('AverageDistanceRij', [ct.c_void_p] * 2 + [ct.c_int] * 3, ct.c_float),
            ('RijRatio', [ct.c_void_p] * 2 + [ct.c_int] * 3, ct.c_float),
            ('MinDistanceRijFull', [ct.c_void_p, ct.c_int], ct.c_float),
            ('isOutOfBoundary', [ct.c_void_p] * 2 + [ct.c_int], ct.c_int),
            ('ClipGradient', [ct.c_void_p, ct.c_int], None),
            ('SumTensor4', [ct.c_void_p] * 2 + [ct.c_int], None),
            ('AddVector4', [ct.c_void_p] * 3 + [ct.c_int, ct.c_float], None),
            ('AddVector4FT', [ct.c_void_p] * 3 + [ct.c_int, ct.c_float, ct.c_float], None),
            ('PerturbVector4', [ct.c_void_p, ct.c_int, ct.c_float], None),
            ('FastClear', [ct.c_void_p, ct.c_int], None),
            ('HollowClear', [ct.c_void_p, ct.c_int, ct.c_int], None),
            ('FastNorm', [ct.c_void_p, ct.c_int], ct.c_float),
            ('FastMask', [ct.c_void_p] * 2 + [ct.c_int], None),
            ('GenerateMask', [ct.c_void_p, ct.c_int, ct.c_float], None),
            ('MaxAbsVector4', [ct.c_void_p, ct.c_int], ct.c_float),
            ('CwiseMulVector4', [ct.c_void_p, ct.c_int, ct.c_float], None),
            ('CreateLBFGS', [ct.c_int, ct.c_void_p, ct.c_void_p], ct.c_void_p),
            ('DeleteLBFGS', [ct.c_void_p], None),
            ('LbfgsInit', [ct.c_void_p, ct.c_float], None),
            ('LbfgsUpdate', [ct.c_void_p], None),
            ('LbfgsDirection', [ct.c_void_p] * 2, None),
            ('AverageState', [ct.c_float] + [ct.c_void_p] * 3 + [ct.c_int] * 2, None),
            ('AverageStateZeroTemperature', [ct.c_void_p] * 3 + [ct.c_int] * 2, ct.c_float),
        )

    def setTypes(self, *tup):
        for func_name, argtypes, restype in tup:
            getattr(self.dll, func_name).argtypes = argtypes
            getattr(self.dll, func_name).restype = restype


ker = Kernel()
