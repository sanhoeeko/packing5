import ctypes as ct

from . import utils as ut

ut.setWorkingDirectory()


class Kernel:
    def __init__(self):
        self.dll = ct.CDLL('./x64/Release/analysisCpp.dll')
        self.setTypes(
            ('disksToVoronoiEdges', [ct.c_int] * 2 + [ct.c_void_p] * 2 + [ct.c_float] * 2, ct.c_int),
            ('trueDelaunay', [ct.c_int] * 2 + [ct.c_void_p] * 3 + [ct.c_float] * 2, ct.c_int),
            ('weightedDelaunay', [ct.c_int] * 2 + [ct.c_void_p] * 3 + [ct.c_float] * 2, ct.c_int),
            ('sumOverWeights', [ct.c_int] * 2 + [ct.c_void_p] * 4, None),
            ('sumOverNeighbors', [ct.c_int] * 2 + [ct.c_void_p] * 4, None),
            ('sumComplex', [ct.c_int] * 2 + [ct.c_void_p] * 4, None),
            ('sumAnisotropicComplex', [ct.c_int] * 2 + [ct.c_void_p] * 4, None),
            ('z_ij_power_p', [ct.c_int] * 2 + [ct.c_void_p] * 4 + [ct.c_float], None),
            ('orientation_diff_ij', [ct.c_int] * 2 + [ct.c_void_p] * 4, None),
            ('pure_rotation_direction_phi', [ct.c_int] * 2 + [ct.c_void_p] * 4, None),
            ('anisotropic_z_ij_power_p', [ct.c_int] * 2 + [ct.c_void_p] * 5 + [ct.c_float] * 2, None),
            ('RijRatio', [ct.c_void_p, ct.c_int], ct.c_float),
            ('CubicMinimum', [ct.c_float] * 4, ct.c_float),
        )

    def setTypes(self, *tup):
        for func_name, argtypes, restype in tup:
            getattr(self.dll, func_name).argtypes = argtypes
            getattr(self.dll, func_name).restype = restype


ker = Kernel()
