import numpy as np

from .kernel import ker


class Boundary:
    def __init__(self, boundary_a: float, boundary_b: float):
        self.A, self.B = boundary_a, boundary_b
        self.max_step_size = 1
        self.func_A = None
        self.func_B = None

    def refresh(self):
        pass  # to be inherited

    def compress(self, t: float):
        self.A = self.func_A(t, self.A)
        self.B = self.func_B(t, self.B)
        self.refresh()

    def setCompressMethod(self, func_A, func_B, max_step_size: float):
        self.func_A = self.restrict_compress_wrapper_A(func_A)
        self.func_B = self.restrict_compress_wrapper_B(func_B)
        self.max_step_size = max_step_size
        return self

    def restrict_compress_wrapper_A(self, compress_func):
        def inner(t, x):
            dA = compress_func(t, x) - self.A
            if abs(dA) > self.max_step_size:
                return self.A + np.sign(dA) * self.max_step_size
            else:
                return self.A + dA

        return inner

    def restrict_compress_wrapper_B(self, compress_func):
        def inner(t, x):
            dB = compress_func(t, x) - self.B
            if abs(dB) > self.max_step_size:
                return self.B + np.sign(dB) * self.max_step_size
            else:
                return self.B + dB

        return inner


class EllipticBoundary(Boundary):
    def __init__(self, boundary_a: float, boundary_b: float):
        super().__init__(boundary_a, boundary_b)
        self.ptr = ker.dll.addEllipticBoundary(self.A, self.B)

    def __del__(self):
        ker.dll.delEllipticBoundary(self.ptr)

    def refresh(self):
        ker.dll.setEllipticBoundary(self.ptr, self.A, self.B)


def NoCompress():
    def inner(t, x):
        return x

    return inner


def RatioCompress(ratio: float):
    q = 1 - ratio

    def inner(t, x):
        return q * x

    return inner
