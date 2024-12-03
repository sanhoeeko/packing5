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
        self.A = self.restrict_A(self.func_A(t, self.A))
        self.B = self.restrict_B(self.func_B(t, self.B))
        self.refresh()

    def setCompressMethod(self, func_A, func_B, max_step_size: float):
        self.func_A = func_A
        self.func_B = func_B
        self.max_step_size = max_step_size
        return self

    def _restrict(self, current_value, target_value):
        delta = target_value - current_value
        if abs(delta) > self.max_step_size:
            return current_value + np.sign(delta) * self.max_step_size
        else:
            return current_value + delta

    def restrict_A(self, target_value):
        return self._restrict(self.A, target_value)

    def restrict_B(self, target_value):
        return self._restrict(self.B, target_value)


class EllipticBoundary(Boundary):
    def __init__(self, boundary_a: float, boundary_b: float):
        super().__init__(boundary_a, boundary_b)
        self.ptr = ker.dll.addEllipticBoundary(self.A, self.B)

    def __del__(self):
        ker.dll.delEllipticBoundary(self.ptr)

    def refresh(self):
        ker.dll.setEllipticBoundary(self.ptr, self.A, self.B)


def NoCompress():
    return lambda t, x: x


def RatioCompress(ratio: float):
    q = 1 - ratio
    return lambda t, x: q * x
