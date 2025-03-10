import numpy as np

from . import utils as ut
from .kernel import ker


class LBFGS:
    def __init__(self, state):
        self.N = state.N
        self.state = state
        self.gradient_cache = ut.CArrayFZeros((self.N, 4))
        self.direction = ut.CArrayFZeros((self.N, 4))
        self.ptr = ker.dll.CreateLBFGS(self.N, self.state.xyt.ptr, self.gradient_cache.ptr)

    def __del__(self):
        ker.dll.DeleteLBFGS(self.ptr)

    def init(self):
        self.gradient_cache.set_data(self.state.CalGradient_pure().data)

    def update(self):
        # x_new has been calculated by `State.descent` method
        self.gradient_cache.set_data(self.state.CalGradient_pure().data)  # calculate g_new
        ker.dll.LbfgsUpdate(self.ptr, self.state.xyt.ptr, self.gradient_cache.ptr)

    def CalDirection(self) -> ut.CArray:
        ker.dll.LbfgsDirection(self.ptr, self.direction.ptr)
        return self.direction

    def directionProduct(self) -> np.float32:
        return ker.dll.DotVector4(self.direction.ptr, self.gradient_cache.ptr, self.N)

    def gradientAmp(self) -> np.float32:
        return self.gradient_cache.norm(self.N)

    def maxGradientAmp(self) -> np.float32:
        return self.gradient_cache.max_abs(self.N)

    def energy(self) -> np.float32:
        return np.sum(self.gradient_cache[:, 3])
