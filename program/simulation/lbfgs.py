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

    def init(self, step_size: float):
        self.gradient_cache.set_data(self.state.CalGradient_pure().data)
        ker.dll.LbfgsInit(self.ptr, step_size)

    def update(self):
        self.gradient_cache.set_data(self.state.CalGradient_pure().data)
        ker.dll.LbfgsUpdate(self.ptr)

    def CalDirection(self) -> ut.CArray:
        ker.dll.LbfgsDirection(self.ptr, self.direction.ptr)
        ker.dll.ClipGradient(self.direction.ptr, self.N)
        return self.direction

    def gradientAmp(self) -> np.float32:
        return self.gradient_cache.norm(self.N)
