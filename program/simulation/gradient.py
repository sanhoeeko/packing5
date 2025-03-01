import numpy as np

import default
from . import utils as ut
from .kernel import ker


class GradientMatrix:
    def __init__(self, state, grid):
        self.N = state.N
        self.state = state
        self.grid = grid
        self.potential = None
        self.capacity = ut.cores
        self.data = ut.CArrayFZeros((self.N, self.capacity, 4))
        self.sum = GradientSum(self)

    @property
    def params(self):
        return (
            self.potential.shape_ptr, self.state.xyt.ptr, self.state.boundary.ptr, self.grid.grid.ptr, self.data.ptr,
            self.grid.lines, self.grid.cols, self.N
        )

    @property
    def params_zero(self):
        return (
            0, self.state.xyt.ptr, self.state.boundary.ptr, self.grid.grid.ptr, self.data.ptr,
            self.grid.lines, self.grid.cols, self.N
        )

    def zero_grad(self):
        ker.dll.FastClear(self.data.ptr, self.N * self.capacity * 4)
        if self.capacity > 1:
            ker.dll.FastClear(self.sum.data.ptr, self.N * 4)

    def calGradientAndEnergy(self):
        ker.dll.CalGradientAndEnergy(*self.params)

    def calGradient(self):
        ker.dll.CalGradient(*self.params)

    def calGradientAsDisks(self):
        ker.dll.CalGradientAsDisks(*self.params_zero)

    def stochasticCalGradient(self, p: float):
        def inner():
            ker.dll.StochasticCalGradient(p, *self.params)

        return inner

    def stochasticCalGradientAsDisks(self, p: float):
        def inner():
            ker.dll.StochasticCalGradientAsDisks(p, *self.params_zero)

        return inner

    def minDistanceRij(self) -> np.float32:
        """
        Python code for MinDistanceRijFull:
            dx = self.state.xyt[:, 0:1] - self.state.xyt[:, 0:1].T
            dy = self.state.xyt[:, 1:2] - self.state.xyt[:, 1:2].T
            r2 = dx * dx + dy * dy + np.diag(np.full((self.N,), np.inf))
            return np.sqrt(np.min(r2))
        """
        # do not use this. I don't know why but there is a bug.
        return ker.dll.MinDistanceRij(self.state.xyt.ptr, self.grid.grid.ptr, self.grid.lines, self.grid.cols, self.N)
        # return self.state.min_dist

    def averageDistanceRij(self) -> np.float32:
        return ker.dll.AverageDistanceRij(self.state.xyt.ptr, self.grid.grid.ptr, self.grid.lines, self.grid.cols,
                                          self.N)

    def rijRatio(self) -> np.float32:
        return ker.dll.RijRatio(self.state.xyt.ptr, self.grid.grid.ptr, self.grid.lines, self.grid.cols, self.N)

    def isTooClose(self) -> bool:
        return self.rijRatio() < 0.1


class GradientSum:
    def __init__(self, Gij):
        self.N = Gij.N
        self.capacity = Gij.capacity
        self.src = Gij.data
        if self.capacity == 1:
            self.data = Gij.data.reshape(self.N, 4)  # self.data.ptr == Gij.data.ptr
        else:
            self.data = ut.CArrayFZeros((self.N, 4))
        self.mean_gradient_amp_cache = ut.Cache(0)
        self.max_gradient_amp_cache = ut.Cache(0)
        self.energy_cache = ut.Cache(0)

    def clear(self):
        self.mean_gradient_amp_cache.valid = False
        self.max_gradient_amp_cache.valid = False
        self.energy_cache.valid = False

    def g(self) -> ut.CArray:
        if self.capacity > 1:
            ker.dll.SumTensor4(self.src.ptr, self.data.ptr, self.N)
        return self.data

    def e(self) -> np.float32:
        return np.sum(self.data[:, 3])

    def E(self) -> np.float32:
        if not self.energy_cache.valid:
            self.g()
            self.energy_cache.set(self.e())
        return self.energy_cache._obj

    def G_mean(self):
        if not self.mean_gradient_amp_cache.valid:
            self.g()
            self.mean_gradient_amp_cache.set(self.data.norm(self.N))
        return self.mean_gradient_amp_cache._obj

    def G_max(self):
        if not self.max_gradient_amp_cache.valid:
            self.g()
            self.max_gradient_amp_cache.set(self.data.max_abs(self.N))
        return self.max_gradient_amp_cache._obj


class Optimizer:
    def __init__(self, state, noise_factor: float, momentum_beta: float, stochastic_p: float, as_disks: bool,
                 need_energy=False):
        self.N = state.N
        self.grid = state.grid
        self.momentum = ut.CArrayFZeros((self.N, 4))
        self.raw_gradient_cache = ut.CArrayFZeros((self.N, 4))
        self.mask = ut.CArray(np.zeros((self.N,), dtype=np.int32))
        self.particles_too_close_cache = False
        self.beta = np.float32(momentum_beta)
        self.noise_factor = np.float32(noise_factor)
        self.pure_gradient_func = state.CalGradient_pure
        func_name = 'calGradient' if not as_disks else 'calGradientAsDisks'
        if need_energy:
            func_name += "AndEnergy"
            if as_disks: raise ValueError("Energy is not available in this mode!")
        self.void_gradient_func = getattr(state.gradient, func_name)

        def __raw_gradient_func() -> ut.CArray:
            self.grid.gridLocate()
            self.void_gradient_func()
            gradient = state.gradient.sum.g()
            gradient.copyto(self.raw_gradient_cache)
            return gradient

        if default.enable_legal_check:
            def _raw_gradient_func() -> ut.CArray:
                gradient = __raw_gradient_func()
                self.particles_too_close_cache = state.gradient.isTooClose()
                return gradient
        else:
            _raw_gradient_func = __raw_gradient_func

        if stochastic_p != 1:
            def raw_gradient_func() -> ut.CArray:
                gradient = _raw_gradient_func()
                ker.dll.FastMask(gradient.ptr, self.mask.ptr, self.N)
                return gradient
        else:
            raw_gradient_func = _raw_gradient_func

        self.raw_gradient_func = raw_gradient_func

        if noise_factor == 0:
            self.noise_gradient_func = self.raw_gradient_func
        else:
            def noise_gradient_func() -> ut.CArray:
                gradient = self.raw_gradient_func()
                ker.dll.PerturbVector4(gradient.ptr, self.N, self.noise_factor)
                return gradient

            self.noise_gradient_func = noise_gradient_func

        if momentum_beta == 0:
            self.func = self.noise_gradient_func
        else:
            def momentum_gradient_func() -> ut.CArray:
                gradient = self.noise_gradient_func()
                ker.dll.CwiseMulVector4(self.momentum.ptr, self.N, self.beta)
                ker.dll.AddVector4(self.momentum.ptr, gradient.ptr, self.momentum.ptr, self.N, 1 - self.beta)
                return self.momentum

            self.func = momentum_gradient_func

    def init(self):
        if self.beta != 0:
            self.momentum = self.pure_gradient_func()

    def calGradient(self):
        return self.func()

    def gradientAmp(self) -> np.float32:
        return ker.dll.FastNorm(self.raw_gradient_cache.ptr, self.N * 4) / np.sqrt(self.N)

    def maxGradient(self) -> np.float32:
        return ker.dll.MaxAbsVector4(self.raw_gradient_cache.ptr, self.N)

    def initMask(self, p: float):
        ker.dll.GenerateMask(self.mask.ptr, self.N, p)
