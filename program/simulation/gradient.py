import numpy as np

from . import utils as ut
from .kernel import ker


class GradientMatrix:
    def __init__(self, state, grid):
        self.N = state.N
        self.state = state
        self.grid = grid
        self.potential = None
        self.capacity = ut.cores * ut.max_neighbors
        self.z = ut.CArray(np.zeros((self.N,), dtype=np.int32))
        self.data = ut.CArrayFZeros((self.N, self.capacity, 4))
        self.sum = GradientSum(self)

    @property
    def params(self):
        return (
            self.potential.shape_ptr, self.state.xyt.ptr, self.state.boundary.ptr, self.grid.grid.ptr,
            self.data.ptr, self.z.ptr,
            self.grid.lines, self.grid.cols, self.N
        )

    @property
    def params_zero(self):
        return (
            0, self.state.xyt.ptr, self.state.boundary.ptr, self.grid.grid.ptr,
            self.data.ptr, self.z.ptr,
            self.grid.lines, self.grid.cols, self.N
        )

    def calGradientAndEnergy(self):
        ker.dll.CalGradientAndEnergy(*self.params)

    def calGradient(self):
        """
        status_code = ker.dll.CalGradient(*self.params)
        if status_code:
            raise ut.CalGradientException
        """
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
        self.z = Gij.z
        self.src = Gij.data
        self.data = ut.CArrayFZeros((self.N, 4))

    def g(self) -> ut.CArray:
        ker.dll.SumTensor4(self.z.ptr, self.src.ptr, self.data.ptr, self.N, self.capacity)
        # gradient is clipped only in "gradient only" mode
        # ker.dll.ClipGradient(self.data.ptr, self.N)
        return self.data

    def e(self) -> np.float32:
        return np.sum(self.data[:, 3])

    def E(self) -> np.float32:
        ker.dll.SumTensor4(self.z.ptr, self.src.ptr, self.data.ptr, self.N, self.capacity)
        return self.e()

    def gE(self) -> (ut.CArray, np.float32):
        ker.dll.SumTensor4(self.z.ptr, self.src.ptr, self.data.ptr, self.N, self.capacity)
        return self.data, self.e()


class Optimizer:
    def __init__(self, state, noise_factor: float, momentum_beta: float, stochastic_p: float, as_disks: bool,
                 need_energy=False):
        self.N = state.N
        self.grid = state.grid
        self.momentum = ut.CArrayFZeros((self.N, 4))
        self.raw_gradient_cache = ut.CArrayFZeros((self.N, 4))
        self.particles_too_close_cache = False
        self.beta = np.float32(momentum_beta)
        self.noise_factor = np.float32(noise_factor)
        self.pure_gradient_func = state.CalGradient_pure
        func_name = 'calGradient' if stochastic_p == 1 else 'stochasticCalGradient'
        if as_disks:
            func_name += 'AsDisks'
        if need_energy:
            func_name = 'calGradientAndEnergy'
        if stochastic_p == 1:
            self.void_gradient_func = getattr(state.gradient, func_name)
        else:
            self.void_gradient_func = getattr(state.gradient, func_name)(stochastic_p)

        def raw_gradient_func() -> ut.CArray:
            self.grid.gridLocate()
            self.void_gradient_func()
            self.particles_too_close_cache = state.gradient.isTooClose()
            gradient = state.gradient.sum.g()
            gradient.copyto(self.raw_gradient_cache)
            return gradient

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
