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

    def calGradient(self):
        return ker.dll.CalGradient(*self.params)

    def calGradientAsDisks(self):
        return ker.dll.CalGradientAsDisks(*self.params_zero)

    def stochasticCalGradient(self, p: float):
        def inner():
            return ker.dll.StochasticCalGradient(p, *self.params)

        return inner

    def stochasticCalGradientAsDisks(self, p: float):
        def inner():
            return ker.dll.StochasticCalGradientAsDisks(p, *self.params_zero)

        return inner


class GradientSum:
    def __init__(self, Gij):
        self.N = Gij.N
        self.capacity = Gij.capacity
        self.z = Gij.z
        self.src = Gij.data
        self.data = ut.CArrayFZeros((self.N, 4))
        self.energy = EnergySum(self)

    def g(self) -> ut.CArray:
        ker.dll.SumTensor4(self.z.ptr, self.src.ptr, self.data.ptr, self.N, self.capacity)
        return self.data


class EnergySum:
    def __init__(self, gi):
        self.src = gi.data
        self.data = np.float32(0)

    def E(self):
        self.data = np.sum(self.src.data[:, :, 3])


class Optimizer:
    def __init__(self, state, noise_factor: float, momentum_beta: float, stochastic_p: float, as_disks: bool):
        self.N = state.N
        self.grid = state.grid
        self.momentum = None
        self.beta = np.float32(momentum_beta)
        func_name = 'calGradient' if stochastic_p == 1 else 'stochasticCalGradient'
        if as_disks:
            func_name += 'AsDisks'
        if stochastic_p == 1:
            self.void_gradient_func = getattr(state.grid.gradient, func_name)
        else:
            self.void_gradient_func = getattr(state.grid.gradient, func_name)(stochastic_p)

        def raw_gradient_func() -> ut.CArray:
            self.grid.gridLocate()
            self.void_gradient_func()
            return state.grid.gradient.sum.g()

        self.raw_gradient_func = raw_gradient_func

        if noise_factor == 0:
            self.noise_gradient_func = self.raw_gradient_func
        else:
            def noise_gradient_func() -> ut.CArray:
                gradient = self.raw_gradient_func()
                ker.dll.PerturbVector4(gradient.ptr, self.N, noise_factor)
                return gradient

            self.noise_gradient_func = noise_gradient_func

        if momentum_beta == 0:
            self.func = self.noise_gradient_func
        else:
            def momentum_gradient_func() -> ut.CArray:
                gradient = self.noise_gradient_func()
                ker.dll.AddVector4(self.momentum.ptr, gradient.ptr, self.N, self.beta)
                return self.momentum

            self.func = momentum_gradient_func

    def init(self):
        if self.beta != 0:
            self.momentum = self.raw_gradient_func()

    def calGradient(self):
        return self.func()
