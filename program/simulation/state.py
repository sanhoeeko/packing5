import numpy as np

import default
from . import utils as ut
from .boundary import EllipticBoundary
from .gradient import Optimizer, GradientMatrix
from .grid import Grid
from .kernel import ker
from .lbfgs import LBFGS
from .mc import StatePool
from .potential import Potential
from .utils import NaNInGradientException, OutOfBoundaryException


class State(ut.HasMeta):
    meta_hint = "N: i4, A: f4, B: f4, gamma: f4, rho: f4, phi: f4, energy: f4, gradient_amp: f4"
    min_grad = 1e-3  # Used in `Simulator` class, `equilibrium` method

    def __init__(self, N: int, n: int, d: float, A: float, B: float, configuration: np.ndarray):
        super().__init__()
        self.N = np.int32(N)
        self.n = np.int32(n)
        self.d = np.float32(d)
        # derived properties
        self.gamma = 1 + (self.n - 1) * self.d / 2
        # data
        self.xyt = ut.CArrayF(configuration)
        self.boundary = EllipticBoundary(A, B)
        # optional objects
        self.optimizer: Optimizer = None
        self.grid = Grid(self)
        self.gradient = GradientMatrix(self, self.grid)
        self.descent_curve = ut.DescentCurve()
        self.state_pool = StatePool(self.N, default.descent_curve_stride)
        self.lbfgs_agent = LBFGS(self)

    @property
    def A(self):
        return self.boundary.A

    @property
    def B(self):
        return self.boundary.B

    @property
    def Gamma(self):
        return self.boundary.A / self.boundary.B

    @property
    def rho(self):
        return self.N / (np.pi * self.boundary.A * self.boundary.B)

    @property
    def phi(self):
        return self.rho * (np.pi + 4 * (self.gamma - 1)) / self.gamma ** 2

    @property
    def energy(self):
        return self.CalEnergy_pure()

    @property
    def gradient_amp(self):
        return ker.dll.FastNorm(self.CalGradient_pure().ptr, self.N * 4) / np.sqrt(self.N)

    @property
    def min_dist(self):
        return ker.dll.MinDistanceRijFull(self.xyt.ptr, self.N)

    @classmethod
    def random(cls, N, n, d, A, B):
        return cls(N, n, d, A, B, randomConfiguration(N, A, B))

    def setPotential(self, potential: Potential):
        self.gradient.potential = potential
        return self

    def setOptimizer(self, noise_factor: float, momentum_beta: float, stochastic_p: float, as_disks: bool,
                     need_energy=False):
        self.optimizer = Optimizer(self, noise_factor, momentum_beta, stochastic_p, as_disks, need_energy)
        self.optimizer.init()
        return self

    def copy(self, train=False) -> 'State':
        s = State(self.N, self.n, self.d, self.boundary.A, self.boundary.B, self.xyt.data.copy())
        if train:
            return s.setPotential(self.gradient.potential)
        else:
            return s

    def clear_dependency(self):
        ker.dll.HollowClear(self.grid.grid.ptr, self.grid.size, ut.max_neighbors)
        ker.dll.FastClear(self.gradient.z.ptr, self.N)

    def record(self, t: int, stride: int, gradient_amp: np.float32, cal_energy: bool):
        if t % stride == 0:
            self.descent_curve.current_gradient_curve[t // stride] = gradient_amp
            if cal_energy:
                self.descent_curve.current_energy_curve[t // stride] = self.CalEnergy_pure()

    def xyt3d(self) -> np.ndarray:
        return self.xyt[:, :3].copy()

    def isOutOfBoundary(self) -> bool:
        return bool(ker.dll.isOutOfBoundary(self.xyt.ptr, self.boundary.ptr, self.N))

    def averageRij_pure(self) -> np.float32:
        self.grid.gridLocate()
        self.grid.gridLocate()
        rij = self.gradient.averageDistanceRij()
        self.clear_dependency()
        return rij

    def legal_pure(self) -> bool:
        self.grid.gridLocate()
        is_too_close = self.gradient.isTooClose()
        self.clear_dependency()
        return not (is_too_close or self.isOutOfBoundary())

    def descent(self, gradient: ut.CArray, step_size: float) -> np.float32:
        g = gradient.norm(self.N)
        s = np.float32(step_size) / g
        if np.isnan(g) or np.isinf(g):
            raise NaNInGradientException()
        # this condition is to avoid division by zero
        if g > 1e-6:
            ker.dll.AddVector4(self.xyt.ptr, gradient.ptr, self.xyt.ptr, self.N, s)
        if self.isOutOfBoundary():
            raise OutOfBoundaryException()
        self.clear_dependency()
        return g

    def initAsDisks(self) -> (np.float32, np.float32):
        """
        All parameters in this method like `n_steps` and `step_size` cannot be changed.
        :return: (final gradient amplitude, final energy)
        """
        min_grad_init = 1e-3
        step_size_init = 1e-3
        n_steps_init = int(1e5)

        self.setOptimizer(0, 0, 1, True)
        gradient_amp = 0
        for t in range(n_steps_init):
            gradient_amp = self.descent(self.optimizer.calGradient(), step_size_init)
            if gradient_amp <= min_grad_init: break
        energy = self.CalEnergy_pure()
        return gradient_amp, energy

    def initAsDisksWithPhi(self, packing_fraction: float):
        current_packing_fraction = self.phi
        length_scale = np.sqrt(current_packing_fraction / packing_fraction)
        state = State.random(self.N, self.n, self.d, length_scale * self.A, length_scale * self.B)
        state.setPotential(self.gradient.potential)
        state.initAsDisks()
        self.xyt.set_data(state.xyt.data / length_scale)

    def brown(self, step_size: float, n_steps: int):
        stride = 10
        samples = 1000
        self.setOptimizer(0.1, 0.9, 1, False, True)

        for i in range(n_steps):
            gradient = self.optimizer.calGradient()
            self.descent(gradient, step_size)
        state_pool = StatePool(self.N, samples // stride)
        for i in range(samples):
            gradient = self.optimizer.calGradient()
            if i % stride == 0:
                if self.optimizer.particles_too_close_cache or self.isOutOfBoundary():
                    state_pool.add(self, 1e5)
                else:
                    energy = self.gradient.sum.e()
                    state_pool.add(self, energy)
            self.descent(gradient, step_size)

        min_state = state_pool.average(temperature=0.1)
        self.xyt.set_data(min_state.data)

    def sgd(self, step_size: float, n_steps: int):
        stride = default.descent_curve_stride
        self.setOptimizer(0.01, 0.1, 1, False)
        self.descent_curve.reserve(n_steps // stride)

        for t in range(int(n_steps) // stride):
            self.state_pool.clear()
            for i in range(stride):
                gradient = self.optimizer.calGradient()
                if self.optimizer.particles_too_close_cache or self.isOutOfBoundary():
                    self.state_pool.add(self, 1e5)
                    self.descent(gradient, step_size)
                else:
                    g = self.optimizer.gradientAmp()
                    self.state_pool.add(self, g)
                    self.descent(gradient, step_size)

            energy, min_state = self.state_pool.average_zero_temperature()
            self.xyt.set_data(min_state.data)
            gradient_amp = np.min(self.state_pool.energies.data)
            self.record(t * stride, stride, gradient_amp, default.if_cal_energy)
            if gradient_amp < 0.1: break

        self.descent_curve.join()

    def lbfgs(self, step_size: float, n_steps: int, stride: int) -> (int, np.ndarray, np.float32):
        """
        :return:
        if cal_energy:
            (relaxations_steps, final gradient amplitude, energy curve)
        else:
            (relaxations_steps, final gradient amplitude, gradient amplitudes)
        """
        min_grad = 0.1
        gradient_amp = 0

        self.lbfgs_agent.init(step_size)
        self.descent_curve.reserve(n_steps // stride)
        state_cache = self.xyt.copy()
        g_cache = self.CalGradient_pure().norm(self.N)

        for t in range(int(n_steps) // stride):
            self.state_pool.clear()
            for i in range(stride):
                gradient_amp = self.lbfgs_agent.gradientAmp()
                self.state_pool.add(self, gradient_amp)
                self.descent(self.lbfgs_agent.CalDirection(), step_size)
                self.lbfgs_agent.update()
                if gradient_amp <= min_grad:
                    self.descent_curve.join()
                    return t, gradient_amp

            energy, min_state = self.state_pool.average_zero_temperature()
            self.xyt.set_data(min_state.data)
            gradient_amp = np.min(self.state_pool.energies.data)
            self.record(t * stride, stride, gradient_amp, default.if_cal_energy)

        if gradient_amp > g_cache:
            self.xyt.set_data(state_cache.data)
            self.descent_curve.rewrite(g=g_cache)
        self.descent_curve.join()

    def CalGradient_pure(self) -> ut.CArray:
        """
        For class State, most methods are impure that leaves some caches.
        The method is pure and returns a gradient CArray.
        """
        self.grid.gridLocate()
        self.gradient.calGradient()
        gradient = self.gradient.sum.g()
        self.clear_dependency()
        return gradient

    def CalGradientNormalized_pure(self, power=1) -> ut.CArray:
        gradient = self.CalGradient_pure()
        g = ker.dll.FastNorm(gradient.ptr, self.N * 4)
        s = np.float32(1) / (g ** power)
        ker.dll.CwiseMulVector4(gradient.ptr, self.N, s)
        return gradient

    def CalEnergy_pure(self) -> np.float32:
        """
        For class State, most methods are impure that leaves some caches.
        The method is pure and returns an energy value.
        """
        self.grid.gridLocate()
        self.gradient.calGradientAndEnergy()
        energy = self.gradient.sum.E()
        self.clear_dependency()
        return energy


def randomConfiguration(N: int, A: float, B: float):
    xyt = np.zeros((N, 4))
    r = np.sqrt(np.random.rand(N))
    phi = np.random.rand(N) * (2 * np.pi)
    xyt[:, 0] = r * np.cos(phi) * A
    xyt[:, 1] = r * np.sin(phi) * B
    xyt[:, 2] = np.random.rand(N) * np.pi
    return xyt
