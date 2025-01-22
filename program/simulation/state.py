import numpy as np

import default
from . import utils as ut
from .boundary import EllipticBoundary
from .gradient import Optimizer, GradientMatrix
from .grid import Grid
from .kernel import ker
from .mc import StatePool
from .potential import Potential
from .utils import NaNInGradientException


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
        # self.lbfgs_agent = LBFGS(self)

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

    def setOptimizer(self, noise_factor: float, momentum_beta: float, stochastic_p: float, as_disks: bool):
        self.optimizer = Optimizer(self, noise_factor, momentum_beta, stochastic_p, as_disks)
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

    def legal_pure(self) -> bool:
        self.grid.gridLocate()
        is_too_close = self.gradient.isTooClose()
        self.clear_dependency()
        return not (is_too_close and self.isOutOfBoundary())

    def descent(self, gradient: ut.CArray, step_size: float) -> np.float32:
        g = ker.dll.FastNorm(gradient.ptr, self.N * 4)
        s = np.float32(step_size)
        if np.isnan(g) or np.isinf(g):
            raise NaNInGradientException()
        # this condition is to avoid division by zero
        if g > 1e-6:
            ker.dll.AddVector4(self.xyt.ptr, gradient.ptr, self.xyt.ptr, self.N, s)
        # if self.isOutOfBoundary():
        #     ker.dll.AddVector4(self.xyt.ptr, gradient.ptr, self.xyt.ptr, self.N, -s)
        #     ker.dll.AddVector4(self.xyt.ptr, self.lbfgs_agent.gradient_cache.ptr, self.xyt.ptr, self.N, 1e-4)
        #     # print("OOB!")
        # always clear dependency. If not, it will cause segment fault.
        self.clear_dependency()
        return np.float32(g / np.sqrt(self.N))

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

    def brown(self, step_size: float, n_steps: int):
        stride = default.descent_curve_stride
        self.setOptimizer(0.1, 0.9, 1, False)
        self.descent_curve.reserve(n_steps // stride)

        for t in range(int(n_steps) // stride):
            self.state_pool.clear()
            for i in range(stride):
                gradient = self.optimizer.calGradient()
                if self.optimizer.particles_too_close_cache:
                    self.state_pool.add(self, 1e6)
                    self.descent(gradient, step_size)
                else:
                    g = self.optimizer.maxGradient()
                    self.state_pool.add(self, g)
                    self.descent(gradient, step_size)

            energy, min_state = self.state_pool.average_zero_temperature()
            self.xyt.set_data(min_state.data)
            gradient_amp = np.min(self.state_pool.energies.data)
            self.record(t * stride, stride, gradient_amp, default.if_cal_energy)
            if gradient_amp < 0.2: break

        self.descent_curve.join()

    def sgd(self, step_size: float, n_steps):
        self.setOptimizer(0, 0.8, 1, False)
        self.descent_curve.reserve(n_steps // 100)
        for t in range(int(n_steps)):
            gradient_amp = self.descent(self.optimizer.calGradient(), step_size)
            self.record(t, 100, gradient_amp, default.if_cal_energy)
        self.descent_curve.join()

    def lbfgs(self, step_size: float, n_steps: int, stride: int) -> (int, np.ndarray, np.float32):
        """
        :return:
        if cal_energy:
            (relaxations_steps, final gradient amplitude, energy curve)
        else:
            (relaxations_steps, final gradient amplitude, gradient amplitudes)
        """
        min_grad = 0.2
        gradient_amp = 0

        self.lbfgs_agent.init(step_size)
        self.descent_curve.reserve(n_steps // stride)

        for t in range(n_steps):
            self.descent(self.lbfgs_agent.CalDirection(), step_size)
            gradient_amp = self.lbfgs_agent.gradientAmp()
            self.lbfgs_agent.update()
            self.record(t, stride, gradient_amp, default.if_cal_energy)

            if t % stride == 0:
                # step_size = stepsize.findBestStepsize(
                #     self, default.max_step_size, default.step_size_searching_samples
                #
                pass

            if gradient_amp <= min_grad:
                self.descent_curve.join()
                return t, gradient_amp

        self.descent_curve.join()
        return n_steps, gradient_amp

    def fineRelax(self, step_size: float, n_steps: int, stride: int) -> (int, np.ndarray, np.float32):
        """
        :return:
        if cal_energy:
            (relaxations_steps, final gradient amplitude, energy curve)
        else:
            (relaxations_steps, final gradient amplitude, gradient amplitudes)
        """
        from simulation import stepsize
        min_grad = 0.1
        gradient_amp = 0

        self.setOptimizer(0, 0.8, 1, False)
        self.descent_curve.reserve(n_steps // stride)

        for t in range(n_steps):
            gradient_amp = self.descent(self.optimizer.calGradient(), step_size)
            self.record(t, stride, gradient_amp, default.if_cal_energy)

            if t % stride == 0:
                step_size = 0.1 * stepsize.findGoodStepsize(
                    self, default.max_step_size, default.step_size_searching_samples
                )

            if gradient_amp <= min_grad:
                self.descent_curve.join()
                return t, gradient_amp

        self.descent_curve.join()
        return n_steps, gradient_amp

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
