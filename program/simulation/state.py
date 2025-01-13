import numpy as np

import default
from . import utils as ut
from .boundary import EllipticBoundary
from .gradient import Optimizer, GradientMatrix
from .grid import Grid
from .kernel import ker
from .lbfgs import LBFGS
from .potential import Potential


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
        self.grid: Grid = None
        self.gradient: GradientMatrix = None
        self.optimizer: Optimizer = None
        self.lbfgs_agent: LBFGS = None

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

    @classmethod
    def random(cls, N, n, d, A, B):
        return cls(N, n, d, A, B, randomConfiguration(N, A, B))

    def train(self):
        self.grid = Grid(self)
        self.gradient = GradientMatrix(self, self.grid)
        self.lbfgs_agent = LBFGS(self)
        return self

    def setPotential(self, potential: Potential):
        self.gradient.potential = potential
        return self

    def setOptimizer(self, noise_factor: float, momentum_beta: float, stochastic_p: float,
                     as_disks: bool, anneal_factor: float = 1):
        self.optimizer = Optimizer(self, noise_factor, momentum_beta, stochastic_p, as_disks, anneal_factor)
        self.optimizer.init()
        return self

    def copy(self, train=False) -> 'State':
        s = State(self.N, self.n, self.d, self.boundary.A, self.boundary.B, self.xyt.data.copy())
        if train:
            return s.train().setPotential(self.gradient.potential)
        else:
            return s

    def clear_dependency(self):
        ker.dll.HollowClear(self.grid.grid.ptr, self.grid.size, ut.max_neighbors)
        ker.dll.FastClear(self.gradient.z.ptr, self.N)

    def xyt3d(self) -> np.ndarray:
        return self.xyt[:, :3].copy()

    def isOutOfBoundary(self) -> bool:
        abs_xyt = np.abs(self.xyt.data)
        x_max = np.max(abs_xyt[:, 0])
        y_max = np.max(abs_xyt[:, 1])
        return x_max >= self.boundary.A or y_max >= self.boundary.B

    def MaskOutOfBoundary(self) -> np.ndarray:
        abs_xyt = np.abs(self.xyt.data)
        mask = np.bitwise_or(abs_xyt[:, 0] >= self.boundary.A, abs_xyt[:, 1] >= self.boundary.B)
        return np.where(mask)[0]

    def descent(self, gradient: ut.CArray, step_size: float) -> np.float32:
        g = ker.dll.FastNorm(gradient.ptr, self.N * 4)
        s = np.float32(step_size) / (np.sqrt(g))
        if np.isnan(g) or np.isinf(g):
            raise ValueError("NAN detected in gradient!")
        # this condition is to avoid division by zero
        if g > 1e-6:
            ker.dll.AddVector4(self.xyt.ptr, gradient.ptr, self.xyt.ptr, self.N, s)
        if self.isOutOfBoundary():
            # raise ValueError("Out of Boundary!")
            # if the descent makes a particle go out of boundary, do not update its coordinates
            # previous_xyt = self.xyt.copy()
            # ker.dll.AddVector4(self.xyt.ptr, gradient.ptr, previous_xyt.ptr, self.N, -s)
            # particle_id_out_of_boundary = self.MaskOutOfBoundary()
            # for i in particle_id_out_of_boundary:
            #     np.copyto(self.xyt.data[i, :], previous_xyt.data[i, :])
            ker.dll.AddVector4(self.xyt.ptr, gradient.ptr, self.xyt.ptr, self.N, -s)
            ker.dll.AddVector4(self.xyt.ptr, self.lbfgs_agent.gradient_cache.ptr, self.xyt.ptr, self.N, 1e-3)
            # print("OOB!")
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

    def brown(self, step_size: float, n_steps: int, n_samples) -> (int, np.ndarray, np.float32):
        self.setOptimizer(0.2, 0.8, 0.5, False)
        for t in range(int(n_steps)):
            self.descent(self.optimizer.calGradient(), step_size)
        state_pool = np.zeros((n_samples, self.N, 4))
        for t in range(int(n_samples)):
            self.descent(self.optimizer.calGradient(), step_size)
            state_pool[t, :, :] = self.xyt.data.copy()
        averaged_state = np.mean(state_pool, axis=0)
        self.xyt.set_data(averaged_state)

    def sgd(self, step_size: float, n_steps) -> (int, np.ndarray, np.float32):
        self.setOptimizer(0, 0.8, 0.2, False)
        for t in range(int(n_steps)):
            self.descent(self.optimizer.calGradient(), step_size)

    def lbfgs(self, step_size: float, n_steps: int, stride: int, cal_energy=False) -> (int, np.ndarray, np.float32):
        """
        :return:
        if cal_energy:
            (relaxations_steps, final gradient amplitude, energy curve)
        else:
            (relaxations_steps, final gradient amplitude, gradient amplitudes)
        """
        from simulation import stepsize
        min_grad = 1e-2
        ge_array = np.full((n_steps // stride,), np.float32(np.nan))
        gradient_amp = 0

        self.lbfgs_agent.init(step_size)
        for t in range(n_steps):
            self.descent(self.lbfgs_agent.CalDirection(), step_size)
            gradient_amp = self.lbfgs_agent.gradientAmp()
            self.lbfgs_agent.update()
            # print(t, self.CalEnergy_pure())
            if t % stride == 0:
                step_size = min(default.max_step_size, 10 * stepsize.findGoodStepsize(
                    self, default.max_step_size, default.step_size_searching_samples
                ))
                ge_array[t // stride] = self.CalEnergy_pure() if cal_energy else gradient_amp
            if gradient_amp <= min_grad:
                return t, gradient_amp, ge_array
        return n_steps, gradient_amp, ge_array

    def fineRelax(self, step_size: float, n_steps: int, stride: int, cal_energy=False) -> (int, np.ndarray, np.float32):
        """
        :return:
        if cal_energy:
            (relaxations_steps, final gradient amplitude, energy curve)
        else:
            (relaxations_steps, final gradient amplitude, gradient amplitudes)
        """
        from simulation import stepsize
        min_grad = 1e-2
        ge_array = np.full((n_steps // stride,), np.float32(np.nan))
        gradient_amp = 0

        self.setOptimizer(0, 0, 1, False)
        for t in range(n_steps):
            gradient_amp = self.descent(self.optimizer.calGradient(), step_size)
            if t % stride == 0:
                step_size = 0.1 * stepsize.findGoodStepsize(
                    self, default.max_step_size, default.step_size_searching_samples
                )
                ge_array[t // stride] = self.CalEnergy_pure() if cal_energy else gradient_amp
            if gradient_amp <= min_grad:
                return t, gradient_amp, ge_array
        return n_steps, gradient_amp, ge_array

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
        s = np.float32(1) / (g ** power / np.sqrt(self.N))
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
