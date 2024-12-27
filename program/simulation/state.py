import numpy as np

from . import utils as ut
from .boundary import EllipticBoundary
from .gradient import Optimizer, GradientMatrix
from .grid import Grid
from .kernel import ker
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

    def descent(self, gradient: ut.CArray, step_size: float) -> np.float32:
        g = ker.dll.FastNorm(gradient.ptr, self.N * 4) / np.sqrt(self.N)
        if np.isnan(g) or np.isinf(g):
            raise ValueError("NAN detected in gradient!")
        ker.dll.AddVector4(self.xyt.ptr, gradient.ptr, self.xyt.ptr, self.N, np.float32(step_size) / g)
        self.clear_dependency()
        return np.float32(g)

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

    def relax(self, step_size: float, n_steps: int, stride: int, cal_energy=False) -> (int, np.ndarray, np.float32):
        """
        :return:
        if cal_energy:
            (relaxations_steps, final gradient amplitude, energy curve)
        else:
            (relaxations_steps, final gradient amplitude, gradient amplitudes)
        """
        ge_array = np.full((n_steps // stride,), np.float32(np.nan))
        gradient_amp = 0

        self.setOptimizer(10.0, 0, 1, False, anneal_factor=0.999)
        for t in range(int(n_steps)):
            gradient_amp = self.descent(self.optimizer.calGradient(), step_size)
            if t % stride == 0:
                if cal_energy:
                    current_ge = self.CalEnergy_pure()
                else:
                    current_ge = gradient_amp
                ge_array[t // stride] = current_ge
            if gradient_amp <= State.min_grad:
                return t, gradient_amp, ge_array
        return n_steps, gradient_amp, ge_array

    def fineRelax(self, step_size: float, n_steps: int, stride: int, cal_energy=False) -> (int, np.ndarray, np.float32):
        min_grad_init = 1e-3

        ge_array = np.full((n_steps // stride,), np.float32(np.nan))
        gradient_amp = 0

        self.setOptimizer(0, 0, 1, True)
        for t in range(n_steps):
            gradient_amp = self.descent(self.optimizer.calGradient(), step_size)
            if t % stride == 0:
                if cal_energy:
                    current_ge = self.CalEnergy_pure()
                else:
                    current_ge = gradient_amp
                ge_array[t // stride] = current_ge
            if gradient_amp <= min_grad_init:
                return t, gradient_amp, ge_array
        return n_steps, gradient_amp, ge_array

    # def fineRelax(self) -> (np.float32, np.float32):
    #     """
    #     Relax near the minimum. Call scipy algorithms.
    #     :return: (final gradient amplitude, final energy)
    #     """
    #     agent = GEAgent(self)
    #     opt_result = opt.minimize(fun=agent.energy_func, x0=self.xyt.data[:, :3].reshape(-1),
    #                               method='TNC', jac=agent.gradient_func)
    #     print(opt_result)
    #     self.xyt.set_data(agent.convert_3N_array_to_4N_matrix(opt_result.x))
    #     final_grad = self.CalGradient_pure()
    #     final_grad_amp = ker.dll.FastNorm(final_grad.ptr, self.N * 4) / np.sqrt(self.N)
    #     final_energy = self.CalEnergy_pure()
    #     return final_grad_amp, final_energy

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

    def CalGradientNormalized_pure(self) -> ut.CArray:
        gradient = self.CalGradient_pure()
        g = ker.dll.FastNorm(gradient.ptr, self.N * 4) / np.sqrt(self.N)
        ker.dll.CwiseMulVector4(gradient.ptr, self.N, np.float32(1) / g)
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


class GEAgent:
    def __init__(self, state: State):
        self.state = state.copy(train=True)
        self.N = state.N
        self.N_void = np.zeros((self.N, 1), dtype=np.float32)
        self.x_cache = None
        self.g_cache = None
        self.E_cache = None

    def convert_3N_array_to_4N_matrix(self, x) -> np.ndarray:
        X = x.reshape(self.N, 3)
        return np.hstack([X, self.N_void])

    def compute(self, x: np.ndarray):
        """
        :param x: 1d array: xyt.reshape(-1)
        """
        self.x_cache = x  # exactly the same object as x
        self.state.xyt.set_data(self.convert_3N_array_to_4N_matrix(x))
        self.state.grid.gridLocate()
        self.state.gradient.calGradientAndEnergy()
        self.g_cache, self.E_cache = self.state.gradient.sum.gE()
        self.state.clear_dependency()

    def gradient_func(self, x: np.ndarray) -> np.ndarray:
        """
        :return: (3 * N,) array
        """
        if self.x_cache is None or not np.array_equal(x, self.x_cache):
            self.compute(x)
        return self.g_cache.data[:, :3].reshape(-1)

    def energy_func(self, x: np.ndarray) -> np.float32:
        if self.x_cache is None or not np.array_equal(x, self.x_cache):
            self.compute(x)
        return self.E_cache
