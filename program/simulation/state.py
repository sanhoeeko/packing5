import numpy as np

from . import utils as ut
from .boundary import EllipticBoundary
from .gradient import Optimizer
from .grid import Grid
from .kernel import ker
from .potential import Potential


class State(ut.HasMeta):
    meta_hint = "N: i4, A: f4, B: f4, gamma: f4, rho: f4, phi: f4"
    min_grad = 1e-4

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
        self.grid = None
        self.optimizer = None

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

    @classmethod
    def random(cls, N, n, d, A, B):
        return cls(N, n, d, A, B, randomConfiguration(N, A, B))

    def setPotential(self, potential: Potential):
        self.grid.gradient.potential = potential
        return self

    def train(self):
        self.grid = Grid(self)
        return self

    def setOptimizer(self, noise_factor: float, momentum_beta: float, stochastic_p: float, as_disks: bool):
        self.optimizer = Optimizer(self, noise_factor, momentum_beta, stochastic_p, as_disks)
        self.optimizer.init()
        return self

    def copy(self) -> 'State':
        return State(self.N, self.n, self.d, self.boundary.A, self.boundary.B, self.xyt.copy())

    def clear_dependency(self):
        ker.dll.HollowClear(self.grid.grid.ptr, self.grid.size, ut.max_neighbors)
        ker.dll.FastClear(self.grid.gradient.z.ptr, self.N)

    def xyt3d(self) -> np.ndarray:
        return self.xyt[:, :3].copy()

    def descent(self, gradient: ut.CArray, step_size: float):
        g = ker.dll.FastNorm(gradient.ptr, self.N * 4) / np.sqrt(self.N)
        if np.isnan(g) or np.isinf(g):
            raise ValueError("NAN detected in gradient!")
        if g > State.min_grad:
            ker.dll.AddVector4(self.xyt.ptr, gradient.ptr, self.N, np.float32(step_size) / g)
        self.clear_dependency()
        return g

    def initAsDisks(self, n_steps):
        grads = np.zeros((n_steps,))
        self.setOptimizer(0, 0, 1, True)
        for t in range(int(n_steps)):
            grad = self.descent(self.optimizer.calGradient(), 1e-3)
            grads[t] = grad
            if grad <= State.min_grad: break
        return grads

    def equilibrium(self, n_steps):
        grads = np.zeros((n_steps,))
        self.setOptimizer(0, 0.9, 0.5, False)
        for t in range(int(n_steps)):
            grad = self.descent(self.optimizer.calGradient(), np.float32(1e-3))
            grads[t] = grad
            if grad <= State.min_grad: break
        return grads


def randomConfiguration(N: int, A: float, B: float):
    xyt = np.zeros((N, 4))
    r = np.sqrt(np.random.rand(N))
    phi = np.random.rand(N) * (2 * np.pi)
    xyt[:, 0] = r * np.cos(phi) * A
    xyt[:, 1] = r * np.sin(phi) * B
    xyt[:, 2] = np.random.rand(N) * np.pi
    return xyt
