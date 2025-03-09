import numpy as np

from . import utils as ut
from .boundary import EllipticBoundary
from .gradient import Optimizer, GradientMatrix
from .grid import Grid
from .kernel import ker
from .potential import PotentialBase
from .relaxation import DescentCurve
from .utils import NaNInGradientException, OutOfBoundaryException


class State(ut.HasMeta):
    meta_hint = ("N: i4, A: f4, B: f4, gamma: f4, rho: f4, phi: f4, "
                 "mean_gradient_amp: f4, max_gradient_amp: f4, energy: f4")

    def __init__(self, N: int, n: int, d: float, A: float, B: float, configuration: np.ndarray, train=True):
        super().__init__()
        self.N = np.int32(N)
        self.n = np.int32(n)
        self.d = np.float32(d)
        # derived properties
        self.gamma = 1 + (self.n - 1) * self.d / 2
        # data
        self.xyt = ut.CArrayF(configuration)
        self.boundary = EllipticBoundary(A, B)
        # necessary objects
        self.optimizer: Optimizer = None
        self.grid = Grid(self)
        self.gradient = GradientMatrix(self, self.grid)
        self.ge_valid = False
        self.compile_relaxation_functions()
        # optional objects
        if train:
            self.descent_curve = DescentCurve()

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
    def mean_gradient_amp(self):
        if not self.ge_valid: self.refreshGE()
        return self.gradient.sum.G_mean()

    @property
    def max_gradient_amp(self):
        if not self.ge_valid: self.refreshGE()
        return self.gradient.sum.G_max()

    @property
    def energy(self):
        if not self.ge_valid: self.refreshGE()
        return self.gradient.sum.E()

    @property
    def min_dist(self):
        return ker.dll.MinDistanceRijFull(self.xyt.ptr, self.N)

    @classmethod
    def random(cls, N, n, d, A, B):
        return cls(N, n, d, A, B, randomConfiguration(N, A, B))

    def calGradient(self):
        g = self.optimizer.calGradient()
        self.ge_valid = True
        return g

    def refreshGE(self):
        self.grid.gridLocate()
        self.gradient.calGradientAndEnergy()
        self.ge_valid = True

    def setPotential(self, potential: PotentialBase):
        self.gradient.potential = potential
        return self

    def setOptimizer(self, noise_factor: float, momentum_beta: float, stochastic_p: float, inertia: float = 1,
                     as_disks=False, need_energy=False):
        self.optimizer = Optimizer(self, noise_factor, momentum_beta, stochastic_p, as_disks, need_energy)
        self.optimizer.init()
        self.setInertia(inertia)
        return self

    def copy(self, train=False) -> 'State':
        s = State(self.N, self.n, self.d, self.boundary.A, self.boundary.B, self.xyt.data.copy(), train)
        return s.setPotential(self.gradient.potential)

    def clear_dependency(self):
        ker.dll.HollowClear(self.grid.grid.ptr, self.grid.size, ut.max_neighbors)
        self.gradient.zero_grad()
        self.gradient.sum.clear()
        self.ge_valid = False

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

    def _descent_inner(self, gradient: ut.CArray, s: float):
        pass  # defined in `self.setInertia`

    def setInertia(self, inertia: float):
        """
        :param inertia: set `inertia` to None to use separated normalization
        """
        if inertia is None:
            def _descent_inner(gradient: ut.CArray, s: float, g: np.float32):
                ft = gradient.max_ft(self.N)
                ker.dll.AddVector4FT(self.xyt.ptr, gradient.ptr, self.xyt.ptr, self.N, s / ft.force, s / ft.torque)
        elif inertia == 1:
            def _descent_inner(gradient: ut.CArray, s: float, g: np.float32):
                ker.dll.AddVector4(self.xyt.ptr, gradient.ptr, self.xyt.ptr, self.N, s / g)
        else:
            def _descent_inner(gradient: ut.CArray, s: float, g: np.float32):
                a = np.float32(s) / g
                ker.dll.AddVector4FT(self.xyt.ptr, gradient.ptr, self.xyt.ptr, self.N, a, a / inertia)
        self._descent_inner = _descent_inner
        return self

    def descent(self, gradient: ut.CArray, step_size: float) -> np.float32:
        # ft = gradient.max_ft(self.N)
        # print(ft.force, ft.torque)
        g = gradient.norm(self.N)
        # this condition is to avoid division by zero
        if g > 1e-6:
            if np.isnan(g) or np.isinf(g):
                raise NaNInGradientException()
            self._descent_inner(gradient, step_size, g)
        if self.isOutOfBoundary():
            raise OutOfBoundaryException()
        return g

    def initAsDisks(self) -> (np.float32, np.float32):
        """
        All parameters in this method like `n_steps` and `step_size` cannot be changed.
        :return: (final gradient amplitude, final energy)
        """
        min_grad_init = 1e-3
        step_size_init = 1e-3
        n_steps_init = int(1e5)

        self.setOptimizer(0, 0, 1, 1, True)
        gradient_amp = 0
        for t in range(n_steps_init):
            gradient_amp = self.descent(self.calGradient(), step_size_init)
            if gradient_amp <= min_grad_init: break
            self.clear_dependency()
        energy = self.CalEnergy_pure()
        return gradient_amp, energy

    def initAsDisksWithPhi(self, packing_fraction: float):
        current_packing_fraction = self.phi
        length_scale = np.sqrt(current_packing_fraction / packing_fraction)
        state = State.random(self.N, self.n, self.d, length_scale * self.A, length_scale * self.B)
        state.setPotential(self.gradient.potential)
        state.initAsDisks()
        self.xyt.set_data(state.xyt.data / length_scale)

    def initAsHardRods(self):
        status_code = ker.dll.SegmentInitialization(self.xyt.ptr, self.N, self.A, self.B, 1 - 1 / self.gamma)
        if status_code != 0:
            raise ut.InitFailException

    def compile_relaxation_functions(self):
        lst = [func(self) for func in self.relaxations]
        self.relaxations = lst
        return self

    def relax(self):
        for func in self.relaxations: func()

    @property
    def n_steps(self) -> int:
        return sum(map(lambda x: x.n_steps, self.relaxations))

    def CalGradient_pure(self) -> ut.CArray:
        """
        For class State, most methods are impure that leaves some caches.
        The method is pure and returns a gradient CArray.
        """
        self.grid.gridLocate()
        self.gradient.calGradient()
        gradient = self.gradient.sum.g().copy()
        self.clear_dependency()
        return gradient

    def CalEnergy_pure(self) -> np.float32:
        """
        For class State, most methods are impure that leaves some caches.
        The method is pure and returns an energy value.
        """
        self.grid.gridLocate()
        # is_too_close = self.gradient.isTooClose()
        # if is_too_close or self.isOutOfBoundary():
        #     return np.float32(np.inf)
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
