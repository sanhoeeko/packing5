import time
import traceback

import numpy as np

import defaults
import simulation.utils as ut
from h5tools.dataset import SimulationData
from h5tools.utils import randomString
from simulation import boundary
from simulation.potential import Potential, PowerFunc
from simulation.state import State
from simulation.stepsize import findBestStepsize


def settings():
    return np.zeros((1,), dtype=[('has_potential', 'b1'), ('has_compress', 'b1'), ('has_schedule', 'b1')])


def all_true(struct: np.ndarray) -> bool:
    return all(struct[0][field] for field in struct.dtype.names)


class Simulator(ut.HasMeta):
    meta_hint = ("id: S4, N: i4, n: i4, d: f4, gamma: f4, "
                 "A0: f4, B0: f4, rho0: f4, phi0: f4, potential_shape: S32, potential_scalar: S32")

    def __init__(self, N: int, n: int, d: float, A0: float, B0: float):
        super().__init__()
        self.has_settings = settings()
        self.N = N
        self.n = n
        self.d = d
        self.A0, self.B0 = A0, B0
        self.id = randomString()
        self.state = State.random(N, n, d, A0, B0).train()

        # derived properties
        self.gamma = self.state.gamma
        self.rho0 = self.state.rho
        self.phi0 = self.state.phi
        self.dataset = None

        # cache for diagnosis
        self.current_step_size = defaults.step_size
        self.current_grads = None

    @property
    def current_relaxations(self):
        """
        :return: turns of relaxations used in current compress. This method is for profiling.
        """
        indices = np.where(np.isnan(self.current_grads))[0]
        if len(indices) == 0:
            return self.max_relaxation
        else:
            return indices[0]

    def is_setting_complete(self):
        return all_true(self.has_settings)

    def create_dataset(self):
        self.dataset = SimulationData(f'{self.id}.h5', self.metadata, self.state.dtype, 'state_table',
                                      descent_curve_size=self.max_relaxation // self.descent_curve_stride)

    @property
    def potential_shape(self) -> str:
        return self.state.gradient.potential.tag['shape']

    @property
    def potential_scalar(self) -> str:
        return self.state.gradient.potential.tag['scalar']

    @classmethod
    def loadState(cls, s: State, potential: Potential, state_id: str = None):
        obj = cls(s.N, s.n, s.d, s.A, s.B)
        obj.state = s.copy()
        if state_id is not None:
            obj.id = state_id
        return obj.setPotential(potential)

    @classmethod
    def fromPackingFractionPhi(cls, N: int, n: int, d: float, phi0: float, Gamma0: float):
        gamma = 1 + (n - 1) * d / 2
        B = np.sqrt(N * (np.pi + 4 * (gamma - 1)) / (np.pi * Gamma0 * phi0)) / gamma
        A = Gamma0 * B
        return cls(N, n, d, A, B)

    def setPotential(self, potential: Potential):
        self.state.setPotential(potential)
        self.has_settings['has_potential'] = True
        return self

    def schedule(self, max_relaxation, descent_curve_stride):
        self.max_relaxation = int(max_relaxation)
        self.descent_curve_stride = int(descent_curve_stride)
        self.has_settings['has_schedule'] = True
        return self

    def setCompressMethod(self, func_A, func_B, max_step_size: float):
        self.state.boundary.setCompressMethod(func_A, func_B, max_step_size)
        self.has_settings['has_compress'] = True
        return self

    def fetchData(self, grads):
        return {
            'configuration': self.state.xyt3d(),
            'descent_curve': grads[::self.descent_curve_stride].astype(np.float32),
        }

    def save(self, grads):
        self.dataset.append(self.state.metadata, self.fetchData(grads))

    def execute(self):
        assert self.is_setting_complete()
        self.create_dataset()
        try:
            self.current_grads = self.state.initAsDisks(self.max_relaxation, defaults.step_size)
            self.save(self.current_grads * defaults.step_size)
            for i in range(defaults.max_compress_turns):
                if self.state.phi > defaults.terminal_phi: break
                self.state.boundary.compress(i)
                current_speed = self.equilibrium()
                self.save(self.current_grads)
                print(f"[{self.id}] Compress {i}: {round(current_speed)} it/s")
        except Exception as e:
            print(f"An exception occurred in simulation [{self.id}]!\n")
            traceback.print_exc()

    def equilibrium(self) -> float:
        """
        :return: current speed. Unit: it/s.
        All black magics for gradient descent should be here.
        """
        start_t = time.perf_counter()
        self.current_grads = np.full((self.max_relaxation,), np.nan)
        part_length = self.max_relaxation // 200
        for i in range(200):
            self.current_step_size = findBestStepsize(self.state, 1e-3, 48)
            grads = self.state.equilibrium(part_length, self.current_step_size)
            self.current_grads[part_length * i:part_length * (i + 1)] = grads * self.current_step_size
        end_t = time.perf_counter()
        elapse_t = end_t - start_t
        return self.current_relaxations / elapse_t


def createSimulator(N, n, d, phi0, Gamma0, compress_func_A, compress_func_B):
    return (Simulator.fromPackingFractionPhi(N, n, d, phi0, Gamma0)
            .setCompressMethod(compress_func_A, compress_func_B, defaults.max_compress)
            .schedule(defaults.max_relaxation, defaults.descent_curve_stride)
            )


def testSingleThread(profile=True):
    N = 1000
    n = 3
    d = 0.05
    phi0 = 0.7
    Gamma0 = 1
    compress_func_A = boundary.NoCompress()
    compress_func_B = boundary.RatioCompress(0.001)
    ex = createSimulator(N, n, d, phi0, Gamma0, compress_func_A, compress_func_B)
    ex.setPotential(Potential(n, d, PowerFunc(2.5)))
    ex.state.gradient.potential.cal_potential(4)
    if profile:
        with ut.Profile('main.prof'):
            ex.execute()
    else:
        ex.execute()
