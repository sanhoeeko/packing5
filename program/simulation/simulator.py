import traceback

import numpy as np

import default
import simulation.utils as ut
from h5tools.dataset import SimulationData
from h5tools.utils import randomString
from . import boundary, stepsize
from .potential import Potential, PowerFunc
from .state import State


def settings():
    return np.zeros((1,), dtype=[('has_potential', 'b1'), ('has_compress', 'b1'), ('has_schedule', 'b1')])


def all_true(struct: np.ndarray) -> bool:
    return all(struct[0][field] for field in struct.dtype.names)


class Simulator(ut.HasMeta):
    meta_hint = ("id: S4, N: i4, n: i4, d: f4, gamma: f4, A0: f4, B0: f4, rho0: f4, phi0: f4, "
                 "potential_shape: S32, potential_scalar: S32, if_cal_energy: i4")

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
        self.current_step_size = default.max_step_size
        self.current_ge = None

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

    def schedule(self, max_relaxation, descent_curve_stride, cal_energy=False):
        self.max_relaxation = int(max_relaxation)
        self.descent_curve_stride = int(descent_curve_stride)
        self.if_cal_energy = cal_energy
        self.has_settings['has_schedule'] = True
        return self

    def setCompressMethod(self, func_A, func_B, max_step_size: float):
        self.state.boundary.setCompressMethod(func_A, func_B, max_step_size)
        self.has_settings['has_compress'] = True
        return self

    def fetchData(self, grads):
        return {
            'configuration': self.state.xyt3d(),
            'descent_curve': grads.astype(np.float32),
        }

    def save(self, grads):
        self.dataset.append(self.state.metadata, self.fetchData(grads))

    def execute(self):
        assert self.is_setting_complete()
        self.create_dataset()
        try:
            self.state.initAsDisks()
            for i in range(default.max_compress_turns):
                if self.state.phi > default.terminal_phi: break
                self.state.boundary.compress(i)
                current_speed = self.equilibrium()
                self.save(self.current_ge)
                print(f"[{self.id}] Compress {i}: {round(current_speed)} it/s")
        except Exception as e:
            print(f"An exception occurred in simulation [{self.id}]!\n")
            traceback.print_exc()

    def equilibrium(self) -> float:
        """
        :return: current speed. Unit: it/s.
        All black magics for gradient descent should be here.
        """
        with ut.Timer() as timer:
            if self.state.CalEnergy_pure() < 100:
                self.state.brown(1e-3, 10000, 1000)
                relaxations_1 = 10000
            else:
                relaxations_1 = 0

            self.current_step_size = stepsize.findBestStepsize(
                self.state, default.max_step_size, default.step_size_searching_samples
            )
            relaxations_2, final_grad, ge_array_2 = self.state.fineRelax(
                self.current_step_size, self.max_relaxation, self.descent_curve_stride, self.if_cal_energy
            )

            self.current_relaxations = relaxations_1 + relaxations_2
            self.current_ge = ge_array_2
        return self.current_relaxations / timer.elapse_t


def createSimulator(N, n, d, phi0, Gamma0, compress_func_A, compress_func_B):
    return (Simulator.fromPackingFractionPhi(N, n, d, phi0, Gamma0)
            .setCompressMethod(compress_func_A, compress_func_B, default.max_compress)
            .schedule(default.max_relaxation, default.descent_curve_stride, default.if_cal_energy)
            )


def testSingleThread(profile=True):
    N = 200
    n = 5
    d = 0.025
    phi0 = 0.7
    Gamma0 = 1
    compress_func_A = boundary.NoCompress()
    compress_func_B = boundary.RatioCompress(0.001)
    ex = createSimulator(N, n, d, phi0, Gamma0, compress_func_A, compress_func_B)
    ex.setPotential(Potential(n, d, PowerFunc(2.5)))
    ex.state.gradient.potential.cal_potential(4)
    if profile:
        with ut.Profile('../main.prof'):
            try:
                ex.execute()
            except KeyboardInterrupt:
                pass  # terminate the simulation and collect profiling data
    else:
        ex.execute()
