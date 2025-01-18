import traceback

import numpy as np

import default
import simulation.utils as ut
from h5tools.dataset import SimulationData
from .potential import Potential
from .state import State


def settings():
    return np.zeros((1,), dtype=[('has_potential', 'b1'), ('has_compress', 'b1'), ('has_schedule', 'b1')])


def all_true(struct: np.ndarray) -> bool:
    return all(struct[0][field] for field in struct.dtype.names)


class Simulator(ut.HasMeta):
    meta_hint = ("id: S4, N: i4, n: i4, d: f4, gamma: f4, A0: f4, B0: f4, rho0: f4, phi0: f4, "
                 "potential_shape: S32, potential_scalar: S32, if_cal_energy: i4")

    def __init__(self, Id: str, N: int, n: int, d: float, A0: float, B0: float):
        """
        :param id0x: Given by Ensemble.
        """
        super().__init__()
        self.has_settings = settings()
        self.N = N
        self.n = n
        self.d = d
        self.A0, self.B0 = A0, B0
        self.id = Id
        self.state = State.random(N, n, d, A0, B0)

        # derived properties
        self.gamma = self.state.gamma
        self.rho0 = self.state.rho
        self.phi0 = self.state.phi
        self.dataset = None

        # cache for diagnosis
        self.current_step_size = default.max_step_size

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
        obj = cls(None, s.N, s.n, s.d, s.A, s.B)
        obj.state = s.copy()
        if state_id is not None:
            obj.id = state_id
        return obj.setPotential(potential)

    @classmethod
    def fromPackingFractionPhi(cls, Id: str, N: int, n: int, d: float, phi0: float, Gamma0: float):
        gamma = 1 + (n - 1) * d / 2
        B = np.sqrt(N * (np.pi + 4 * (gamma - 1)) / (np.pi * Gamma0 * phi0)) / gamma
        A = Gamma0 * B
        return cls(Id, N, n, d, A, B)

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

    def fetchData(self):
        g_array, e_array = self.state.descent_curve.get(self.dataset.descent_curve_size)
        return {
            'configuration': self.state.xyt3d(),
            'gradient_curve': g_array,
            'energy_curve': e_array
        }

    def save(self):
        self.dataset.append(self.state.metadata, self.fetchData())

    def execute(self):
        assert self.is_setting_complete()
        self.create_dataset()
        try:
            self.state.initAsDisks()
            for i in range(default.max_compress_turns):
                if self.state.phi > default.terminal_phi: break
                self.state.descent_curve.clear()

                b0 = self.state.boundary.B
                self.state.boundary.compress(i)
                b1 = self.state.boundary.B
                self.state.xyt.data[:, 1] *= b1 / b0  # affine transformation

                current_speed = self.equilibrium()
                self.save()
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
            self.state.brown(1e-3, 10000)
            # relaxations_2, final_grad = self.state.lbfgs(
            #     5e-4, 20000, self.descent_curve_stride
            # )
            # relaxations_3, final_grad = self.state.fineRelax(
            #     1e-5, 100000, self.descent_curve_stride
            # )

            self.current_relaxations = default.max_pre_relaxation  # + relaxations_2 + relaxations_3
        return self.current_relaxations / timer.elapse_t


def createSimulator(Id: str, N, n, d, phi0, Gamma0, compress_func_A, compress_func_B):
    return (Simulator.fromPackingFractionPhi(Id, N, n, d, phi0, Gamma0)
            .setCompressMethod(compress_func_A, compress_func_B, default.max_compress)
            .schedule(default.max_relaxation, default.descent_curve_stride, default.if_cal_energy)
            )
