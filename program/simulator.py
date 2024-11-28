import traceback

import numpy as np

from h5tools.dataset import SimulationData
from h5tools.utils import randomString
from simulation.potential import Potential
from simulation.state import State


class Simulator:
    def __init__(self, N: int, n: int, d: float, A0: float, B0: float, potential: Potential):
        self.simulation_id = randomString()
        self.state = State.random(N, n, d, A0, B0).train(potential)
        self.metadata = self.state.metadata.copy()

    @property
    def potential_tag(self):
        return self.state.n, self.state.d

    @classmethod
    def loadState(cls, s: State, potential: Potential, state_id: str = None):
        obj = cls(s.N, s.n, s.d, s.A, s.B, potential)
        obj.state = s.copy()
        if state_id is not None:
            obj.simulation_id = state_id
        return obj

    @classmethod
    def fromPackingFractionPhi(cls, N: int, n: int, d: float, phi0: float, Gamma0: float, potential: Potential):
        gamma = 1 + (n - 1) * d / 2
        B = np.sqrt(N * (np.pi + 4 * (gamma - 1)) / (np.pi * Gamma0 * phi0)) / gamma
        A = Gamma0 * B
        return cls(N, n, d, A, B, potential)

    def schedule(self, max_relaxation, descent_curve_stride):
        self.max_relaxation = int(max_relaxation)
        self.descent_curve_stride = int(descent_curve_stride)
        self.dataset = SimulationData(f'{self.simulation_id}.h5', list(zip(State.metalist, self.metadata)),
                                      descent_curve_size=max_relaxation // descent_curve_stride)
        return self

    def setCompressMethod(self, func_A, func_B, max_step_size: float):
        self.state.boundary.setCompressMethod(func_A, func_B, max_step_size)
        return self

    def fetchData(self, grads):
        return {
            'configuration': self.state.xyt3d(),
            'descent_curve': grads[::self.descent_curve_stride].copy(),
        }

    def save(self, grads):
        self.dataset.append(self.state.metadata, self.fetchData(grads))

    def execute(self):
        try:
            grads = self.state.initAsDisks(self.max_relaxation)
            self.save(grads)
            for i in range(100):
                if self.state.phi > 1.0:
                    break
                self.state.boundary.compress(i)
                grads = self.state.equilibrium(self.max_relaxation)
                self.save(grads)
        except Exception as e:
            print(f"An exception occurred in simulation [{self.simulation_id}]!\n")
            traceback.print_exc()
