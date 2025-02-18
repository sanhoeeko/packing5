import os
from threading import Thread

import numpy as np

from h5tools.h5tools import stack_h5_datasets
from h5tools.utils import dict_to_numpy_struct, randomString
from .potential import RadialFunc, RodPotential
from .simulator import createSimulator


class Ensemble:
    def __init__(self, n: int, d: float, scalar_func: RadialFunc):
        self.n, self.d = n, d
        self.id = randomString()
        self.simulators = []
        self.potential = RodPotential(n, d, scalar_func)
        self.dataset = None

    def setSimulationProperties(self, N, phi0, Gamma0, compress_func_A, compress_func_B):
        self.kwargs = {
            'N': N, 'n': self.n, 'd': self.d, 'phi0': phi0, 'Gamma0': Gamma0,
            'compress_func_A': compress_func_A, 'compress_func_B': compress_func_B,
        }
        return self

    @property
    def metadata(self) -> np.ndarray:
        return dict_to_numpy_struct(self.potential.tag, 32)

    def addSimulator(self):
        """
        This method cannot be called in parallel!
        """
        Id = f"{self.id}_{len(self.simulators)}"
        simulator = createSimulator(Id, **self.kwargs)
        simulator.setPotential(self.potential)
        self.simulators.append(simulator)

    def setReplica(self, n_replica: int):
        for i in range(n_replica):
            self.addSimulator()
        return self

    def init(self):
        self.potential.cal_potential(threads=len(self.simulators))

    def pack(self):
        stack_h5_datasets(self.id)
        # delete old files
        for i in range(len(self.simulators)):
            os.remove(f"{self.id}_{i}.h5")

    def execute(self):
        self.init()
        self.threads = []
        for s in self.simulators:
            thread = Thread(target=s.execute)
            self.threads.append(thread)
            thread.start()
        for thread in self.threads:
            thread.join()
        self.pack()


def CreateEnsemble(N, n, d, phi0, Gamma0, compress_func_A, compress_func_B, radial_func):
    return Ensemble(n, d, radial_func).setSimulationProperties(
        N, phi0, Gamma0, compress_func_A, compress_func_B
    )
