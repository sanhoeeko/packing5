import os
import time
from collections.abc import Iterable

import dask
import numpy as np

from h5tools.dataset import compress_file
from h5tools.h5tools import stack_h5_datasets, pack_h5_files
from h5tools.utils import current_time, FuncCartesian, dict_to_numpy_struct, randomString
from .potential import RadialFunc, Potential, PowerFunc
from .simulator import createSimulator


class Ensemble:
    def __init__(self, n: int, d: float, scalar_func: RadialFunc):
        self.n, self.d = n, d
        self.id = randomString()
        self.simulators = []
        self.potential = Potential(n, d, scalar_func)
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
        tasks = [dask.delayed(s.execute)() for s in self.simulators]
        dask.compute(*tasks)
        self.pack()


def createEnsemble(radial_func):
    def inner(N, n, d, phi0, Gamma0, compress_func_A, compress_func_B):
        return Ensemble(n, d, radial_func).setSimulationProperties(
            N, phi0, Gamma0, compress_func_A, compress_func_B
        )

    return inner


class Experiment:
    """
    N: (int) particle number
    n: (int) number of disk per rod
    d: (float) distance between disks
    Gamma0: (float) initial boundary aspect ratio
    phi0: (float) initial packing fraction
    compress_func_A: (Callable)
    compress_func_B: (Callable)
    """

    def __init__(self, replica: int, **kwargs):
        self.start_time = current_time()
        # preprocess of inputs
        for k, v in kwargs.items():
            if not isinstance(v, Iterable):
                kwargs[k] = [v]
        self.thread_pool = []

        radial_func = PowerFunc(2.5)
        self.ensembles = FuncCartesian(createEnsemble(radial_func), **kwargs)
        for e in self.ensembles:
            e.setReplica(replica)

    @property
    def meta_dtype(self):
        return [('start_time', 'S32'), ('time_elapse', 'i8')]

    @property
    def metadata(self):
        # do not use `utils.dict_to_numpy_struct` because there is a special i8 type.
        return np.array([(self.start_time, self.time_elapse_s)], dtype=self.meta_dtype)

    @property
    def files(self) -> list:
        return [f'{s.id}.h5' for s in self.ensembles]

    def pack(self):
        pack_h5_files(self.files, 'data.h5')
        # delete old files
        for filename in self.files:
            os.remove(filename)

    def compress_file(self):
        original_size, compressed_size = compress_file('data.h5')
        compress_rate_percent = int((1 - compressed_size / original_size) * 100)
        print(f"Successfully compressed. Compress rate: {compress_rate_percent}%.")

    def execute(self):
        tasks = [dask.delayed(s.execute)() for s in self.ensembles]
        start = time.perf_counter()
        dask.compute(*tasks)
        end = time.perf_counter()
        self.time_elapse_s = int(end - start)
        self.pack()
        self.compress_file()
