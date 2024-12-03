import os
import threading
import time
from collections.abc import Iterable
from threading import Thread

import dask
import numpy as np

from h5tools.dataset import package_simulations_into_experiment
from h5tools.utils import flatten, current_time, FuncCartesian, dict_to_numpy_struct
from simulation.potential import RadialFunc, Potential, PowerFunc
from simulator import Simulator, createSimulator


class SimulationsCommonParticle:
    def __init__(self, n: int, d: float, scalar_func: RadialFunc):
        self.n, self.d = n, d
        self.simulators = []
        self.potential = Potential(n, d, scalar_func)
        self.dataset = None

    @property
    def metadata(self) -> np.ndarray:
        return dict_to_numpy_struct(self.potential.tag, 32)

    @property
    def id(self) -> list:
        return [s.id for s in self.simulators]

    def append(self, simulator: Simulator):
        assert simulator.state.n == self.n
        assert simulator.state.d == self.d
        simulator.setPotential(self.potential)
        self.simulators.append(simulator)

    def init(self):
        self.potential.cal_potential(threads=len(self.simulators))

    def package(self):
        temp_file_name = f"temp_{threading.get_ident()}.h5"
        self.dataset = package_simulations_into_experiment(
            temp_file_name, self.metadata, [s.dataset for s in self.simulators]
        )
        # delete old data
        for name in self.id:
            os.remove(f"{name}.h5")
        # rename new data
        os.rename(temp_file_name, f"{self.id[0]}.h5")

    def execute(self):
        self.init()
        self.threads = []
        for s in self.simulators:
            thread = Thread(target=s.execute)
            self.threads.append(thread)
            thread.start()
        for thread in self.threads:
            thread.join()
        self.package()


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
        self.SCPs = []
        self.thread_pool = []

        simulators = flatten([FuncCartesian(createSimulator, **kwargs) for _ in range(replica)])
        tags = set(map(lambda x: (x.n, x.d), simulators))
        radial_func = PowerFunc(2.5)
        for n, d in tags:
            self.SCPs.append(SimulationsCommonParticle(n, d, radial_func))
        for s in simulators:
            for scp in self.SCPs:
                if scp.n == s.n and scp.d == s.d:
                    scp.append(s)
                    break

    @property
    def meta_dtype(self):
        return [('start_time', 'S32', 'time_elapse', 'i8')]

    @property
    def metadata(self):
        return np.array([[self.start_time, self.time_elapse_s]], dtype=self.meta_dtype)

    def package(self):
        self.dataset = package_simulations_into_experiment(
            'data.h5', self.metadata, [s.dataset for s in self.SCPs]
        )

    def execute(self):
        tasks = [dask.delayed(s.execute)() for s in self.SCPs]
        start = time.perf_counter()
        dask.compute(*tasks)
        end = time.perf_counter()
        self.time_elapse_s = int(end - start)
        self.package()
