from threading import Thread

import defaults
import simulation.utils as ut
from simulation.potential import RadialFunc, Potential, PowerFunc
from simulator import Simulator


class SimulationsCommonParticle:
    def __init__(self, n: int, d: float, scalar_func: RadialFunc):
        self.n, self.d = n, d
        self.simulators = []
        self.cal_potential = lambda threads: Potential(n, d, scalar_func, threads)

    @property
    def potential_tag(self):
        return self.n, self.d

    def append(self, simulator: Simulator):
        assert simulator.state.n == self.n
        assert simulator.state.d == self.d
        self.simulators.append(simulator)

    def execute(self):
        # calculate and link to the potential
        self.potential = self.cal_potential(len(self.simulators))
        for simulator in self.simulators:
            simulator.state.potential = self.potential

        # thread allocation
        self.threads = []
        for s in self.simulators:
            thread = Thread(target=s.execute)
            self.threads.append(thread)
            thread.start()
        for thread in self.threads:
            thread.join()


def createSimulator(N, n, d, phi0, Gamma0, compress_func_A, compress_func_B):
    return (Simulator.fromPackingFractionPhi(N, n, d, phi0, Gamma0, None)
            .setCompressMethod(compress_func_A, compress_func_B, defaults.max_compress)
            .schedule(defaults.max_relaxation, defaults.descent_curve_stride)
            )


class Experiment:
    def __init__(self, **kwargs):
        self.SCPs = []
        simulators = ut.ClassCartesian(Simulator, **kwargs)
        tags = list(map(lambda x: x.potential_tag, simulators))
        radial_func = PowerFunc(2.5)
        for n, d in tags:
            self.SCPs.append(SimulationsCommonParticle(n, d, radial_func))
        for s in simulators:
            for scp in self.SCPs:
                if scp.potential_tag == s.potential_tag:
                    scp.append(s)
                    break

    def execute(self):
        pass
