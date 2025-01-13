import platform

import simulation.utils as ut
from h5tools.utils import randomString
from simulation import boundary
from simulation.potential import Potential, PowerFunc
from simulation.simulator import createSimulator


def testSingleThread(profile=True):
    N = 200
    n = 5
    d = 0.025
    phi0 = 0.7
    Gamma0 = 1
    compress_func_A = boundary.NoCompress()
    compress_func_B = boundary.RatioCompress(0.001)
    ex = createSimulator(f'{randomString()}_0', N, n, d, phi0, Gamma0, compress_func_A, compress_func_B)
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


if __name__ == '__main__':
    print("Current python version:", platform.python_version())
    testSingleThread(True)
