import platform

import simulation.potential as pot
import simulation.utils as ut
from h5tools.utils import randomString
from simulation import boundary
from simulation.simulator import createSimulator


def testSingleThread(profile=True):
    N = 1024
    n = 10
    d = 0.1
    phi0 = 0.7
    Gamma0 = 1
    compress_func_A = boundary.NoCompress()
    compress_func_B = boundary.RatioCompress(0.004)
    ex = createSimulator(f'{randomString()}_0', N, n, d, phi0, Gamma0, compress_func_A, compress_func_B)
    # ex.setPotential(pot.Potential(n, d, pot.PowerFunc(2.5)))
    # ex.setPotential(pot.RodPotential(n, d, pot.ModifiedPower(2.5, x0=1.0)))
    ex.setPotential(pot.SegmentPotential(1.5, pot.ModifiedPower(2.5, x0=1.0)))
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
