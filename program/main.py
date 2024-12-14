# if __name__ == '__main__':
#     from experiment import Experiment
#     from simulation import boundary
#     ex = Experiment(
#         replica=3,
#         N=1024,
#         n=6,
#         d=0.05,
#         Gamma0=1,
#         phi0=0.6,
#         compress_func_A=boundary.NoCompress(),
#         compress_func_B=boundary.RatioCompress(0.002)
#     )
#     ex.execute()


if __name__ == '__main__':
    from analysis.database import Database
    import matplotlib.pyplot as plt
    from art.viewer import RenderSetup, InteractiveViewer
    from art.art import plotListOfArray

    db = Database('data.h5')
    InteractiveViewer(db.simulation_at(0, 1), RenderSetup('S_local', False, 'default', True)).show()
    # plotListOfArray(db.simulation_at(0, 1).normalizedDescentCurve())
    # for j in range(3):
    #     plt.plot(db.property('phi')[0, 0, :-1], db.simulation_at(0, j).stateDistance())
    # plt.show()

# if __name__ == '__main__':
#     from analysis.database import Database
#     from simulation.stepsize import energyScan
#     import matplotlib.pyplot as plt
#     from simulation.kernel import ker
#     from simulation.potential import Potential, PowerFunc
#     import numpy as np
#     from simulation.utils import CArray
#
#     db = Database('data.h5')
#     s = db.simulation_at(0, 1).state_at(400).train()
#     s.setOptimizer(0, 0, 1, False)
#     s.setPotential(Potential(s.n, s.d, PowerFunc(2.5)).cal_potential(4))
#     gradient = s.optimizer.calGradient()
#     g = ker.dll.FastNorm(gradient.ptr, s.N * 4) / np.sqrt(s.N)
#     grad = CArray(gradient.data / g)
#     xs, ys = energyScan(s, grad, 1e-3, 32)
#     plt.scatter(xs, ys)
#     plt.show()
