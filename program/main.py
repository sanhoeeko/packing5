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
#         compress_func_B=boundary.RatioCompress(0.001)
#     )
#     ex.execute()
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from analysis.database import Database
    from analysis.analysis import CorrelationFunc

    db = Database('data.h5')
    # InteractiveViewer(db.simulation_at(0, 0), RenderSetup('S_local', False, 'default', True)).show()
    # plotListOfArray(db.simulation_at(0, 0).descent_curve)
    # for j in range(3):
    #     plt.plot(db.property('phi')[0, 0, :-1], db.simulation_at(0, j).stateDistance())
    # plt.show()
    # for j in range(3):
    #     plt.plot(db.property('phi')[0, 0, :], db.property('gradient_amp')[0, j, :])
    # plt.show()
    res = db.apply(CorrelationFunc('Phi6', 'S_local', False))
    print(res.shape)
    plt.plot(res[0, 0])
    plt.show()
