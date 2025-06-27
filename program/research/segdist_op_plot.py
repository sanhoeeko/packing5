import pickle as pkl

import numpy as np
from matplotlib import pyplot as plt

import analysis.utils as ut
from analysis.database import Database
from analysis.orders import StaticOrders


def Calculation(filenames: list[str], save=False, test=True, output_name='data', horizontal_axis: np.ndarray = None):
    """
    :param calculation: calculation(simu: PickledSimulation) -> np.ndarray
    :param save: True => save a pickle | False => visualization right now
    :param test: True => small amount of data | False => all data
    :param aggregate_method: sum | average
    """
    op_gamma = []
    gammas = [1.2] if test else np.arange(1.1, 3, 0.1)
    ensembles_per_file = 1 if test else 5
    for gamma in gammas:
        ops = []
        for filename in filenames:
            db = Database(filename)
            e = db.find(gamma=gamma)[0]
            N = int(e.metadata['N'])
            field_data = np.zeros((N * e.n_density * ensembles_per_file, 2))
            for j in range(ensembles_per_file):
                for k in range(e.n_density):
                    if e[j][k]['metadata']['phi'] > 0.84: break
                    xyt_c = ut.CArray(e[j].xyt[k])
                    voro = e[j].delaunayAt(k)
                    start_idx = (j * e.n_density + k) * N
                    field_data[start_idx: start_idx + N, 0] = voro.FarthestSegmentDist(xyt_c)
                    # field_data[start_idx: start_idx + N, 1] = voro.EllipticPhi6(xyt_c, e[j].metadata['gamma'])
                    field_data[start_idx: start_idx + N, 1] = voro.S_local(xyt_c)
            ops.append(field_data)
        op_gamma.append(np.vstack(ops))
    if save:
        with open(f'{output_name}.pkl', 'wb') as f:
            pkl.dump(op_gamma, f)
    else:
        arr = ut.clipArray(op_gamma[0])
        plt.scatter(arr[:, 0], arr[:, 1], s=1, alpha=0.1)
        plt.show()


if __name__ == '__main__':
    Calculation(['../data-20250420-2.h5'], save=False, test=True)
