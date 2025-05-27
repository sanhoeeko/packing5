import numpy as np
from matplotlib import pyplot as plt

from analysis.database import Database

if __name__ == '__main__':
    op_gamma = []
    gammas = np.arange(1.1, 3, 0.9)
    ensembles_per_file = 5
    db = Database('../data-20250419.h5')
    for gamma in gammas:
        ops = []
        e = db.find(gamma=gamma)[0]
        for j in range(ensembles_per_file):
            simu = e[j]
            ops.append(simu.bondCreation(num_threads=4, upper_h=1.2))
        op = sum(ops) / len(ops)
        op_gamma.append(op)
    for op, gamma in zip(op_gamma, gammas):
        s = db.find(gamma=gamma)[0][0]
        plt.plot(s.propertyInterval('phi', upper_h=1.2)[1:], op)
    plt.show()
