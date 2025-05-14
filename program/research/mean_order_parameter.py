import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np

from analysis.database import Database
from analysis.post_analysis import MeanCIDatabase
from art import curves as art


def read_op(filename: str, order_parameter_name: str, from_to: tuple):
    db = MeanCIDatabase(filename)
    # db.to_csv('defect', 'defect.csv')
    # db.to_csv('S_local', 'S_local.csv')
    dic = db.extract_data(order_parameter_name, truncate_h=1.2)
    f, t = from_to
    art.plotMeanCurvesWithCI(dic[db.x_axis_name][f:t], dic['mean'][f:t], dic['ci'][f:t], dic['gammas'][f:t],
                             x_label='volume fraction', y_label=order_parameter_name)
    plt.show()


def calculate_op(filenames: list[str], order_parameter_name: str, from_to: tuple):
    op_gamma = []
    for gamma in np.arange(1.1, 3, 0.2):
        ops = []
        for filename in filenames:
            db = Database(filename)
            e = db.find(gamma=gamma)[0]
            for j in range(5):
                simu = e[j]
                op_single = simu.op(order_parameter_name, upper_h=1.2, num_threads=4)
                ops.append(op_single)
        op = sum(ops) / len(ops)
        op_gamma.append(op)
    with open(f'{order_parameter_name}.pkl', 'wb') as f:
        pkl.dump(op_gamma, f)
    # plt.plot(e[0].state_info['phi'], op)
    # plt.show()


if __name__ == '__main__':
    read_op('../analysis-data-20250514.h5', 'S_local', (0, 20))
    # calculate_op(
    #     ['../data-20250420.h5', ],
    #     'isolatedDefect', (0, 20))
