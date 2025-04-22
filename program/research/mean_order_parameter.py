import matplotlib.pyplot as plt

from analysis.post_analysis import MeanCIDatabase
from art import curves as art


def batch_analyze(filename: str, order_parameter_name: str, from_to: tuple):
    db = MeanCIDatabase(filename)
    # db.to_csv('EllipticPhi6', 'ephi6.csv')
    dic = db.extract_data(order_parameter_name, truncate_h=1.2)
    f, t = from_to
    art.plotMeanCurvesWithCI(dic[db.x_axis_name][f:t], dic['mean'][f:t], dic['ci'][f:t], dic['gammas'][f:t],
                             x_label='volume fraction', y_label=order_parameter_name)
    plt.show()


if __name__ == '__main__':
    batch_analyze('merge-analysis-0420.h5', 'EllipticPhi6', (0, 20))
