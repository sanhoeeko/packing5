import matplotlib.pyplot as plt

from analysis.post_analysis import MeanCIDatabase
from art import curves as art


def batch_analyze(filename: str, order_parameter_name: str, from_to: tuple):
    db = MeanCIDatabase(filename)
    x, y = zip(*db.orderParameterList(order_parameter_name))
    mean, ci = zip(*y)
    gammas = db.summary['gamma']
    f, t = from_to
    art.plotMeanCurvesWithCI(x[f:t], mean[f:t], ci[f:t], gammas[f:t],
                             x_label='volume fraction', y_label=order_parameter_name)
    plt.show()


if __name__ == '__main__':
    batch_analyze('../analysis-20250328.h5', 'EllipticPhi6', (0, 20))
