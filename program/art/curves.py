import numpy as np

from .art import Figure, ListColor01


def plotListOfArray(lst: np.ndarray):
    colors = ListColor01('cool', len(lst))
    with Figure() as f:
        for i in range(len(lst)):
            f.ax.plot(lst[i], color=colors[i], alpha=0.5)


def plotMeanCurvesWithCI(x: np.ndarray, y_mean_lst: list[np.ndarray], y_ci_lst: list[np.ndarray],
                         x_label='', y_label=''):
    assert len(y_mean_lst) == len(y_ci_lst)
    colors = ListColor01('cool', len(y_mean_lst))
    with Figure() as f:
        for i, (y_mean, y_ci) in enumerate(zip(y_mean_lst, y_ci_lst)):
            color = colors[i]
            f.ax.fill_between(x, y_mean - y_ci, y_mean + y_ci, color=color, alpha=0.2)
            f.ax.plot(x, y_mean, color=color)
        f.labels(x_label, y_label)


def scatterCorrelations(x: np.ndarray, y: np.ndarray):
    """
    :param x: (samples, N) array
    :param y: (samples, N) array
    """
    assert x.shape == y.shape
    colors = ListColor01('cool', x.shape[0])
    with Figure() as f:
        for i in range(x.shape[0]):
            f.ax.scatter(x[i, :], y[i, :], color=colors[i], s=1, alpha=0.1)
        f.region([0, 1], [0, 1])