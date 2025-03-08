import numpy as np

from .art import Figure, ListColor01, add_energy_level_colorbar


def plotListOfArray(lst: np.ndarray, labels: tuple[str, str] = None, y_restriction: float = None):
    colors = ListColor01('cool', len(lst))
    with Figure() as f:
        for i in range(len(lst)):
            f.ax.plot(lst[i], color=colors[i], alpha=0.5)
        if labels is not None:
            f.labels(*labels)
        if y_restriction is not None:
            f.region(None, (0, y_restriction), False)


def plotMeanCurvesWithCI(x_lst: list[np.ndarray], y_mean_lst: list[np.ndarray], y_ci_lst: list[np.ndarray],
                         gammas: np.ndarray = None, gamma_label='gamma', x_label='', y_label=''):
    """
    x_lst can be None
    """
    colormap = 'jet'

    if x_lst is None:
        x_lst = [np.arange(len(y)) for y in y_mean_lst]
    assert len(x_lst) == len(y_mean_lst) == len(y_ci_lst)

    colors = ListColor01(colormap, len(y_mean_lst))
    with Figure() as f:
        for i, (x, y_mean, y_ci) in enumerate(zip(x_lst, y_mean_lst, y_ci_lst)):
            color = colors[i]
            f.ax.fill_between(x, y_mean - y_ci, y_mean + y_ci, color=color, alpha=0.2)
            f.ax.plot(x, y_mean, color=color)
        f.labels(x_label, y_label)
        if gammas is not None:
            add_energy_level_colorbar(f.ax, colormap, gammas, gamma_label)


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
