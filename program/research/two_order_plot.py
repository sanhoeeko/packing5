import matplotlib.pyplot as plt
import numpy as np

import default
from analysis.post_analysis import RawOrderDatabase, MeanCIDatabase
from analysis.utils import reference_phi
from art.art import ListColor01, add_energy_level_colorbar, Figure
from art.curves import arrow_plot


def two_order_scatter(ax, filename: str, order1: str, order2: str, alpha=0.1):
    db = RawOrderDatabase(filename)
    colormap = 'jet'
    colors = ListColor01(colormap, len(db))
    gammas = db.summary['gamma']
    for i, ensemble in enumerate(db):
        o1 = ensemble[order1].reshape(-1)
        o2 = ensemble[order2].reshape(-1)
        ax.scatter(o1, o2, s=1, alpha=alpha, c=colors[i])
    add_energy_level_colorbar(ax, colormap, gammas, 'gamma')


def two_order_stream(fig: Figure, filename: str, order1: str, order2: str, alpha=1.0, interval=(0, 1)):
    db = MeanCIDatabase(filename)
    colormap = 'jet'
    colors = ListColor01(colormap, len(db))
    gammas = db.summary['gamma']
    phi_max = reference_phi(gammas, default.h_max)
    for i, ensemble in enumerate(db):
        idx = np.where(ensemble.x_axis > phi_max[i])[0][0]
        mean_1, ci_1 = ensemble[order1]
        mean_1, ci_1 = mean_1[:idx], ci_1[:idx]
        mean_2, ci_2 = ensemble[order2]
        mean_2, ci_2 = mean_2[:idx], ci_2[:idx]
        arrow_plot(fig, ensemble.x_axis[:idx], mean_1, mean_2, arrow_size=0.015, n_arrows=8, alpha=alpha,
                   c=colors[i])
    fig.region(interval, interval)
    add_energy_level_colorbar(fig.ax, colormap, gammas, 'gamma')


if __name__ == '__main__':
    with Figure() as fig:
        two_order_stream(fig, 'merge-analysis-0407.h5', 'S_local', 'S_global', alpha=0.8, interval=(0, 1))
