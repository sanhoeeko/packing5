import matplotlib.pyplot as plt
import numpy as np

import default
from analysis.post_analysis import RawOrderDatabase, MeanCIDatabase
from art.art import ListColor01, add_energy_level_colorbar, Figure
from art.curves import arrow_plot
from analysis.utils import reference_phi


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


def two_order_stream(filename: str, order1: str, order2: str, alpha=1):
    db = MeanCIDatabase(filename)
    colormap = 'jet'
    colors = ListColor01(colormap, len(db))
    gammas = db.summary['gamma']
    phi_max = reference_phi(gammas, default.h_max)
    with Figure() as fig:
        for i, ensemble in enumerate(db):
            idx = np.where(ensemble.x_axis > phi_max[i])[0][0]
            mean_1, ci_1 = ensemble[order1]
            mean_1, ci_1 = mean_1[:idx], ci_1[:idx]
            mean_2, ci_2 = ensemble[order2]
            mean_2, ci_2 = mean_2[:idx], ci_2[:idx]
            arrow_plot(fig, ensemble.x_axis[:idx], mean_1, mean_2, arrow_size=0.02, n_arrows=10, alpha=alpha, c=colors[i])
        fig.ax.set_aspect(1)
        fig.ax.set_xlim(0, 1)
        fig.ax.set_ylim(0, 1)
    add_energy_level_colorbar(fig.ax, colormap, gammas, 'gamma')


if __name__ == '__main__':
    two_order_stream('merge-analysis-0407.h5', 'S_local', 'EllipticPhi6', alpha=0.5)
    plt.show()
