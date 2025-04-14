import matplotlib.pyplot as plt

import default
from art.art import Figure
from h5tools.dataset import *
from research.two_order_plot import two_order_scatter, two_order_stream


def _plot_hyperbola(ax, a: float = 1):
    theta = np.linspace(0, np.pi / 4, 400)
    r = 1 / np.sqrt(np.cos(2 * theta))
    x = a * r * np.cos(theta)
    y = a / np.sqrt(1 - a ** 2) * r * np.sin(theta)
    ax.plot(x, y, label=r'$y = \sqrt{x^2 - a^2}$', color='black', alpha=1, linewidth=1)
    x0 = np.linspace(0, 1, 400)
    ax.plot(x0, x0, color='grey', alpha=1, linewidth=1)


def plot_hyperbola(ax):
    plt.rcParams.update({'font.size': 20})
    _plot_hyperbola(ax, a=default.S_local_background)
    ax.set_aspect(1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.xlabel(r'$S_\text{local}$')
    plt.ylabel(r'$S_\text{global}$')
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)


if __name__ == '__main__':
    with Figure() as fig:
        # two_order_scatter(fig.ax, 'merge-full-0407.h5', 'S_local', 'S_global', alpha=0.04)
        two_order_stream(fig,'merge-analysis-0407.h5', 'S_local', 'S_global', alpha=0.8, interval=(0, 1))
        plot_hyperbola(fig.ax)
