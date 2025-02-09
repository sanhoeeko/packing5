import matplotlib.pyplot as plt

from analysis.analysis import *
from analysis.database import PickledSimulation
from h5tools.dataset import *


def plot_hyperbola(ax, a: float = 1):
    theta = np.linspace(0, np.pi / 4, 400)
    r = 1 / np.sqrt(np.cos(2 * theta))
    x = a * r * np.cos(theta)
    y = a / np.sqrt(1 - a ** 2) * r * np.sin(theta)
    ax.plot(x, y, label=r'$y = \sqrt{x^2 - a^2}$', color='black',
            alpha=0.2, linewidth=2)


def scatter(S_local: np.ndarray, S_global: np.ndarray):
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots()
    ax.scatter(S_local, S_global, s=1)
    plot_hyperbola(ax, a=0.338)
    ax.set_aspect(1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.xlabel(r'$S_\text{local}$')
    plt.ylabel(r'$S_\text{global}$')
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    plt.show()


def S_local_vs_global(simu: PickledSimulation):
    S_local = simu.op('S_local')
    S_global = simu.op('S_global')
    scatter(S_local, S_global)


if __name__ == '__main__':
    auto_pack()
    db = Database('data.h5')
    S_local_vs_global(db[0].simulation_at(0))
