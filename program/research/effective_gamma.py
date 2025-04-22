import matplotlib.pyplot as plt

from analysis.analysis import *
from analysis.database import PickledSimulation
from analysis.voronoi import Voronoi
from h5tools.cubic import CubicMinimumXNan
from h5tools.dataset import *


def scanGamma(state: dict, steps: int) -> (np.ndarray, np.ndarray):
    """
    function act on a single configuration.
    :return: (the best gamma, maximum EllipticPhi6)
    """
    voro = Voronoi.fromStateDict(state).delaunay()
    gs = np.linspace(1, 2, steps)
    ys = np.zeros_like(gs)
    if voro is None:
        return np.array([np.float32(np.nan)] * len(gs)), np.array([np.float32(np.nan)] * len(gs))
    for i in range(len(gs)):
        ys[i] = np.mean(voro.EllipticPhi6(ut.CArray(state['xyt']), gs[i]))
    return gs, ys


def scanBestGamma(state: dict) -> (float, float):
    """
    function act on a single configuration.
    :return: (the best gamma, maximum EllipticPhi6)
    """
    gs, ys = scanGamma(state, 101)
    index = np.argmax(ys)
    return gs[index], ys[index]


def cubicBestGamma(state: dict) -> (float, float):
    """
    function act on a single configuration.
    :return: (the best gamma, maximum EllipticPhi6)
    """
    gs, ys = scanGamma(state, 21)
    g_min = CubicMinimumXNan(gs, -ys, 1, 2)  # cubic maximum
    if np.isnan(g_min):
        return np.float32(np.nan), np.float32(np.nan)
    voro = Voronoi.fromStateDict(state).delaunay()
    y_practical = np.mean(voro.EllipticPhi6(ut.CArray(state['xyt']), g_min))
    return g_min, y_practical


def BestGamma(simulation: PickledSimulation):
    gs = np.zeros((len(simulation)))
    PhiEs = np.zeros_like(gs)
    for i, state in enumerate(simulation):
        gs[i], PhiEs[i] = cubicBestGamma(state)
    Phis = simulation.op('EllipticPhi6')
    plt.plot(gs)
    plt.plot(PhiEs)
    plt.plot(Phis)
    plt.show()


def GammaLandscape(simulation: PickledSimulation):
    gs = None
    yss = []
    for i, state in enumerate(simulation):
        gs, ys = scanGamma(state)
        yss.append(ys)
    yss = np.array(yss).T
    plt.plot(yss)
    plt.show()


def StarGamma(simulation: PickledSimulation):
    PhiEs = np.zeros((len(simulation)))
    for i, state in enumerate(simulation):
        voro = Voronoi.fromStateDict(state).delaunay()
        PhiEs[i] = np.mean(voro.EllipticPhi6(ut.CArray(state['xyt']), ut.gamma_star(state['metadata']['gamma'])))
    Phis = simulation.op('EllipticPhi6')
    plt.plot(PhiEs)
    plt.plot(Phis)
    plt.show()


if __name__ == '__main__':
    ut.setWorkingDirectory()
    db = Database('../data-20250406.h5')
    e = db.find(gamma=1.1)[0]
    StarGamma(e.simulation_at(0))
    # GammaLandscape(db[0].simulation_at(0))
