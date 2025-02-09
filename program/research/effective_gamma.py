import matplotlib.pyplot as plt
import numpy as np

from analysis.analysis import *
from analysis.database import PickledSimulation
from analysis.voronoi import Voronoi
from h5tools.dataset import *


def scanGamma(state: dict) -> (np.ndarray, np.ndarray):
    """
    function act on a single configuration.
    :return: (the best gamma, maximum EllipticPhi6)
    """
    voro = Voronoi.fromStateDict(state).delaunay(False)
    gs = np.linspace(1, 2, 101)
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
    gs, ys = scanGamma(state)
    index = np.argmax(ys)
    return gs[index], ys[index]


def Phi6(state: dict) -> float:
    voro = Voronoi.fromStateDict(state).delaunay(False)
    if voro is None:
        return np.float32(np.nan)
    return np.mean(voro.Phi6(ut.CArray(state['xyt'])))


def BestGamma(simulation: PickledSimulation):
    gs = np.zeros((len(simulation)))
    Phis = np.zeros_like(gs)
    PhiEs = np.zeros_like(gs)
    for i, state in enumerate(simulation):
        gs[i], PhiEs[i] = scanBestGamma(state)
        Phis[i] = Phi6(state)
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
    yss = np.array(yss)
    plt.plot(yss)
    plt.show()


if __name__ == '__main__':
    auto_pack()
    db = Database('data.h5')
    # BestGamma(db[0].simulation_at(0))
    GammaLandscape(db[0].simulation_at(0))
