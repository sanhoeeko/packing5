import numpy as np

from . import utils as ut
from .kernel import ker
from .state import State


def energyScan(s: State, g: ut.CArray, max_stepsize: float, n_samples: int):
    state = s.copy(train=True)
    ratios = 1.2 ** np.arange(0, n_samples, 1)
    xs = max_stepsize / ratios
    ys = np.zeros_like(xs)
    for i in range(n_samples):
        state.xyt.set_data(s.xyt.data + xs[i] * g.data)
        ys[i] = state.calEnergy()
    return xs, ys


def findBestStepsize(s: State,  max_stepsize: float, n_samples: int) -> np.float32:
    gradient = s.optimizer.calGradient()
    g = ker.dll.FastNorm(gradient.ptr, s.N * 4) / np.sqrt(s.N)
    normalized_gradient = ut.CArray(gradient.data / g)
    xs, ys = energyScan(s, normalized_gradient, max_stepsize, n_samples)
    index = np.argmin(ys)
    return np.float32(xs[index])
