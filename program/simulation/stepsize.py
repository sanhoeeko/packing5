import numpy as np

from . import utils as ut
from .kernel import ker
from .state import State

alpha = np.float32(1.414)


def energyScan(s: State, g: ut.CArray, max_stepsize: float, n_samples: int):
    state = s.copy(train=True)
    ratios = alpha ** np.arange(0, n_samples, 1)
    xs = max_stepsize / ratios
    ys = np.zeros_like(xs)
    for i in range(n_samples):
        ker.dll.AddVector4(s.xyt.ptr, g.ptr, state.xyt.ptr, state.N, xs[i])
        if state.isOutOfBoundary():
            ys[i] = np.inf
        else:
            ys[i] = state.CalEnergy_pure()
    return xs, ys


def findBestStepsize(s: State, max_stepsize: float, n_samples: int) -> np.float32:
    normalized_gradient = s.CalGradientNormalized_pure(0)
    xs, ys = energyScan(s, normalized_gradient, max_stepsize, n_samples)
    index = np.argmin(ys)
    return np.float32(xs[index])


def findGoodStepsize(s: State, max_stepsize: float, n_samples: int) -> np.float32:
    normalized_gradient = s.CalGradientNormalized_pure(0)
    state = s.copy(train=True)
    stepsize = np.float32(max_stepsize)
    energy = state.CalEnergy_pure()
    for i in range(n_samples):
        current_stepsize = stepsize / alpha
        ker.dll.AddVector4(s.xyt.ptr, normalized_gradient.ptr, state.xyt.ptr, state.N, current_stepsize)
        if state.isOutOfBoundary():
            continue
        current_energy = state.CalEnergy_pure()
        if current_energy > energy:
            return stepsize
        stepsize = current_stepsize
        energy = current_energy
    return stepsize
