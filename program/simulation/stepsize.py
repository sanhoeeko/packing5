import numpy as np

from h5tools.cubic import CubicMinimumX
from . import utils as ut
from .kernel import ker

alpha = np.float32(0.96)
beta = np.float32(0.6)


def energyScan(s, g: ut.CArray, step_sizes: np.ndarray):
    """
    :param s: State
    """
    state = s.copy(train=True)
    ys = np.zeros_like(step_sizes)
    for i in range(len(step_sizes)):
        ker.dll.AddVector4(s.xyt.ptr, g.ptr, state.xyt.ptr, state.N, step_sizes[i])
        if state.isOutOfBoundary():
            ys[i] = np.inf
        else:
            ys[i] = state.CalEnergy_pure()
    return ys


def findBestStepsize(s, max_stepsize: float, n_samples: int) -> np.float32:
    """
    :param s: State
    """
    normalized_gradient = s.CalGradientNormalized_pure(1)
    if normalized_gradient is None:
        return max_stepsize
    xs = max_stepsize * alpha ** np.arange(n_samples)
    ys = energyScan(s, normalized_gradient, xs)
    index = np.argmin(ys)
    return np.float32(xs[index])


def findGoodStepsize(s, max_stepsize: float, n_samples: int) -> np.float32:
    """
    :param s: State
    """
    normalized_gradient = s.CalGradientNormalized_pure(1)
    if normalized_gradient is None:
        return max_stepsize
    state = s.copy(train=True)
    stepsize = np.float32(max_stepsize)
    energy = state.CalEnergy_pure()
    for i in range(n_samples):
        current_stepsize = stepsize * alpha
        ker.dll.AddVector4(s.xyt.ptr, normalized_gradient.ptr, state.xyt.ptr, state.N, current_stepsize)
        if state.isOutOfBoundary():
            continue
        current_energy = state.CalEnergy_pure()
        if current_energy > energy:
            return stepsize
        stepsize = current_stepsize
        energy = current_energy
    return stepsize


def findCubicStepsize(s, max_stepsize: float, n_samples: int) -> np.float32:
    """
    :param s: State
    """
    normalized_gradient = s.CalGradientNormalized_pure(1)
    if normalized_gradient is None:
        return max_stepsize
    xs = max_stepsize * beta ** np.arange(n_samples)
    ys = energyScan(s, normalized_gradient, xs)
    return CubicMinimumX(xs, ys, 0, max_stepsize)
