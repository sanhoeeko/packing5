from enum import IntEnum
from typing import Any, Callable

import numpy as np

from h5tools.cubic import CubicMinimumX
from . import utils as ut
from .kernel import ker


class StepsizeHelper(IntEnum):
    Nothing = 0
    Best = 1
    Good = 2
    Cubic = 3
    Armijo = 4


def StepsizeHelperSwitch(helper: StepsizeHelper) -> Callable[[Any, ut.CArray, float], float]:
    funcs = [None, findBestStepsize, findGoodStepsize, findCubicStepsize]
    return funcs[helper]


alpha = np.float32(0.96)
beta = np.float32(0.4)
n_samples = 32
cubic_samples = 8


def energyScan(s, g: ut.CArray, step_sizes: np.ndarray, need_energy=False):
    """
    :param s: State
    """
    state = s.copy(train=False)
    ys = np.zeros_like(step_sizes)
    for i in range(len(step_sizes)):
        ker.dll.AddVector4(s.xyt.ptr, g.ptr, state.xyt.ptr, state.N, step_sizes[i])
        if state.isOutOfBoundary():
            ys[i] = np.inf
        else:
            ys[i] = state.CalEnergy_pure() if need_energy else state.mean_gradient_amp
    return ys


def findBestStepsize(s, g: ut.CArray, max_stepsize: float) -> np.float32:
    """
    :param s: State
    """
    absg = g.norm(g.data.shape[0])
    if absg < 1e-6: return max_stepsize
    normalized_gradient = ut.CArray(g.data / absg)

    xs = max_stepsize * alpha ** np.arange(n_samples)
    ys = energyScan(s, normalized_gradient, xs)
    index = np.argmin(ys)
    return np.float32(xs[index])


def findGoodStepsize(s, g: ut.CArray, max_stepsize: float) -> np.float32:
    """
    :param s: State
    """
    absg = g.norm(g.data.shape[0])
    if absg < 1e-6: return max_stepsize
    normalized_gradient = ut.CArray(g.data / absg)

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


def findCubicStepsize(s, g: ut.CArray, max_stepsize: float) -> np.float32:
    """
    :param s: State
    :param g: non-normalized gradient from external scope
    """
    absg = g.norm(g.data.shape[0])
    if absg < 1e-6: return max_stepsize
    normalized_gradient = ut.CArray(g.data / absg)
    xs = max_stepsize * beta ** np.arange(cubic_samples)
    ys = energyScan(s, normalized_gradient, xs)
    return CubicMinimumX(xs, ys, 1e-6, max_stepsize)
