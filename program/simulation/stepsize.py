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


def StepsizeHelperSwitch(helper: StepsizeHelper) -> Callable[[Any, Any, ut.CArray, float], float]:
    funcs = [None, findBestStepsize, findGoodStepsize, findCubicStepsize]
    return funcs[helper]


alpha = np.float32(0.96)
beta = np.float32(0.4)
n_samples = 32
cubic_samples = 8


class EnergyScanner:
    def __init__(self, ref_state):
        # ref_state: State
        self.state = ref_state.copy(train=False)

    def scan(self, s, g: ut.CArray, step_sizes: np.ndarray, need_energy=False):
        # s: State
        ys = np.zeros_like(step_sizes)
        for i in range(len(step_sizes)):
            ker.dll.AddVector4(s.xyt.ptr, g.ptr, self.state.xyt.ptr, self.state.N, step_sizes[i])
            if self.state.isOutOfBoundary():
                ys[i] = np.inf
            else:
                ys[i] = self.state.CalEnergy_pure() if need_energy else self.state.mean_gradient_amp
            self.state.clear_dependency()
        return ys

    def good(self, s, g: ut.CArray, max_stepsize: float, need_energy=False):
        # s: State
        stepsize = np.float32(max_stepsize)
        energy = s.CalEnergy_pure() if need_energy else s.mean_gradient_amp
        for i in range(n_samples):
            current_stepsize = stepsize * alpha
            ker.dll.AddVector4(s.xyt.ptr, g.ptr, self.state.xyt.ptr, self.state.N, current_stepsize)
            if self.state.isOutOfBoundary():
                continue
            current_energy = self.state.CalEnergy_pure() if need_energy else self.state.mean_gradient_amp
            if current_energy > energy:
                return stepsize
            stepsize = current_stepsize
            energy = current_energy
            self.state.clear_dependency()
        return stepsize


def findBestStepsize(scanner: EnergyScanner, s, g: ut.CArray, max_stepsize: float) -> np.float32:
    """
    :param s: State
    """
    absg = g.norm(g.data.shape[0])
    if absg < 1e-6: return max_stepsize
    normalized_gradient = ut.CArray(g.data / absg)

    xs = max_stepsize * alpha ** np.arange(n_samples)
    ys = scanner.scan(s, normalized_gradient, xs)
    index = np.argmin(ys)
    return np.float32(xs[index])


def findGoodStepsize(scanner: EnergyScanner, s, g: ut.CArray, max_stepsize: float) -> np.float32:
    """
    :param s: State
    """
    absg = g.norm(g.data.shape[0])
    if absg < 1e-6: return max_stepsize
    normalized_gradient = ut.CArray(g.data / absg)
    return scanner.good(s, normalized_gradient, max_stepsize)


def findCubicStepsize(scanner: EnergyScanner, s, g: ut.CArray, max_stepsize: float) -> np.float32:
    """
    :param s: State
    :param g: non-normalized gradient from external scope
    """
    absg = g.norm(g.data.shape[0])
    if absg < 1e-6: return max_stepsize
    normalized_gradient = ut.CArray(g.data / absg)
    xs = max_stepsize * beta ** np.arange(cubic_samples)
    ys = scanner.scan(s, normalized_gradient, xs)
    return CubicMinimumX(xs, ys, 1e-6, max_stepsize)


class ArmijoAgent:
    def __init__(self, ref_state):
        # ref_state: State
        self.state = ref_state.copy(train=False)
        self.c1 = 1e-4
        self.beta = 0.5

    def armijo(self, s, g: ut.CArray, direction_product: float):
        alpha = 1.0
        current_loss = s.CalEnergy_pure()
        for ls in range(20):
            ker.dll.AddVector4(s.xyt.ptr, g.ptr, self.state.xyt.ptr, self.state.N, alpha)
            fx_new = self.state.CalEnergy_pure()
            if fx_new <= current_loss + self.c1 * alpha * direction_product:
                break
            alpha *= self.beta
        return alpha


def ArmijoStepsize(agent: ArmijoAgent, s, g: ut.CArray, max_stepsize: float) -> np.float32:
    """
    Especially for LBFGS
    """
    direction_product = s.lbfgs_agent.directionProduct()
    return agent.armijo(s, g, direction_product)
