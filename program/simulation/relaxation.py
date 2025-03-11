from enum import Enum
from typing import Union

import numpy as np

import default
from . import stepsize as ss, utils as ut
from .lbfgs import LBFGS
from .mc import StatePool


class Criterion(Enum):
    NoCriterion = 0
    MeanGradientAmp = 1
    MaxGradientAmp = 2
    EnergyFlat = 3


class EnergyCounter:
    def __init__(self, threshold_slope: float, stride: int, occurrence: int):
        self.threshold_difference = threshold_slope * stride
        self.stride = stride
        self.energy_lst = []
        self.expected_occurrence = occurrence
        self.current_occurrence = 0

    def clear(self):
        self.energy_lst.clear()
        self.current_occurrence = 0

    def _judge_single(self, energy: float):
        if energy < default.energy_eps:
            return True
        return len(self.energy_lst) >= self.stride and (
                (1 - energy / self.energy_lst[-self.stride]) < self.threshold_difference
        )

    def judge(self, energy: float):
        self.energy_lst.append(energy)
        if self._judge_single(energy):
            self.current_occurrence += 1
        else:
            self.current_occurrence = 0
        return self.current_occurrence >= self.expected_occurrence


class DescentCurve:
    def __init__(self):
        self.clear()

    def clear(self):
        self.cnt = 0
        self.mean_gradient_curve = np.zeros((default.max_energy_curve_capacity,))
        self.max_gradient_curve = np.zeros((default.max_energy_curve_capacity,))
        self.energy_curve = np.zeros((default.max_energy_curve_capacity,))

    def append(self, mean_gradient_amp: float, max_gradient_amp: float, energy: float):
        self.mean_gradient_curve[self.cnt] = mean_gradient_amp
        self.max_gradient_curve[self.cnt] = max_gradient_amp
        self.energy_curve[self.cnt] = energy
        self.cnt += 1
        if self.cnt > default.max_energy_curve_capacity:
            raise IndexError("Too many energy records!")

    def get(self, length: int) -> (np.ndarray, np.ndarray, np.ndarray):
        def process_array(arr_full):
            arr = arr_full[:self.cnt]

            def recurse(arr):
                len_arr = len(arr)
                if len_arr < length:
                    res = np.full((length,), np.float32(np.nan))
                    res[:len_arr] = arr
                    return res
                elif length <= len_arr < 2 * length:
                    return arr[-length:]
                else:
                    step = len_arr // length
                    sampled_arr = arr[::step]
                    return recurse(sampled_arr)

            return recurse(arr)

        return (process_array(self.mean_gradient_curve), process_array(self.max_gradient_curve),
                process_array(self.energy_curve))


def Relaxation(
        noise_factor: float,
        momentum_beta: float,
        stochastic_p: float,
        inertia: float,
        stepsize: float,
        relaxation_steps: Union[int, float],
        state_pool_stride: Union[int, float],
        enable_lbfgs: bool,
        record_descent: bool,
        auto_stepsize: ss.StepsizeHelper,
        criterion: Criterion
):
    def inner(state):
        # Determine stepsize:
        if auto_stepsize is ss.StepsizeHelper.Nothing:
            def stepsize_provider(g: ut.CArray) -> float:
                return stepsize
        else:
            stepsizeFinder = ss.StepsizeHelperSwitch(auto_stepsize)
            if auto_stepsize is ss.StepsizeHelper.Armijo:
                assert enable_lbfgs, "Armijo step size is specially designed for LBFGS."
                if state.train and not hasattr(state, 'armijo_agent'):
                    setattr(state, 'armijo_agent', ss.ArmijoAgent(state))

                def stepsize_provider(g: ut.CArray) -> float:
                    return stepsize * stepsizeFinder(state.armijo_agent, state, g, 1e-3)
            else:
                if state.train and not hasattr(state, 'scanner'):
                    setattr(state, 'scanner', ss.EnergyScanner(state))

                def stepsize_provider(g: ut.CArray) -> float:
                    return stepsize * stepsizeFinder(state.scanner, state, g, 1e-3)

        # Determine whether refresh optimizer
        if stochastic_p != 1:
            def refresh_optimizer():
                state.optimizer.initMask(stochastic_p)
        else:
            def refresh_optimizer():
                pass

        # Simply calculate gradient or use LBFGS
        if enable_lbfgs:
            if state.train and not hasattr(state, 'lbfgs_agent'):
                setattr(state, 'lbfgs_agent', LBFGS(state))

            def getGradient() -> ut.CArray:
                d = state.lbfgs_agent.CalDirection()
                return d

            def getGradientAmp() -> np.float32:
                return state.lbfgs_agent.gradientAmp()
        else:
            def getGradient() -> ut.CArray:
                return state.calGradient()

            def getGradientAmp() -> np.float32:
                return state.optimizer.gradientAmp()

        # Termination criteria
        if criterion == Criterion.MeanGradientAmp:
            def judgeTermination() -> bool:
                return state.mean_gradient_amp < default.terminal_grad_for_mean_gradient_amp
        elif criterion == Criterion.MaxGradientAmp:
            def judgeTermination() -> bool:
                return state.max_gradient_amp < default.terminal_grad_for_max_gradient_amp
        elif criterion == Criterion.EnergyFlat:
            setattr(state, 'energy_counter', EnergyCounter(
                default.terminal_energy_slope, default.energy_counter_stride, default.energy_counter_occurrence))

            def judgeTermination() -> bool:
                return state.mean_gradient_amp < 1e-6 or state.energy_counter.judge(state.energy)
        else:
            def judgeTermination() -> bool:
                return False

        # Method to record descent curve
        if not record_descent:
            def record(t):
                pass
        else:
            if default.if_cal_energy:
                def record(t):
                    if t % default.descent_curve_stride == 0:
                        state.descent_curve.append(state.mean_gradient_amp, state.max_gradient_amp, state.energy)
            else:
                def record(t):
                    if t % default.descent_curve_stride == 0:
                        state.descent_curve.append(state.mean_gradient_amp, state.max_gradient_amp, np.nan)

        # Clear something to prepare for relaxation
        optimizer_energy_option = default.if_cal_energy and record_descent

        def prepare_relaxation():
            state.setOptimizer(noise_factor, momentum_beta, stochastic_p, inertia, False, optimizer_energy_option)
            if criterion == Criterion.EnergyFlat:
                state.energy_counter.clear()
            if enable_lbfgs:
                state.lbfgs_agent.init()
            if hasattr(state, 'scanner'):
                state.scanner.state.setPotential(state.gradient.potential)
            if hasattr(state, 'armijo_agent'):
                state.armijo_agent.state.setPotential(state.gradient.potential)

        # If state pools are not applied
        if state_pool_stride == 1:
            def relax():
                prepare_relaxation()
                for t in range(int(relaxation_steps)):
                    refresh_optimizer()
                    gradient = getGradient()
                    sz = stepsize_provider(gradient)
                    state.descent(gradient, sz)
                    if enable_lbfgs: state.lbfgs_agent.update()
                    record(t)
                    if_terminate = judgeTermination()
                    state.clear_dependency()
                    if if_terminate: break
        else:
            # If state pools are applied
            def relax():
                prepare_relaxation()
                for t in range(int(relaxation_steps) // state_pool_stride):
                    relax.state_pool.clear()
                    refresh_optimizer()
                    sz = stepsize_provider(getGradient())
                    for i in range(state_pool_stride):
                        gradient = getGradient()
                        if state.optimizer.particles_too_close_cache or state.isOutOfBoundary():
                            relax.state_pool.add(state, 1e5)
                        else:
                            relax.state_pool.add(state, getGradientAmp())
                        state.descent(gradient, sz)
                        if enable_lbfgs: state.lbfgs_agent.update()
                        state.clear_dependency()
                    energy, min_state = relax.state_pool.average_zero_temperature()
                    state.xyt.set_data(min_state.data)
                    record(t)
                    if judgeTermination(): break

            # Each state pool is managed by function but not State to avoid conflict
            relax.state_pool = StatePool(state.N, state_pool_stride)

        # Information: computational cost
        relax.n_steps = relaxation_steps
        return relax

    return inner
