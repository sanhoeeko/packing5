from enum import Enum
from typing import Union

import numpy as np

import default
from . import stepsize as ss
from .mc import StatePool


class Criterion(Enum):
    NoCriterion = 0
    MeanGradientAmp = 1
    MaxGradientAmp = 2
    EnergyFlat = 3


class EnergyCounter:
    def __init__(self, threshold_slope: float, stride: int, occurrence: int):
        self.threshold_difference = threshold_slope / stride
        self.stride = stride
        self.energy_lst = []
        self.expected_occurrence = occurrence
        self.current_occurrence = 0

    def _judge_single(self, energy: float):
        return len(self.energy_lst) >= self.stride and (
                (self.energy_lst[-self.stride] - energy) < self.threshold_difference
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
        def process_array(arr):
            if self.cnt < length:
                res = np.full((length,), np.float32(np.nan))
                res[:self.cnt] = arr[:self.cnt]
                return res
            elif length <= self.cnt < 2 * length:
                return arr[-length:]
            else:
                step = self.cnt // length
                sampled_arr = arr[::step]
                return process_array(sampled_arr)

        return (process_array(self.mean_gradient_curve), process_array(self.max_gradient_curve),
                process_array(self.energy_curve))


def Relaxation(
        noise_factor: float,
        momentum_beta: float,
        stochastic_p: float,
        stepsize: float,
        relaxation_steps: Union[int, float],
        state_pool_stride: Union[int, float],
        record_stride: Union[int, float],
        auto_stepsize: bool,
        record_energy: bool,
        criterion: Criterion
):
    def inner(state):
        # Determine stepsize:
        if auto_stepsize:
            def stepsize_provider() -> float:
                return stepsize * ss.findCubicStepsize(state, 1e-2, 6)
        else:
            def stepsize_provider() -> float:
                return stepsize

        # Termination criteria
        if criterion == Criterion.MeanGradientAmp:
            def judgeTermination() -> bool:
                return state.optimizer.gradientAmp() < default.terminal_grad_for_mean_gradient_amp
        elif criterion == Criterion.MaxGradientAmp:
            def judgeTermination() -> bool:
                return state.optimizer.maxGradient() < default.terminal_grad_for_max_gradient_amp
        elif criterion == Criterion.EnergyFlat:
            setattr(state, 'energy_counter', EnergyCounter(
                default.terminal_energy_slope, default.energy_counter_stride, default.energy_counter_occurrence))

            def judgeTermination() -> bool:
                energy = state.CalEnergy_pure()
                return state.energy_counter.judge(energy)
        else:
            def judgeTermination() -> bool:
                return False

        # Method to record descent curve
        if record_energy:
            def record(t):
                if t % record_stride == 0:
                    state.descent_curve.append(state.mean_gradient_amp, state.max_gradient_amp, state.energy)
        else:
            def record(t):
                if t % record_stride == 0:
                    state.descent_curve.append(state.mean_gradient_amp, state.max_gradient_amp, np.nan)

        # If state pools are not applied
        if state_pool_stride == 1:
            def relax():
                state.setOptimizer(noise_factor, momentum_beta, stochastic_p, False)
                sz = stepsize_provider()
                for t in range(int(relaxation_steps)):
                    gradient = state.optimizer.calGradient()
                    state.descent(gradient, sz)
                    record(t)
                    if judgeTermination(): break
        else:
            # If state pools are applied
            def relax():
                state.setOptimizer(noise_factor, momentum_beta, stochastic_p, False)
                for t in range(int(relaxation_steps) // state_pool_stride):
                    state.state_pool.clear()
                    sz = stepsize_provider()
                    for i in range(state_pool_stride):
                        gradient = state.optimizer.calGradient()
                        if state.optimizer.particles_too_close_cache or state.isOutOfBoundary():
                            state.state_pool.add(state, 1e5)
                            state.descent(gradient, sz)
                        else:
                            g = state.optimizer.gradientAmp()
                            state.state_pool.add(state, g)
                            state.descent(gradient, sz)
                    energy, min_state = state.state_pool.average_zero_temperature()
                    state.xyt.set_data(min_state.data)
                    record(t)
                    if judgeTermination(): break
                state.descent_curve.join()

            # Each state pool is managed by function but not State to avoid conflict
            relax.state_pool = StatePool(state.N, state_pool_stride)

        # Information: computational cost
        relax.n_steps = relaxation_steps
        return relax

    return inner
