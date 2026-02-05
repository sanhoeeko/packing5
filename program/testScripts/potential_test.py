import time

import default
from testScripts.kernel_for_test import TestPotential, setWorkingDirectory

setWorkingDirectory()

from math import pi

import matplotlib.pyplot as plt
import numpy as np

from simulation.potential import RodPotential, PowerFunc


class TestResult:
    def __init__(self, args: tuple, dif: np.ndarray, mean_value: float):
        self.args = args
        self.dif = dif
        clean_dif = dif[~(np.isnan(dif) | np.isinf(dif))]
        self.mean_e = np.mean(clean_dif)
        self.median_e = np.median(clean_dif)
        self.max_e = np.max(clean_dif)
        self.mean_e_ratio = self.mean_e / mean_value
        self.median_e_ratio = self.median_e / mean_value
        self.max_e_ratio = self.max_e / mean_value

    def __repr__(self):
        return (f"Ratios: Median relative error: {self.median_e_ratio}, "
                f"Mean relative error: {self.mean_e_ratio}, Max relatve error: {self.max_e_ratio}")

    def show(self, expr=None, *num_list_of_args):
        if expr is None:
            plt.scatter(self.args[num_list_of_args[0]], self.dif)
        else:
            x = np.vectorize(expr)(*[self.args[i] for i in num_list_of_args])
            plt.scatter(x, self.dif, s=1)


def potentialTest(n: int, d: float, m: int, check_zero=False):
    """
    Test if the interpolated potential agrees with the directly calculated potential.
    m: number of sample points
    """
    a, b = 1, 1 / (1 + (n - 1) * d / 2)
    x = np.random.uniform(-2 * a, 2 * a, (m,))
    y = np.random.uniform(-a - b, a + b, (m,))
    t1 = np.random.uniform(0, 2 * pi, (m,))
    t2 = np.random.uniform(0, 2 * pi, (m,))
    potential = TestPotential(RodPotential(n, d, PowerFunc(2.5)).cal_potential(threads=4))

    d = np.vectorize(potential.segDist)(x, y, t1, t2)
    mask = d >= 2 - default.h_max
    x, y, t1, t2 = x[mask], y[mask], t1[mask], t2[mask]

    v = np.vectorize(potential.interpolatePotential)(x, y, t1, t2)
    v_ref = np.vectorize(potential.precisePotential)(x, y, t1, t2)
    if not check_zero:
        mask = v_ref > 1e-6
        v, v_ref = v[mask], v_ref[mask]
        x, y, t1, t2 = x[mask], y[mask], t1[mask], t2[mask]

    print(f"mean potential: {np.mean(v_ref)}")
    dif = np.abs(v - v_ref)
    # dif = np.abs(v - v_ref) / v_ref
    return TestResult((x, y, t1, t2), dif, np.mean(v_ref))


def gradientTest(n: int, d: float, m: int, check_zero=False):
    """
    Test if the interpolated gradient agrees with the directly calculated gradient.
    m: number of sample points
    """
    a, b = 1, 1 / (1 + (n - 1) * d / 2)
    x = np.random.uniform(-2 * a, 2 * a, (m,))
    y = np.random.uniform(-a - b, a + b, (m,))
    t1 = np.random.uniform(0, 2 * pi, (m,))
    t2 = np.random.uniform(0, 2 * pi, (m,))
    potential = TestPotential(RodPotential(n, d, PowerFunc(2.5)).cal_potential(threads=4))

    d = np.vectorize(potential.segDist)(x, y, t1, t2)
    mask = d >= 2 - default.h_max
    x, y, t1, t2 = x[mask], y[mask], t1[mask], t2[mask]
    m = x.size

    g = np.zeros((m, 6))
    g_ref = np.zeros((m, 6))

    # for i in range(m):
    #     try:
    #         g[i] = potential.interpolateGradient(x[i], y[i], t1[i], t2[i])
    #         g_ref[i] = potential.preciseGradient(x[i], y[i], t1[i], t2[i])
    #     except:
    #         g_should = potential.preciseGradient(x[i], y[i], t1[i], t2[i])
    #         print(f"(x={x[i]}, y={y[i]}, t1={t1[i]}, t2={t2[i]}) should be {g_should}")

    time_start = time.perf_counter()
    for i in range(m):
        g[i] = potential.interpolateGradient(x[i], y[i], t1[i], t2[i])
    time_end = time.perf_counter()
    time_per_case = (time_end - time_start) / m
    print(f"Gradient test: time elapsed: {(time_per_case * 10 ** 6):.3f} us, {m} cases")

    for i in range(m):
        g_ref[i] = potential.preciseGradient(x[i], y[i], t1[i], t2[i])

    r = g.reshape(-1).dot(g_ref.reshape(-1)) / g.reshape(-1).dot(g.reshape(-1))
    # there are bad cases, do not use np.mean
    print(f"regressive ratio: {r}")
    g_abs = np.sqrt(np.sum(g_ref ** 2, axis=1))
    print(f"mean amplitude of gradients: {np.mean(g_abs)}")
    g = r * g
    if not check_zero:
        mask = np.sqrt(np.sum(g_ref ** 2, axis=1)) > 1e-6
        g, g_ref = g[mask, :], g_ref[mask, :]
        x, y, t1, t2 = x[mask], y[mask], t1[mask], t2[mask]

    dif = np.sqrt(np.sum((g - g_ref) ** 2 / 6, axis=1))
    return TestResult((x, y, t1, t2), dif, np.mean(g_abs))


if __name__ == '__main__':
    res = gradientTest(5, 0.05, 1000000)
    print(res)
    res.show(lambda x, y: np.sqrt(x ** 2 + y ** 2), 0, 1)
    plt.show()
