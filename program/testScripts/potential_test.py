from testScripts.kernel_for_test import TestPotential, setWorkingDirectory

setWorkingDirectory()

from math import pi

import matplotlib.pyplot as plt
import numpy as np

from simulation.potential import Potential, PowerFunc


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
        return (f"Ratios: Median error: {self.median_e_ratio}, "
                f"Mean error: {self.mean_e_ratio}, Max error: {self.max_e_ratio}")

    def show(self, expr=None, *num_list_of_args):
        if expr is None:
            plt.scatter(self.args[num_list_of_args[0]], self.dif)
        else:
            x = np.vectorize(expr)(*[self.args[i] for i in num_list_of_args])
            plt.scatter(x, self.dif, s=1)


def potentialTest(n: int, d: float, m: int):
    """
    Test if the interpolated potential agrees with the directly calculated potential.
    m: number of sample points
    """
    a, b = 1, 1 / (1 + (n - 1) * d / 2)
    x = np.random.uniform(-2 * a, 2 * a, (m,))
    y = np.random.uniform(-a - b, a + b, (m,))
    t1 = np.random.uniform(0, 2 * pi, (m,))
    t2 = np.random.uniform(0, 2 * pi, (m,))
    potential = TestPotential(Potential(n, d, PowerFunc(2.5)).cal_potential(threads=4))

    v = np.vectorize(potential.interpolatePotential)(x, y, t1, t2)
    v_ref = np.vectorize(potential.precisePotential)(x, y, t1, t2)
    print(f"mean potential: {np.mean(v_ref)}")
    dif = np.abs(v - v_ref)
    return TestResult((x, y, t1, t2), dif, np.mean(v_ref))


def gradientTest(n: int, d: float, m: int):
    """
    Test if the interpolated gradient agrees with the directly calculated gradient.
    m: number of sample points
    """
    a, b = 1, 1 / (1 + (n - 1) * d / 2)
    x = np.random.uniform(-2 * a, 2 * a, (m,))
    y = np.random.uniform(-a - b, a + b, (m,))
    t1 = np.random.uniform(0, 2 * pi, (m,))
    t2 = np.random.uniform(0, 2 * pi, (m,))
    potential = TestPotential(Potential(n, d, PowerFunc(2.5)).cal_potential(threads=4))

    g = np.zeros((m, 6))
    g_ref = np.zeros((m, 6))
    for i in range(m):
        try:
            g[i] = potential.interpolateGradient(x[i], y[i], t1[i], t2[i])
            g_ref[i] = potential.preciseGradient(x[i], y[i], t1[i], t2[i])
        except:
            g_should = potential.preciseGradient(x[i], y[i], t1[i], t2[i])
            print(f"(x={x[i]}, y={y[i]}, t1={t1[i]}, t2={t2[i]}) should be {g_should}")

    r = g.reshape(-1).dot(g_ref.reshape(-1)) / g.reshape(-1).dot(g.reshape(-1))
    # there are bad cases, do not use np.mean
    print(f"regressive ratio: {r}")
    g_abs = np.sqrt(np.sum(g_ref ** 2, axis=1))
    print(f"mean amplitude of gradients: {np.mean(g_abs)}")
    dif = np.sqrt(np.sum((r * g - g_ref) ** 2 / 6, axis=1))
    return TestResult((x, y, t1, t2), dif, np.mean(g_abs))


if __name__ == '__main__':
    res = gradientTest(6, 0.05, 100000)
    print(res)
    res.show(lambda x, y: np.sqrt(x ** 2 + y ** 2), 0, 1)
    plt.show()
