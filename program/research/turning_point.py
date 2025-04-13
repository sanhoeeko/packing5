import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress

from analysis.analysis import averageEnergy
from analysis.database import Database
from analysis.post_analysis import MeanCIDatabase


def regression_trends(x: np.ndarray, y: np.ndarray, func=None):
    """
    参数:
    x (array_like): 自变量数据
    y (array_like): 因变量数据

    功能:
    1. 从最后2个点开始逐步增加数据量进行线性回归
    2. 绘制斜率、截距和R²随数据量变化的趋势图
    """
    assert len(x) == len(y) and len(x) > 2
    n = len(x)

    ks = []
    slopes = []
    intercepts = []
    r_squared = []

    for k in range(2, n):
        if func is None:
            x_sub = x[-k:]
            y_sub = y[-k:]
        else:
            x_sub = func(x[-k:] - x[-k - 1])
            y_sub = func(y[-k:])
        res = linregress(x_sub, y_sub)
        ks.append(k)
        slopes.append(res.slope)
        intercepts.append(res.intercept)
        r_squared.append(res.rvalue ** 2)

    return ks, slopes, intercepts, r_squared


def find_first_r_squared_drop(x, y, threshold=0.9999):
    """
    Find the first critical point in the regression trend results where R² is below the threshold, (x, y, k, b)
    Then calculate the linear critical point Xc, where k*Xc + b = 0
    :return: (overestimated) index of Xc
    """
    ks, slopes, intercepts, r_squared = regression_trends(x, y)
    for i, r2 in enumerate(r_squared):
        if r2 < threshold:
            Xl = x[-ks[i]]
            slope, intercept = slopes[i], intercepts[i]
            Xc = -intercept / slope
            print(f'Xl={Xl}, Xc={Xc}')
            for i, xi in enumerate(x):
                if xi > Xc:
                    return i
    return len(x)


def find_highest_prominence_peak(y: np.ndarray, window_length=21, polyorder=3, smooth=True, **kwargs):
    if smooth:
        y_processed = savgol_filter(y, window_length, polyorder)
    else:
        y_processed = y.copy()
    peaks, properties = find_peaks(
        y_processed,
        prominence=(None, None),  # 强制计算所有峰的突出度
        **kwargs
    )
    if len(peaks) == 0: return None
    prominences = properties["prominences"]
    max_prom_idx = np.argmax(prominences)
    return peaks[max_prom_idx]


def powerFit(x: np.ndarray, y: np.ndarray):
    idx = find_first_r_squared_drop(x, y)
    x = x[:idx]
    y = y[:idx]
    ks, slopes, intercepts, r_squared = regression_trends(x, y, np.log)
    jdx = find_highest_prominence_peak(r_squared, 9, 2)
    res = linregress(np.log(x[-jdx + 1:] - x[-jdx]), np.log(y[-jdx + 1:]))
    print(f"power={res.slope}, R2={res.rvalue}")
    return len(x) - jdx + 1


def first_turning_points(filename: str):
    db = Database(filename)
    phi, energy_mean, energy_ci = db.apply(lambda ensemble: averageEnergy(ensemble, 'phi'))
    turning_points = []
    for ph, e in zip(phi, energy_mean):
        idx = powerFit(ph, e)
        print(f"Xo={ph[idx]}")
        turning_points.append(idx)
    return turning_points


def EPhi6_under_turning_point(filename: str, pts: list):
    db = MeanCIDatabase(filename)
    dic = db.extract_data('EllipticPhi6')
    e_phi6 = dic['mean']
    e_phi6 = [y[:pts[i]] for i, y in enumerate(e_phi6)]
    for i, y in enumerate(e_phi6):
        plt.plot(dic['phi'][i][:len(y)], y)
    plt.show()


if __name__ == '__main__':
    pts = first_turning_points('../data-20250406.h5')
    EPhi6_under_turning_point('merge-analysis-0407.h5', pts)
