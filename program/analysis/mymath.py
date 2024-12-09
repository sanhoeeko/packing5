import numpy as np
from dask import delayed, compute
from scipy import stats
from scipy.interpolate import CubicSpline


def CIRadius(data: np.ndarray, axis: int, confidence=0.95):
    """
    SEM: standard error of mean.
    CI: confidence interval.
    SEM = sqrt(Sn2 / (n * (n - 1))), where Sn2 is sum of squares.
    CI radius = t_factor(confidence) * SEM.
    When data are few, t distribution is far from normal distribution, then we need t factor,
    which makes CI slightly larger.
    """
    # Calculate the standard deviation along the specified axis. `ddof=1` -> (n-1) denominator
    sample_std = np.nanstd(data, axis=axis, ddof=1)

    # Calculate the standard error of the mean along the specified axis
    n = np.sum(~np.isnan(data), axis=axis)
    sem = sample_std / np.sqrt(n)

    # Get the t-value for the given confidence level and degrees of freedom
    dof = n - 1
    t_value = stats.t.ppf((1 + confidence) / 2, dof)

    # Calculate the radius of the confidence interval
    ci_radius = t_value * sem

    return ci_radius


def interpolate_tensor(x: np.ndarray, y: np.ndarray, eps: float, num_threads=1):
    shape = x.shape
    x_max = np.max(x)
    x_min = np.min(x)
    X = np.linspace(x_min, x_max, int(np.ceil((x_max - x_min) / eps)))
    Y_shape = list(y.shape)
    Y_shape[-1] = X.shape[-1]
    Y = np.zeros(tuple(Y_shape))

    # interpolate with respect to the last dimension
    def interpolate_slice(i):
        for j in range(shape[1]):
            cs = CubicSpline(x[i, j, :], y[i, j, :], extrapolate=False)
            Y[i, j, :] = cs(X)
        return Y[i, :, :]

    if num_threads == 1:
        for i in range(shape[0]): interpolate_slice(i)
    else:
        tasks = [delayed(interpolate_slice)(i) for i in range(shape[0])]
        results = compute(*tasks, scheduler='threads', num_workers=num_threads)
        for i, result in enumerate(results):
            Y[i, :, :] = result
    return X, Y
