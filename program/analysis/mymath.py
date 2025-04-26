import numpy as np
from dask import delayed, compute
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from . import utils as ut
from .kernel import ker


class DirtyDataException(Exception):
    def __init__(self, nth_state: int, *args):
        super().__init__(*args)
        self.nth_state = nth_state


def CIRadius(data: np.ndarray, axis: int, confidence=0.95):
    """
    SEM: standard error of mean.
    CI: confidence interval.
    SEM = sqrt(Sn2 / (n * (n - 1))), where Sn2 is sum of squares.
    CI radius = t_factor(confidence) * SEM.
    When data are few, t distribution is far from normal distribution, then we need t factor,
    which makes CI slightly larger.
    If there is only one sample, CIRadius will return nan.
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


def interpolate_x(x: np.ndarray, eps: float):
    x_max = np.max(x)
    x_min = np.min(x)
    X = np.linspace(x_min, x_max, int(np.ceil((x_max - x_min) / eps)))
    return X


def interpolate_y(x: np.ndarray, y: np.ndarray, X: np.ndarray, num_threads=1) -> (np.ndarray, np.ndarray):
    shape = x.shape
    Y_shape = list(y.shape)
    Y_shape[-1] = X.shape[-1]
    Y = np.zeros(tuple(Y_shape))

    # validity check
    invalidity = np.isnan(y) | np.isinf(y)
    if np.any(invalidity):
        nan_position = np.where(invalidity)[-1][0]
        raise DirtyDataException(nan_position, "NAN value detected in interpolation!")

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
    return Y


def interpolate_tensor(x: np.ndarray, y: np.ndarray, eps: float, num_threads=1):
    X = interpolate_x(x, eps)
    Y = interpolate_y(x, y, X, num_threads)
    return X, Y


def nanstack(arrays, axis=0):
    max_shape = [max(arr.shape[i] if i < len(arr.shape) else 0 for arr in arrays) for i in
                 range(len(max(arrays, key=lambda x: x.ndim).shape))]

    result_shape = max_shape.copy()
    result_shape.insert(axis, len(arrays))
    result = np.full(result_shape, np.nan)

    for i, arr in enumerate(arrays):
        slices = [slice(None)] * len(max_shape)
        for j in range(len(arr.shape)):
            slices[j] = slice(0, arr.shape[j])
        slices.insert(axis, i)
        result[tuple(slices)] = arr

    return result


def isParticleTooClose(xyt: ut.CArray) -> bool:
    ratio = ker.dll.RijRatio(xyt.ptr, xyt.data.shape[0])
    return ratio < 0.01


def isParticleOutOfBoundary(xyt: ut.CArray, A: float, B: float) -> bool:
    return bool(ker.dll.isOutOfBoundary(xyt.ptr, xyt.data.shape[0], A, B))


def bin_and_smooth(x, y, num_bins=100, apply_gaussian=False, sigma=1) -> (np.ndarray, np.ndarray):
    """
    Perform binning and averaging with optional Gaussian smoothing.

    Parameters:
    - x (np.ndarray): The input x-values (non-uniform).
    - y (np.ndarray): The corresponding y-values.
    - num_bins (int): Number of bins to divide x into.
    - apply_gaussian (bool): Whether to apply Gaussian smoothing.
    - sigma (float): The standard deviation for Gaussian kernel (if smoothing).

    Returns:
    - x_binned (np.ndarray): The binned x-values (bin centers).
    - y_binned (np.ndarray): The binned and (optionally) smoothed y-values.
    """
    # Define bin edges
    bins = np.linspace(x.min(), x.max(), num_bins + 1)

    # Digitize x into bins
    bin_indices = np.digitize(x, bins)

    # Compute bin centers and averages
    x_binned = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    y_binned = [y[bin_indices == i].mean() if np.any(bin_indices == i) else np.nan
                for i in range(1, len(bins))]

    # Remove NaN values from empty bins
    x_binned = np.array([v for v in x_binned if not np.isnan(v)])
    y_binned = np.array([v for v in y_binned if not np.isnan(v)])

    # Apply Gaussian smoothing if required
    if apply_gaussian:
        y_binned = gaussian_filter1d(y_binned, sigma=sigma)

    return x_binned, y_binned
