import numpy as np

from analysis.kernel import ker


def cubic_fit(x: np.ndarray, y: np.ndarray) -> (np.ndarray, float):
    """
    :param x, y: 1d arrays. They should have the same length and length >= 4.
    :return: coefficients, mse
    """
    coefficients = np.polyfit(x, y, 3)
    y_pred = np.poly1d(coefficients)(x)
    mse = np.mean((y - y_pred) ** 2)
    return coefficients, mse


def cubic_minimum(coefficients: np.ndarray) -> np.float32:
    """
    :param coefficients: [a, b, c, d] for a x^3 + b x^2 + c x + d
    :return: x value. If it doesn't exist, return nan.
    """
    a, b, c, d = coefficients.astype(np.float32)
    return ker.dll.CubicMinimum(a, b, c, d)


def CubicMinimumX(x: np.ndarray, y: np.ndarray, x_min: np.float32, x_max: np.float32) -> np.float32:
    coefficients, mse = cubic_fit(x, y)
    if mse > 1:
        print(f"Warning from CubicMinimumX: MSE too large: MSE={mse}")
    x0 = cubic_minimum(coefficients)
    if np.isnan(x0) or not x_min < x0 < x_max:
        poly = np.poly1d(coefficients)
        if poly(x_min) < poly(x_max):
            return x_min
        else:
            return x_max
    else:
        return x0


def CubicMinimumXNan(x: np.ndarray, y: np.ndarray, x_min: np.float32, x_max: np.float32) -> np.float32:
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return np.float32(np.nan)
    else:
        return CubicMinimumX(x, y, x_min, x_max)
