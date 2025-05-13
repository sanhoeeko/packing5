from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np

from analysis.h5tools import invalid_value_of


def setWorkingDirectory():
    import os, sys
    if not sys.platform.startswith('linux'):
        working_dir = "D:/py/packing5/program"
        os.chdir(working_dir)


setWorkingDirectory()


# C++ data management

class CArray:
    def __init__(self, arr: np.ndarray, dtype=None):
        if dtype is None:
            dtype = arr.dtype
        if arr.flags['C_CONTIGUOUS']:
            self.data: np.ndarray = arr.astype(dtype)
        else:
            self.data: np.ndarray = np.ascontiguousarray(arr, dtype=dtype)
        self.ptr = self.data.ctypes.data

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def copy(self) -> 'CArray':
        return CArray(self.data.copy())


def CArrayF(arr: np.ndarray):
    return CArray(arr, np.float32)


def CArrayFZeros(*args, **kwargs):
    return CArray(np.zeros(*args, **kwargs), np.float32)


# parallel


def Map(threads: int):
    if threads > 1:
        def map_func(func, tasks):
            with ThreadPoolExecutor(max_workers=min(threads, len(tasks))) as executor:
                return list(executor.map(func, tasks))
    else:
        def map_func(func, tasks):
            return [func(task) for task in tasks]
    return map_func


def actual_length_of_1d_array(arr: np.ndarray) -> int:
    """
    :return: the first index of invalid value; if there is not, return the length of array.
    """
    val = invalid_value_of(arr)
    diff = arr == val
    if not np.any(diff):
        return len(arr)
    else:
        return np.where(diff)[0][0]


def struct_to_dict(arr: np.ndarray):
    fields = arr.dtype.names
    values = list(arr.item())
    for i in range(len(values)):
        if type(values[i]) is bytes:
            values[i] = values[i].decode('utf-8')
    return dict(zip(fields, values))


def dict_to_struct_array(dic: dict) -> np.ndarray:
    """
    Assume that all fields in the struct array / dict are 'f4'.
    """

    def get_shape(obj):
        if isinstance(obj, (list, tuple)):
            return get_shape(obj[0])
        elif isinstance(obj, dict):
            return get_shape(next(iter(obj.values())))
        else:
            return obj.shape

    field_names = list(dic.keys())
    dtype = [(name, 'f4') for name in field_names]
    shape = get_shape(dic)
    structured_array = np.zeros(shape, dtype=dtype)
    for name in field_names:
        structured_array[name] = dic[name]
    return structured_array


def apply_struct(func, *args, **kwargs):
    """
    :param func: numpy function. e.g., np.mean, np.abs , ...
    :return: a wrapped function for structured array
    """

    def inner(arr) -> np.ndarray:
        field_names = arr.dtype.names
        dic = {name: func(arr[name], *args, **kwargs) for name in field_names}
        return dict_to_struct_array(dic)

    return inner


def first_larger_than(arr: np.ndarray, val) -> int:
    return np.where(arr > val)[0][0]


def reference_phi(gamma: Union[float, np.ndarray], h: float) -> Union[float, np.ndarray]:
    return (np.pi + 4 * (gamma - 1)) / (2 * gamma * (2 - h))


def indexInterval(phis: np.ndarray, gamma: float, phi_c: float = None, upper_h: float = None) -> (int, int):
    if phi_c is None:
        lower_index = 0
    else:
        lower_index = first_larger_than(phis, phi_c)
    if upper_h is None:
        upper_index = len(phis)
    else:
        upper_index = first_larger_than(phis, reference_phi(gamma, upper_h))
    return lower_index, upper_index


def gamma_star(gamma: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    # return 1 + (gamma - 1) * 2 / np.sqrt(3)
    return gamma * np.sqrt(3) / 2
