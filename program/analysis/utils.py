from concurrent.futures import ThreadPoolExecutor

import numpy as np

from analysis.h5tools import invalid_value_of


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
