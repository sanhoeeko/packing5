from concurrent.futures import ThreadPoolExecutor

import numpy as np


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


def Map(threads: int):
    if threads > 1:
        def map_func(func, tasks):
            with ThreadPoolExecutor(max_workers=min(threads, len(tasks))) as executor:
                return list(executor.map(func, tasks))
    else:
        def map_func(func, tasks):
            return [func(task) for task in tasks]

    return map_func
