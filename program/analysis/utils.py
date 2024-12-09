from concurrent.futures import ThreadPoolExecutor

import h5py
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


# h5 interaction


def dict_to_analysis_hdf5(file_name: str, data_dict: dict):
    """
    :param data_dict: key: name of order parameter; value: tuple of (mean, ci)
    """
    with h5py.File(file_name, 'w') as hdf5_file:
        for key, value in data_dict.items():
            if isinstance(value, tuple):
                group = hdf5_file.create_group(key)
                group.create_dataset('mean', data=value[0], dtype=np.float32)
                group.create_dataset('ci', data=value[1], dtype=np.float32)
            else:
                hdf5_file.create_dataset(key, data=value, dtype=np.float32)


def add_array_to_hdf5(file_name: str, name: str, data: np.ndarray):
    with h5py.File(file_name, 'a') as hdf5_file:
        hdf5_file.create_dataset(name, data=data, dtype=data.dtype)
