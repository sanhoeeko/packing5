from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np

import default
from h5tools.h5analysis import invalid_value_of


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

    def check(self):
        # to fix some bugs in multiprocess (Maybe in which `CArray`s are incorrectly copied)
        self.ptr = self.data.ctypes.data


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


def filenamesFromTxt(txt_filename: str) -> list[str]:
    """
    :param txt_filename: txt file that records all data files' names
    :return: list of data files' names
    """
    with open(txt_filename, 'r') as f:
        lst = f.readlines()
    res = []
    for s in lst:
        s_strip = s.strip()
        if len(s_strip) > 0: res.append(s_strip)
    return res


def first_larger_than(arr: np.ndarray, val) -> int:
    return np.where(arr > val)[0][0]


def reference_phi(gamma: Union[float, np.ndarray], h: float) -> Union[float, np.ndarray]:
    return (np.pi + 4 * (gamma - 1)) / (2 * gamma * (2 - h))


def indexInterval(phis: np.ndarray, gamma: float, phi_c: float = None, upper_h: float = None,
                  upper_phi: float = None) -> (int, int):
    if phi_c is None:
        lower_index = 0
    else:
        lower_index = first_larger_than(phis, phi_c)
    if upper_h is None and upper_phi is None:
        upper_index = len(phis)
    else:
        if upper_h is not None and upper_phi is not None:
            raise ValueError
        elif upper_phi is None:
            upper_index = first_larger_than(phis, reference_phi(gamma, upper_h))
        else:
            upper_index = first_larger_than(phis, upper_phi)
    return lower_index, upper_index


def clipArray(arr: np.ndarray, val=0):
    """
    :return: remove zeros of nans at the end of the array
    """
    idx = np.where(arr == val)[0][0]
    return arr[:idx, ...]


def gamma_star(gamma: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    # return 1 + (gamma - 1) * 2 / np.sqrt(3)
    return gamma * np.sqrt(3) / 2


def phi(N: int, gamma: float, A: float, B: float):
    return N / (np.pi * A * B) * (np.pi + 4 * (gamma - 1)) / gamma ** 2


def InternalMask(phi: float, A: float, B: float, xyt: np.ndarray) -> np.ndarray[bool]:
    ratio = (default.phi_c - phi) / (default.phi_c - default.phi_0)
    if ratio <= 0:
        return np.zeros(xyt.shape[0], dtype=bool)
    b = B * ratio
    r = xyt[:, 0] ** 2 / A ** 2 + xyt[:, 1] ** 2 / b ** 2
    return r < 1


def InternalMask2(phi: float, A: float, B: float, xyt: np.ndarray) -> np.ndarray[bool]:
    u_ratio = 1 / 6  # boundary_Y / total_Y
    d_ratio = 1 / 3
    lr_ratio = 1 / 2  # boundary_X / total_X
    return ~np.bitwise_and(
        np.abs(xyt[:, 1]) > B * (1 - d_ratio), np.abs(xyt[:, 1]) < B * (1 - u_ratio),
        np.abs(xyt[:, 0]) < A * (1 - lr_ratio)
    )


def mask_structured_array(structured_arr: np.ndarray, mask: np.ndarray[bool]):
    zero_record = np.zeros((), dtype=structured_arr.dtype)
    structured_arr[~mask] = zero_record
    return structured_arr


def r_phi(phi: float):
    if phi < default.phi_c:
        r = (phi - default.phi_0) / (default.phi_c - default.phi_0) * (default.phi_c / phi)
    else:
        r = 1
    return r


def y_rank(N: int, phi: float, xyt: np.ndarray):
    y_ratio = 0.5
    r = r_phi(phi)
    abs_y = np.abs(xyt[:, 1])
    idx = min(int(round(r * N)), N - 1)
    critical_abs_y = np.sort(abs_y)[::-1][idx] * y_ratio
    mask = abs_y >= critical_abs_y
    return mask


def y_rank_2(N: int, phi: float, A: float, xyt: np.ndarray):
    r = r_phi(phi)
    x, y = xyt[:, 0], xyt[:, 1]
    abs_y = np.abs(y) / np.sqrt(A ** 2 - x ** 2) + 0.01
    idx = min(int(round(r * N)), N - 1)
    critical_abs_y = np.sort(abs_y)[::-1][idx]
    mask = abs_y >= critical_abs_y
    return mask
