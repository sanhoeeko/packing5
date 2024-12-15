import cProfile
import os.path
import re
import time

import numpy as np


def setWorkingDirectory():
    import os, sys
    if sys.platform.startswith('linux'):
        working_dir = "/home/gengjie/packing5/program"
        os.chdir(working_dir)


setWorkingDirectory()

# determine key parameters

with open(os.path.join(os.getcwd(), 'simulation/packing5Cpp/defs.h'), 'r') as f:
    defs_content = f.read()


def find_definition(macro: str):
    return int(re.search(rf"#define\s+{macro}\s+(\d+)", defs_content).group(1))


# cores = find_definition('cores')
cores = 1

max_neighbors = find_definition('max_neighbors')
digit_x = find_definition('DIGIT_X')
digit_y = find_definition('DIGIT_Y')
digit_t = find_definition('DIGIT_T')
digit_r = find_definition('DIGIT_R')
potential_table_shape = (2 ** digit_x, 2 ** digit_y, 2 ** digit_t)
sz1d = 2 ** digit_r


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

    def set_data(self, src: np.ndarray):
        """
        Dangerous! Only for test! We assume that all calculation including CArray should happen in C++.
        """
        np.copyto(self.data, src)


def CArrayF(arr: np.ndarray):
    return CArray(arr, np.float32)


def CArrayFZeros(*args, **kwargs):
    return CArray(np.zeros(*args, **kwargs), np.float32)


# h5 metadata management

def strToTuple(s: str):
    return tuple(map(lambda x: x.strip(), s.split(':')))


class HasMeta:
    """
    Ensure that there is a meta_hint: str at the top of class,
    format: <attr_name 1>: <type 1>, <attr_name 2>: <type 2>
    """

    def __init__(self):
        self.dtype = list(map(strToTuple, self.meta_hint.split(',')))
        self.key_list = [x[0] for x in self.dtype]

    @property
    def metadata(self) -> np.ndarray:
        values = [getattr(self, key) for key in self.key_list]
        return np.array([tuple(values)], dtype=self.dtype)


class Profile:
    def __init__(self, output_file: str):
        self.file_name = output_file
        self.profiler = cProfile.Profile()
        self.t0 = 0

    def __enter__(self):
        self.profiler.enable()
        self.t0 = time.time()
        return self.profiler

    def __exit__(self, exc_type, exc_value, traceback):
        t = time.time()
        self.profiler.disable()
        self.profiler.dump_stats(self.file_name)
        print(f"Program finished in {t - self.t0} seconds.")
