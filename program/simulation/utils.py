import cProfile
import ctypes as ct
import os.path
import re
import time

import h5py
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


cores = find_definition('cores')
max_neighbors = find_definition('max_neighbors')
digit_x = find_definition('DIGIT_X')
digit_y = find_definition('DIGIT_Y')
digit_t = find_definition('DIGIT_T')
digit_r = find_definition('DIGIT_R')
potential_table_shape = (2 ** digit_x, 2 ** digit_y, 2 ** digit_t)
sz1d = 2 ** digit_r


class ForceTorque(ct.Structure):
    _fields_ = [('force', ct.c_float), ('torque', ct.c_float)]


# exceptions

class ParticlesTooCloseException(Exception):
    def __init__(self): super().__init__("Particle too close!")


class OutOfBoundaryException(Exception):
    def __init__(self): super().__init__("Particle out of boundary!")


class FinalIllegalException(Exception):
    def __init__(self): super().__init__("Restart relaxation to find a legal configuration!")


class CalGradientException(Exception):
    def __init__(self): super().__init__("An C++ error occurred while calculating gradient!")


class NaNInGradientException(Exception):
    def __init__(self): super().__init__("NAN detected in gradient!")


class InitFailException(Exception):
    def __init__(self): super().__init__("Random Initialization failed!")


# C++ data management

class CArray:
    def __init__(self, arr: np.ndarray, dtype=None):
        if arr.flags['C_CONTIGUOUS']:
            if dtype is None:
                self.data = arr
            else:
                self.data = arr.astype(dtype)
        else:
            if dtype is None:
                self.data = np.ascontiguousarray(arr, dtype=np.float32)
            else:
                self.data = np.ascontiguousarray(arr, dtype=dtype)
        self.ptr = self.data.ctypes.data

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def copy(self) -> 'CArray':
        return CArray(self.data.copy())

    def copyto(self, dst: 'CArray'):
        np.copyto(dst.data, self.data)

    def set_data(self, src: np.ndarray):
        """
        Dangerous! Only for test! We assume that all calculation including CArray should happen in C++.
        """
        np.copyto(self.data, src)

    def norm(self, N: int) -> np.float32:
        from .kernel import ker
        return np.float32(ker.dll.FastNorm(self.ptr, N * 4) / np.sqrt(N))

    def max_abs(self, N: int) -> np.float32:
        from .kernel import ker
        return ker.dll.MaxAbsVector4(self.ptr, N * 4)

    def max_ft(self, N: int) -> ForceTorque:
        from .kernel import ker
        return ker.dll.FastMaxFT(self.ptr, N)

    def reshape(self, *shape):
        return CArray(self.data.reshape(*shape), None)


def CArrayF(arr: np.ndarray):
    if arr.dtype == np.float32:
        return CArray(arr, None)
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


class Cache:
    def __init__(self, obj):
        self.valid = False
        self._obj = obj

    def get(self):
        if self.valid:
            return self._obj
        else:
            raise ValueError("Cache is invalid!")

    def set(self, value):
        self._obj = value
        self.valid = True


class Timer:
    def __enter__(self):
        self.start_t = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_t = time.perf_counter()
        self.elapse_t = self.end_t - self.start_t


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


def add_dynamic_methods(cls, methods, heading_name: str):
    lst_name = heading_name + 's'
    setattr(cls, lst_name, [])
    for i, method in enumerate(methods):
        method_name = f"{heading_name}_{i}"
        setattr(cls, method_name, method)
        getattr(cls, lst_name).append(getattr(cls, method_name))


def save_array(file_name: str):
    def inner(variable_name: str, array: np.ndarray):
        with h5py.File(file_name, 'a') as f:
            if variable_name in f:
                dataset = f[variable_name]
                current_shape = dataset.shape
                lines = current_shape[0] + 1
                if lines > 10000: raise OSError  # test
                dataset.resize((lines,) + current_shape[1:])
                dataset[current_shape[0], :] = array
            else:
                maxshape = (None,) + array.shape
                array = array.reshape((1,) + array.shape)
                f.create_dataset(variable_name, data=array, maxshape=maxshape, chunks=True)

    return inner


dump = save_array('dump.h5')
