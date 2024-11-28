import inspect
import re
from itertools import product

import numpy as np

# determine key parameters

with open('./simulation/packing5Cpp/defs.h', 'r') as f:
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


def CArrayF(arr: np.ndarray):
    return CArray(arr, np.float32)


def CArrayFZeros(*args, **kwargs):
    return CArray(np.zeros(*args, **kwargs), np.float32)


# h5 metadata management

class HasMeta:
    """
    Ensure that there is a metalist: list[str] at the top of class
    """

    @property
    def metadata(self):
        return [getattr(self, key) for key in self.metalist]


def ClassCartesian(cls, **attribute_lists):
    # Get the names of the parameters of the class constructor
    signature = inspect.signature(cls.__init__)
    allowed_attributes = set(signature.parameters.keys())
    allowed_attributes.discard('self')

    # Check if all required properties are present
    for key in attribute_lists:
        if key not in allowed_attributes:
            raise ValueError(f"Invalid attribute: {key}")
    for attr in allowed_attributes:
        if attr not in attribute_lists:
            raise ValueError(f"Missing attribute: {attr}")

    cartesian_product = product(*attribute_lists.values())
    return [cls(**dict(zip(attribute_lists.keys(), attributes))) for attributes in cartesian_product]
