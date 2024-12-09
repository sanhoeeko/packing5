import cProfile
import inspect
import random
import string
import threading
import time
from itertools import product

import numpy as np


class RandomStringGenerator:
    def __init__(self):
        self.lock = threading.Lock()

    def generate(self):
        with self.lock:
            return ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))


_random_string_generator = RandomStringGenerator()


def randomString():
    return _random_string_generator.generate()


def flatten(llst: list):
    return [x for lst in llst for x in lst]


def current_time() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def dict_to_numpy_struct(input_dict: dict, max_str_len: int):
    dtype = []
    for key, value in input_dict.items():
        if isinstance(value, str):
            dtype.append((key, f'S{max_str_len}'))
        elif isinstance(value, float):
            dtype.append((key, 'f4'))
        elif isinstance(value, int):
            dtype.append((key, 'i4'))
        else:
            dtype.append((key, type(value)))

    structured_array = np.zeros(1, dtype=dtype)
    for key, value in input_dict.items():
        structured_array[key][0] = value
    return structured_array


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


def FuncCartesian(func, **attribute_lists):
    # Get the names of the parameters of the class constructor
    signature = inspect.signature(func)
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
    return [func(**dict(zip(attribute_lists.keys(), attributes))) for attributes in cartesian_product]
