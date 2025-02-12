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


class SimpleLogger:
    def __init__(self):
        self.last_message = None
        self.count = 0

    def flush(self):
        self._output_last_message()

    def log(self, message=None):
        if message == self.last_message:
            self.count += 1
        else:
            self._output_last_message()
            self.last_message = message
            self.count = 1

    def _output_last_message(self):
        if self.count > 1:
            print(f"{self.last_message} x{self.count}")
        elif self.count == 1:
            print(self.last_message)
        self.last_message = None
        self.count = 0
