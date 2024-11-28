import cProfile
import random
import string
import threading
import time


class RandomStringGenerator:
    def __init__(self):
        self.lock = threading.Lock()

    def generate(self):
        with self.lock:
            return ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))


_random_string_generator = RandomStringGenerator()


def randomString():
    return _random_string_generator.generate()


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
