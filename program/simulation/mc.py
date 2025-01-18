import numpy as np

from . import utils as ut
from .kernel import ker


class StatePool:
    def __init__(self, N: int, capacity: int):
        self.N = N
        self.capacity = capacity
        self.pool = ut.CArrayFZeros((capacity, N, 4))
        self.energies = ut.CArrayFZeros((capacity,))
        self.current_ptr = 0

    # def average(self, percentage: float) -> ut.CArray:
    #     least_n = int(self.capacity * percentage)

    def average(self) -> ut.CArray:
        idx = np.argmin(self.energies.data)
        return ut.CArray(self.pool[idx, :, :])

    def add(self, state, value):
        np.copyto(self.pool[self.current_ptr, :, :], state.xyt.data)
        # self.energies.data[self.current_ptr] = ker.dll.FastNorm(state.CalGradient_pure().ptr, state.N * 4)
        self.energies.data[self.current_ptr] = value
        self.current_ptr += 1
