import numpy as np

from . import utils as ut
from .kernel import ker


class StatePool:
    def __init__(self, N: int, capacity: int):
        self.N = N
        self.capacity = capacity
        self.pool = ut.CArrayFZeros((capacity, N, 4))
        self.averaged = ut.CArrayFZeros((N, 4))
        self.energies = ut.CArrayFZeros((capacity,))
        self.current_ptr = 0

    def average(self, temperature: float) -> ut.CArray:
        ker.dll.AverageState(temperature, self.pool.ptr, self.energies.ptr, self.averaged.ptr, self.N, self.capacity)
        return self.averaged

    def average_zero_temperature(self) -> ut.CArray:
        """
        Python code:
        idx = np.argmin(self.energies.data)
        return ut.CArray(self.pool[idx, :, :])
        """
        ker.dll.AverageStateZeroTemperature(self.pool.ptr, self.energies.ptr, self.averaged.ptr, self.N, self.capacity)
        return self.averaged

    def add(self, state, value):
        np.copyto(self.pool.data[self.current_ptr, :, :], state.xyt.data)
        self.energies.data[self.current_ptr] = value
        self.current_ptr += 1

    def clear(self):
        self.current_ptr = 0
