import numpy as np

from . import utils as ut
from .gradient import GradientMatrix
from .kernel import ker


class Grid:
    max_size = 8192

    def __init__(self, state):
        self.N = state.N
        self.state = state
        self.indices = ut.CArray(np.full((self.N,), -1, dtype=np.int32))
        grid = np.full((Grid.max_size, ut.max_neighbors), -1, dtype=np.int32)
        grid[:, 0] = 0
        self.grid = ut.CArray(grid)
        self.gradient = GradientMatrix(state, self)

    def init(self, A: float, B: float):
        """
        Call it when the boundary changes
        """
        self.m = np.int32(np.ceil(A / 2))
        self.n = np.int32(np.ceil(B / 2))
        self.x_shift = self.m + 1
        self.y_shift = self.n + 1
        self.lines = 2 * self.y_shift
        self.cols = 2 * self.x_shift
        self.size = self.lines * self.cols

    def gridLocate(self):
        """
        # Python code:
        i = np.round(self.parent.x / 2).astype(np.int32) + self.x_shift
        j = np.round(self.parent.y / 2).astype(np.int32) + self.y_shift
        self.indices.data[:] = (j * self.cols + i)[:]
        """
        self.init(self.state.A, self.state.B)
        ker.dll.GridLocate(self.state.xyt.ptr, self.indices.ptr, self.x_shift, self.y_shift, self.cols, self.N)
        ker.dll.GridTransform(self.indices.ptr, self.grid.ptr, self.N)
