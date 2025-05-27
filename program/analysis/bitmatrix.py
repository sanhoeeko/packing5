import numpy as np

from . import utils as ut
from .kernel import ker


class BitMatrix:
    min_bits = 256

    def __init__(self, rows: int):
        assert rows * rows % BitMatrix.min_bits == 0, f"The number of bits must be a multiple of {BitMatrix.min_bits}"
        self.rows = rows
        self.num_bytes = rows * rows // 8
        self.arr = ut.CArray(np.zeros((self.num_bytes,), dtype=np.uint8))

    def __sub__(self, o: 'BitMatrix'):
        res = BitMatrix(self.rows)
        ker.dll.bitmap_subtract(self.arr.ptr, o.arr.ptr, res.arr.ptr, self.num_bytes)
        return res

    def toBoolMatrix(self):
        return np.unpackbits(self.arr.data.reshape((self.rows, -1)), axis=1, bitorder='little').astype(bool)

    def toPairs(self) -> np.ndarray[np.int32]:
        res = ut.CArray(np.zeros((self.num_bytes, 2), dtype=np.int32))
        n_edges = ker.dll.bitmap_to_pairs(self.arr.ptr, res.ptr, self.rows)
        return res.data[:n_edges, :]

    def count(self) -> int:
        return int(ker.dll.bitmap_count(self.arr.ptr, self.num_bytes))
