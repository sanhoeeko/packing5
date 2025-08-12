import numpy as np

from . import utils as ut
from .kernel import ker


class BitMatrix:
    min_bits = ker.dll.num_rod_required_for_bitmap()  # 8 or 256 depending on dll's version

    def __init__(self, rows: int):
        """
        :param rows: number of rods
        """
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


def detectEvents(previous_bonds: BitMatrix, current_bonds: BitMatrix, previous_z: ut.CArray,
                 current_z: ut.CArray) -> ut.CArray:
    """
    :return: (num_events, 5) CArray, columns are defined by:
    struct DefectEvent {
        int related_particles;
        int previous_negative_charge = 0, previous_positive_charge = 0;
        int current_negative_charge = 0, current_positive_charge = 0;
    }
    """
    new_bonds = current_bonds - previous_bonds
    max_events = new_bonds.count()
    events = ut.CArray(np.zeros((max_events, 5), dtype=np.int32))
    num_events = ker.dll.FindEventsInBitmap(new_bonds.rows, current_bonds.arr.ptr, new_bonds.arr.ptr, previous_z.ptr,
                                            current_z.ptr, events.ptr)
    assert num_events <= max_events
    events = events[:num_events]
    return events
