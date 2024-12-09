import matplotlib.pyplot as plt
import numpy as np
import src.art as art
import src.utils as ut


class DistSlice:
    def __init__(self, dist_array: np.ndarray):
        self.data = dist_array

    def __str__(self):
        plt.plot(self.data)
        plt.show()
        return '<DistSlice object>'


class Distribution:
    def __init__(self, mat: np.ndarray):
        self.data = mat

    def __str__(self):
        lst = self.data.T.tolist()
        art.plotListOfArray(ut.sample_equal_stride(lst, 20))
        return '<Distribution object>'

    def showAsImage(self):
        plt.imshow(self.data)
        plt.show()

    def slice(self, index: int):
        return DistSlice(self.data[:, index])

    def final(self):
        return self.slice(self.data.shape[1] - 1)
