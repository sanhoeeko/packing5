import numpy as np

from analysis.analysis import GeneralCalculation
from analysis.database import PickledSimulation


def test1(filenames: list[str], save=False, test=True):
    def calculation(simu: PickledSimulation):
        return simu.bondCreation(num_threads=4, upper_h=1.2)

    GeneralCalculation(filenames, calculation, save, test, 'bond_creation', 'average')


def test2(filenames: list[str], save=False, test=True):
    max_track_length = 40

    def calculation(simu: PickledSimulation):
        return simu.eventStat(max_track_length, num_threads=4, upper_h=1.2)

    GeneralCalculation(filenames, calculation, save, test, 'event_size', 'sum',
                       horizontal_axis=np.arange(1, max_track_length))


if __name__ == '__main__':
    test2(['../data-20250419.h5'])
