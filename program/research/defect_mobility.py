import numpy as np

from analysis.analysis import GeneralCalculation
from analysis.database import PickledSimulation


def get_phi(filenames: list[str], save=False, test=True):
    def get(simu: PickledSimulation):
        return simu.propertyInterval('phi', upper_h=1.2)

    GeneralCalculation(filenames, get, save, test, 'phi', 'average')


def test_bond_creation(filenames: list[str], save=False, test=True):
    def calculation(simu: PickledSimulation):
        return simu.bondCreation(num_threads=4, upper_h=1.2)

    GeneralCalculation(filenames, calculation, save, test, 'bond_creation', 'average')


def test_events(filenames: list[str], save=False, test=True):
    max_track_length = 40

    def calculation(simu: PickledSimulation):
        hist = simu.eventStat(max_track_length, num_threads=4, phi_c=0.86, upper_h=1.2)
        return hist

    GeneralCalculation(filenames, calculation, save, test, 'event_size', 'sum',
                       horizontal_axis=np.arange(1, max_track_length))


def test_stable_defect_number(filenames: list[str], save=False, test=True):
    def calculation(simu: PickledSimulation):
        return simu.stableDefects(num_threads=4, phi_c=0.86, upper_h=1.2)

    GeneralCalculation(filenames, calculation, save, test, 'event_size', 'average')


if __name__ == '__main__':
    test_stable_defect_number(['../data-20250419.h5'])
