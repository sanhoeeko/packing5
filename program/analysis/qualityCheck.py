import numpy as np

from art.art import Figure
from art.curves import plotMeanCurvesWithCI, plotListOfArray
from .analysis import averageEnergy
from .database import Database


def ScalarCurve(x_name: str, y_name: str):
    if y_name == 'state_distance':
        def inner(db: Database, i: int, j: int) -> (np.ndarray, np.ndarray):
            return db.property(x_name)[i, j, :-1], db.simulation_at(i, j).stateDistance()
    else:
        def inner(db: Database, i: int, j: int) -> (np.ndarray, np.ndarray):
            return db.property(x_name)[i, j, :], db.property(y_name)[i, j, :]

    return inner


def plotCurves(db: Database, x_name: str, y_name: str):
    curve = ScalarCurve(x_name, y_name)
    with Figure() as fig:
        for i in range(db.m_groups):
            for j in range(db.n_parallels):
                fig.ax.scatter(*curve(db, i, j), s=2)


def checkGradient(db: Database):
    plotCurves(db, 'phi', 'gradient_amp')


def checkStateDistance(db: Database):
    plotCurves(db, 'phi', 'state_distance')


def checkEnergy(db: Database):
    plotMeanCurvesWithCI(*averageEnergy(db, 'phi'), x_label='packing fraction', y_label='energy')


def checkDescentCurveAt(db: Database, i: int, j: int):
    plotListOfArray(db.simulation_at(i, j).normalizedDescentCurve())
