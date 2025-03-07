import matplotlib.pyplot as plt
import numpy as np

from art.art import Figure
from art.curves import plotMeanCurvesWithCI, plotListOfArray
from h5tools.utils import flatten
from . import mymath as mm
from .analysis import averageEnergy
from .database import Database, PickledEnsemble


def ScalarCurve(x_name: str, y_name: str):
    if y_name == 'state_distance':
        def inner(e: PickledEnsemble) -> (np.ndarray, np.ndarray):
            return [(e.property(x_name)[i, :-1], e.simulation_at(i).stateDistance()) for i in range(len(e))]
    else:
        def inner(e: PickledEnsemble) -> (np.ndarray, np.ndarray):
            return [(e.property(x_name)[i, :], e.property(y_name)[i, :]) for i in range(len(e))]

    return inner


def plotCurves(db: Database, x_name: str, y_name: str):
    with Figure() as fig:
        xs_ys = flatten(db.apply(ScalarCurve(x_name, y_name)))
        for x, y in xs_ys:
            fig.ax.scatter(x, y, s=2)
        fig.labels(x_name, y_name)


def checkGradient(db: Database):
    plotCurves(db, 'phi', 'normalized_gradient_amp')


def checkStateDistance(db: Database):
    plotCurves(db, 'phi', 'state_distance')


def checkEnergy(db: Database):
    plotMeanCurvesWithCI(
        *db.apply(lambda ensemble: averageEnergy(ensemble, 'phi')),
        x_label='packing fraction', y_label='energy'
    )


def checkEnergyCurveAt(db: Database, i: int, j: int):
    plotListOfArray(db[i].simulation_at(j).energyCurve(), ('descent steps', 'E/E0'))


def checkGradientCurveAt(db: Database, i: int, j: int):
    plotListOfArray(db[i].simulation_at(j).meanGradientCurve(), ('descent steps', 'mean gradient'))


def checkMaxGradientCurveAt(db: Database, i: int, j: int):
    plotListOfArray(db[i].simulation_at(j).maxGradientCurve(), ('descent steps', 'max gradient'))


def checkLegal(db: Database):
    """
    imshow the number of illegal configurations for each ensemble, each rho/phi
    """
    lst = [ensemble.illegalMap() for ensemble in db]
    illegal_tensor = mm.nanstack(lst, axis=0)
    lines = illegal_tensor.shape[0] * illegal_tensor.shape[1]
    illegal_map = illegal_tensor.reshape(lines, illegal_tensor.shape[2])
    plt.imshow(illegal_map)
    plt.show()
