import matplotlib.pyplot as plt
import numpy as np

from art.curves import plotMeanCurvesWithCI, plotListOfArray, scatterList
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


def scatterCurves(db: Database, x_name: str, y_name: str, y_restriction: float = None):
    xs_ys = flatten(db.apply(ScalarCurve(x_name, y_name)))
    gammas = db.summary['gamma']
    scatterList(xs_ys, x_name, y_name, y_restriction, gammas)


def checkGradient(db: Database, y_restriction: float = None):
    scatterCurves(db, 'phi', 'normalized_gradient_amp', y_restriction)


def checkStateDistance(db: Database):
    scatterCurves(db, 'phi', 'state_distance')


def checkEnergy(db: Database):
    plotMeanCurvesWithCI(
        *db.apply(lambda ensemble: averageEnergy(ensemble, 'phi')),
        x_label='packing fraction', y_label='energy', gammas=db.summary['gamma']
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
