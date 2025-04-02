from analysis.visualize import AnalysisData
from analysis.database import Database
from h5tools.dataset import *
from analysis.qualityCheck import *
from art.viewer import *
from analysis.analysis import *

if __name__ == '__main__':
    auto_pack()
    db = Database('data.h5')
    checkMaxGradient(db)
    checkEnergy(db)
    checkMeanGradientCurveAt(db, 0, 0)
    checkMaxGradientCurveAt(db, 0, 0)
    checkEnergyCurveAt(db, 0, 0)
    InteractiveViewer(db[0].simulation_at(0), RenderSetup('EllipticPhi6')).show()
    # calAllOrderParameters(db, 'rho', num_threads=4)
