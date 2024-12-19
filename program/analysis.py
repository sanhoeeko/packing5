from analysis.visualize import AnalysisData
from analysis.database import Database
from h5tools.dataset import *
from analysis.qualityCheck import *
from art.viewer import *
from analysis.analysis import *

if __name__ == '__main__':

    db = Database('data.h5')
    # checkDescentCurveAt(db, 0, 0)
    calAllOrderParameters(db, 'rho')
    # InteractiveViewer(db.simulation_at(0, 0), RenderSetup('S_local', False, 'default', True)).show()
    # checkDescentCurveAt(db, 0, 0)

    # db = AnalysisData('analysis.h5')
    # db.plot('Phi6')

    # phi6 = db.apply(OrderParameterFunc('Phi6', False, False))
    # s_local = db.apply(OrderParameterFunc('S_local', False, False))
    # scatterCorrelations(phi6[0, 0, :100], s_local[0, 0, :100])