from analysis.visualize import AnalysisData
from analysis.database import Database
from h5tools.dataset import *
from analysis.qualityCheck import *
from art.viewer import *
from analysis.analysis import *

if __name__ == '__main__':
    auto_pack()
    db = Database('data.h5')
    # checkGradient(db)
    # calAllOrderParameters(db, 'rho')
    # InteractiveViewer(db[0].simulation_at(0), RenderSetup('z_number', False, 'voronoi', True)).show()
    checkEnergyCurveAt(db, 0, 0)

    # db = AnalysisData('analysis.h5')
    # db.plot('Phi6')

    # phi6 = db.apply(OrderParameterFunc('Phi6', False, False))
    # s_local = db.apply(OrderParameterFunc('S_local', False, False))
    # scatterCorrelations(phi6[0, 0, :100], s_local[0, 0, :100])