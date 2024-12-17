from analysis.visualize import AnalysisData

if __name__ == '__main__':
    from analysis.database import Database
    from analysis.qualityCheck import *
    from art.viewer import *
    from analysis.analysis import *

    # db = Database('data.h5')
    # calAllOrderParameters(db, 'rho')
    # InteractiveViewer(db.simulation_at(0, 0), RenderSetup('S_local', False, 'default', True)).show()
    # checkStateDistance(db)

    db = AnalysisData('analysis.h5')
    db.plot('Phi6')

    # phi6 = db.apply(OrderParameterFunc('Phi6', False, False))
    # s_local = db.apply(OrderParameterFunc('S_local', False, False))
    # scatterCorrelations(phi6[0, 0, :100], s_local[0, 0, :100])