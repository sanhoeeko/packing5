from analysis.analysis import calAllOrderParameters
from analysis.qualityCheck import *
from art.viewer import RenderSetup, InteractiveViewer


def batch_analyze(filename: str):
    db = Database(filename)
    # checkLegal(db)
    # print(db)
    # db.search_max_gradient()
    # checkGradient(db)
    # checkEnergy(db)
    e = db.find(gamma=1.9)[0]
    plotListOfArray(e[0].gradientCurve())
    InteractiveViewer(e[0], RenderSetup('EllipticPhi6')).show()
    # calAllOrderParameters(db, 'phi', num_threads=4)


if __name__ == '__main__':
    batch_analyze('data-20250215-0.h5')
