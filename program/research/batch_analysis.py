from analysis.analysis import calAllOrderParameters
from analysis.qualityCheck import *
from art.viewer import RenderSetup, InteractiveViewer


def batch_analyze(filename: str):
    db = Database(filename)
    # checkLegal(db)
    # print(db)
    # db.search_max_gradient()
    # checkGradient(db, 0.1)
    # checkEnergy(db)
    e = db.find(gamma=1.9)[0]
    # plotListOfArray(e[0].energyCurve(), y_restriction=2)
    # InteractiveViewer(e[0], RenderSetup('S_local')).show()
    calAllOrderParameters(db, 'phi', num_threads=4)


if __name__ == '__main__':
    batch_analyze('data-20250306-0.h5')
