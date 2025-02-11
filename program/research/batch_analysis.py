from analysis.qualityCheck import *
from art.viewer import RenderSetup, InteractiveViewer


def batch_analyze(filename: str):
    db = Database(filename)
    # print(db)
    # db.search_max_gradient()
    # checkGradient(db)
    # checkEnergy(db)
    e = db.find(gamma=1.65)[0]
    plotListOfArray(e[1].gradientCurve())
    InteractiveViewer(e[1], RenderSetup('z_number')).show()


if __name__ == '__main__':
    batch_analyze('data-20250211-0.h5')
