from analysis.qualityCheck import *
from art.viewer import RenderSetup, InteractiveViewer


def batch_analyze(filename: str):
    db = Database(filename)
    # checkLegal(db)
    # print(db)
    db.search_max_gradient()
    # checkMaxGradient(db, 0.1)
    # checkEnergy(db)
    e = db.find(gamma=2.4)[0][2]
    # plotListOfArray(e.energyCurve(), y_restriction=2)
    # plotListOfArray(e.meanGradientCurve(), y_restriction=2)
    # plotListOfArray(e.maxGradientCurve(), y_restriction=2)
    InteractiveViewer(e, RenderSetup('SegmentDistRankMask')).setMarkerSetup('').show()


if __name__ == '__main__':
    batch_analyze('../data-20250420-2.h5')
