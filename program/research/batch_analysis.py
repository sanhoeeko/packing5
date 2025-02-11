from analysis.qualityCheck import *


def batch_analyze(filename: str):
    db = Database(filename)
    print(db)
    checkGradient(db)
    checkEnergy(db)


if __name__ == '__main__':
    batch_analyze('data0.h5')
