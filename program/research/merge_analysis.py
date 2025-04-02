from analysis.post_analysis import MergePostDatabase
from analysis.utils import setWorkingDirectory
from research.two_order_plot import RawOrderDatabase

setWorkingDirectory()

if __name__ == '__main__':
    MergePostDatabase(RawOrderDatabase, 'merge-full.h5')('../full-20250314.h5', '../full-20250315.h5')
