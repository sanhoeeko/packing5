from analysis.post_analysis import MergePostDatabase
from analysis.utils import setWorkingDirectory
from research.two_order_plot import RawOrderDatabase

setWorkingDirectory()

if __name__ == '__main__':
    MergePostDatabase(RawOrderDatabase, 'merge-full-0407.h5')(
        '../full-data-20250404.h5',
        '../full-data-20250405.h5',
        '../full-data-20250406.h5',
        '../full-data-20250406-2.h5',
        '../full-data-20250407.h5',
    )
    RawOrderDatabase('merge-full-0407.h5').mean_ci('merge-analysis-0407.h5')
