import matplotlib.pyplot as plt

from analysis.analysis import GeneralCalculation
from analysis.database import PickledSimulation
from analysis.post_analysis import MeanCIDatabase
from art import curves as art


def read_op(filename: str, order_parameter_name: str, from_to: tuple):
    db = MeanCIDatabase(filename)
    # db.to_csv('defect', 'defect.csv')
    # db.to_csv('S_local', 'S_local.csv')
    dic = db.extract_data(order_parameter_name, truncate_h=1.2)
    f, t = from_to
    art.plotMeanCurvesWithCI(dic[db.x_axis_name][f:t], dic['mean'][f:t], dic['ci'][f:t], dic['gammas'][f:t],
                             x_label='volume fraction', y_label=order_parameter_name)
    plt.show()


def calculate_op(filenames: list[str], order_parameter_name: str, save=False, test=True):
    def calculation(simu: PickledSimulation):
        return simu.op(order_parameter_name, upper_h=1.2, num_threads=4)

    GeneralCalculation(filenames, calculation, save, test, output_name=order_parameter_name, aggregate_method='average')


if __name__ == '__main__':
    # read_op('../analysis-data-20250514.h5', 'S_local', (0, 20))
    calculate_op(
        ['../data-20250419.h5', ],
        'total_topological_charge')
