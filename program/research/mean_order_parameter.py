import matplotlib.pyplot as plt

from analysis.post_analysis import PostDatabase, PostData
from art import curves as art


class MeanCIDatabase(PostDatabase):
    def id(self, ensemble_id: str):
        return MeanCIData(self.file[ensemble_id], self.x_axis_name)


class MeanCIData(PostData):
    def process_data(self, key, data_group):
        self.dic[key] = (data_group['mean'][:], data_group['ci'][:])


def batch_analyze(filename: str, order_parameter_name: str, from_to: tuple):
    db = MeanCIDatabase(filename)
    x, y = zip(*db.orderParameterList(order_parameter_name))
    mean, ci = zip(*y)
    gammas = db.summary['gamma']
    f, t = from_to
    art.plotMeanCurvesWithCI(x[f:t], mean[f:t], ci[f:t], gammas[f:t],
                             x_label='volume fraction', y_label=order_parameter_name)
    plt.show()


if __name__ == '__main__':
    batch_analyze('../analysis-20250313.h5', 'S_local', (0, 20))
