import matplotlib.pyplot as plt

from analysis.post_analysis import PostDatabase, PostData


class RawOrderDatabase(PostDatabase):
    def id(self, ensemble_id: str):
        return RawOrderData(self.file[ensemble_id], self.x_axis_name)


class RawOrderData(PostData):
    def process_data(self, key, data_group):
        self.dic[key] = data_group[:]


def two_order_plot(filename: str, order1: str, order2: str):
    db = RawOrderDatabase(filename)
    for ensemble in db:
        o1 = ensemble[order1].reshape(-1)
        o2 = ensemble[order2].reshape(-1)
        plt.scatter(o1, o2, s=1)


if __name__ == '__main__':
    two_order_plot('../full-20250313.h5', 'MeanSegmentDist', 'S_local')
    plt.show()
