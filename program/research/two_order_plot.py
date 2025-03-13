import matplotlib.pyplot as plt

from analysis.post_analysis import PostDatabase, PostData
from art.art import ListColor01, add_energy_level_colorbar


class RawOrderDatabase(PostDatabase):
    def id(self, ensemble_id: str):
        return RawOrderData(self.file[ensemble_id], self.x_axis_name)


class RawOrderData(PostData):
    def process_data(self, key, data_group):
        self.dic[key] = data_group[:]


def two_order_plot(ax, filename: str, order1: str, order2: str):
    db = RawOrderDatabase(filename)
    colormap = 'jet'
    colors = ListColor01(colormap, len(db))
    gammas = db.summary['gamma']
    for i, ensemble in enumerate(db):
        o1 = ensemble[order1].reshape(-1)
        o2 = ensemble[order2].reshape(-1)
        ax.scatter(o1, o2, s=1, alpha=0.1, c=colors[i])
    add_energy_level_colorbar(ax, colormap, gammas, 'gamma')


if __name__ == '__main__':
    fig, ax = plt.subplots()
    two_order_plot(ax, '../full-20250313.h5', 'S_local', 'EllipticPhi6')
    plt.show()
