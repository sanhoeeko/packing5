import h5py
import matplotlib.pyplot as plt

from analysis.database import DatabaseBase
from analysis.h5tools import struct_array_to_dataframe
from art import curves as art


class MeanCIDatabase(DatabaseBase):
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.file = h5py.File(self.file_name, 'r')
        self.x_axis_name = self.file.attrs['x_axis_name']
        self.summary_table_array = self.file['summary_table'][:]
        self.ids = self.summary_table_array['id'].tolist()
        self.summary = self.process_summary(struct_array_to_dataframe(self.summary_table_array))

    def id(self, ensemble_id: str):
        return MeanCIData(self.file[ensemble_id], self.x_axis_name)

    def orderParameterList(self, prop: str) -> list[tuple]:
        return [(ensemble_data.x_axis, ensemble_data[prop]) for ensemble_data in self]


class MeanCIData:
    def __init__(self, h5_group, x_axis_name: str):
        self.dic = {}
        self.x_axis_name = x_axis_name
        for key in h5_group.keys():
            if key == x_axis_name:
                self.x_axis = h5_group[key][:]
            else:
                self.dic[key] = (h5_group[key]['mean'][:], h5_group[key]['ci'][:])
        for k, v in self.dic.items():
            self.n_density = len(v[0])
            break

    def __getitem__(self, item):
        return self.dic[item]


def batch_analyze(filename: str, from_to: tuple):
    db = MeanCIDatabase(filename)
    x, y = zip(*db.orderParameterList('EllipticPhi6'))
    mean, ci = zip(*y)
    f, t = from_to
    art.plotMeanCurvesWithCI(x[f:t], mean[f:t], ci[f:t])
    plt.show()


if __name__ == '__main__':
    batch_analyze('analysis-20250306-0.h5', (0, 15))
