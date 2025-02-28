import h5py
import matplotlib.pyplot as plt

from analysis.database import DatabaseBase
from analysis.h5tools import struct_array_to_dataframe
from art import curves as art


class MeanCIDatabase(DatabaseBase):
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.file = h5py.File(self.file_name, 'r')
        self.summary_table_array = self.file['summary_table'][:]
        self.ids = self.summary_table_array['id'].tolist()
        self.summary = self.process_summary(struct_array_to_dataframe(self.summary_table_array))
        self.x_axis_name = self.file['x_axis_name']

    def id(self, ensemble_id: str):
        return MeanCIData(self.file[ensemble_id])

    def orderParameterList(self, prop: str) -> list[tuple]:
        return [ensemble_data[prop] for ensemble_data in self]


class MeanCIData:
    def __init__(self, h5_group):
        self.dic = {}
        for key in h5_group.keys():
            self.dic[key] = (h5_group[key]['mean'][:], h5_group[key]['ci'][:])
        for k, v in self.dic.items():
            self.n_density = len(v[0])
            break

    def __getitem__(self, item):
        return self.dic[item]


def batch_analyze(filename: str):
    db = MeanCIDatabase(filename)
    mean, ci = zip(*db.orderParameterList('S_global'))
    art.plotMeanCurvesWithCI(None, mean, ci)
    plt.show()


if __name__ == '__main__':
    batch_analyze('analysis-20250215-0.h5')
