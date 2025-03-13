import h5py

from analysis.database import DatabaseBase
from analysis.h5tools import struct_array_to_dataframe


class PostDatabase(DatabaseBase):
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.file = h5py.File(self.file_name, 'r')
        self.x_axis_name = self.file.attrs['x_axis_name']
        self.summary_table_array = self.file['summary_table'][:]
        self.ids = self.summary_table_array['id'].tolist()
        self.summary = self.process_summary(struct_array_to_dataframe(self.summary_table_array))

    def orderParameterList(self, prop: str) -> list[tuple]:
        return [(ensemble_data.x_axis, ensemble_data[prop]) for ensemble_data in self]


class PostData:
    def __init__(self, h5_group, x_axis_name: str):
        self.dic = {}
        self.x_axis_name = x_axis_name
        for key in h5_group.keys():
            if key == x_axis_name:
                self.x_axis = h5_group[key][:]
            else:
                self.process_data(key, h5_group[key])

        for k, v in self.dic.items():
            self.n_density = len(v[0])
            break

    def process_data(self, key, data_group):
        raise NotImplementedError

    def __getitem__(self, item):
        return self.dic[item]
