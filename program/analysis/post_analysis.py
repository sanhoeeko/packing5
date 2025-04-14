import h5py
import numpy as np
import pandas as pd

from analysis.analysis import averageByReplica
from analysis.database import DatabaseBase
from analysis.h5tools import struct_array_to_dataframe, add_property_to_hdf5, add_array_to_hdf5, dict_to_analysis_hdf5
from h5tools.merge import group_similar_ids


class PostDatabase(DatabaseBase):
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.file = h5py.File(self.file_name, 'r')
        self.x_axis_name = self.file.attrs['x_axis_name']
        self._summary_table_array = self.file['summary_table'][:]
        self.ids = self._summary_table_array['id'].tolist()
        self.summary = self.process_summary(struct_array_to_dataframe(self._summary_table_array))

    def orderParameterList(self, prop: str) -> list[tuple]:
        return [(ensemble_data.x_axis, ensemble_data[prop]) for ensemble_data in self]

    @property
    def keys(self):
        return list(self.file[self.ids[0]].keys())


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


def MergePostDatabase(cls, output_filename: str):
    assert issubclass(cls, PostDatabase)
    output_file = h5py.File(output_filename, 'w')

    def inner(*filenames: str):
        def fetch(key: str):
            for pdb in pdbs:
                if key in pdb.file.keys():
                    return pdb.file[key]

        pdbs = [cls(filename) for filename in filenames]
        x_axis_name = pdbs[0].x_axis_name
        mapping = group_similar_ids([pdb.summary for pdb in pdbs])
        print("Mapping detected:")
        for line in mapping:
            print(line)
            h5_handles = [fetch(key) for key in line]
            keys = list(h5_handles[0].keys())
            new_group = output_file.create_group(line[0])
            for key in keys:
                if key == x_axis_name:
                    merged_ndarray = h5_handles[0][x_axis_name][:]
                else:
                    merged_ndarray = np.concatenate([group[key][:] for group in h5_handles], axis=0)
                new_group.create_dataset(key, shape=merged_ndarray.shape, dtype=merged_ndarray.dtype,
                                         data=merged_ndarray)
        # add x-axis
        add_property_to_hdf5(output_filename, 'x_axis_name', x_axis_name)

        # add metadata
        add_array_to_hdf5(output_filename, 'summary_table', pdbs[0]._summary_table_array)

    return inner


class MeanCIDatabase(PostDatabase):
    def id(self, ensemble_id: str):
        return MeanCIData(self.file[ensemble_id], self.x_axis_name)

    def extract_data(self, order_parameter_name: str) -> dict:
        x, y = zip(*self.orderParameterList(order_parameter_name))
        mean, ci = zip(*y)
        gammas = self.summary['gamma']
        return {self.x_axis_name: x, 'mean': mean, 'ci': ci, 'gammas': gammas}

    def to_csv(self, order_parameter_name: str, filename: str):
        dic = self.extract_data(order_parameter_name)
        gammas = dic['gammas']
        dfs = []
        for i, gamma in enumerate(gammas):
            appendix = f'(gamma={gamma:.1f})'
            x_header = self.x_axis_name + appendix
            mean_header = 'mean' + appendix
            ci_header = 'ci' + appendix
            mat = np.hstack([
                dic[self.x_axis_name][i].reshape(-1, 1),
                dic['mean'][i].reshape(-1, 1),
                dic['ci'][i].reshape(-1, 1),
            ])
            dfs.append(pd.DataFrame(mat, columns=[x_header, mean_header, ci_header]))
        df = pd.concat(dfs, axis=1)
        df.to_csv(filename)


class MeanCIData(PostData):
    def process_data(self, key, data_group):
        self.dic[key] = (data_group['mean'][:], data_group['ci'][:])


class RawOrderDatabase(PostDatabase):
    def id(self, ensemble_id: str):
        return RawOrderData(self.file[ensemble_id], self.x_axis_name)

    def mean_ci(self, out_file: str):
        """
        Create a mean-CI hdf5 data file.
        """
        dic = {}
        for id_str in self.ids:
            group = self.file[id_str]
            sub_dic = {}
            for key in self.keys:
                x, y_mean, y_ci = averageByReplica(group[self.x_axis_name][:], group[key][:])
                sub_dic[key] = (y_mean, y_ci)
            sub_dic[self.x_axis_name] = group[self.x_axis_name][:]
            dic[id_str] = sub_dic

        # add x-axis
        dict_to_analysis_hdf5(out_file, dic)
        add_property_to_hdf5(out_file, 'x_axis_name', self.x_axis_name)

        # add metadata
        add_array_to_hdf5(out_file, 'summary_table', self._summary_table_array)


class RawOrderData(PostData):
    def process_data(self, key, data_group):
        self.dic[key] = data_group[:]
