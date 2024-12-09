"""
analysis.database: Data Access Layer
"""

import h5py
import numpy as np
import pandas as pd

from . import utils as ut


class LazyArray:
    def __init__(self, hdf5_file: str, dataset_name: str):
        self.file = h5py.File(hdf5_file, 'r')
        self.dataset = self.file[dataset_name]
        self.shape = self.dataset.shape
        self.dtype = self.dataset.dtype

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f"LazyArray(shape={self.shape}, dtype={self.dtype})"

    def close(self):
        self.file.close()


def read_hdf5_to_dict(file_path: str) -> dict:
    data_dict = {}
    with h5py.File(file_path, 'r') as file:
        for dataset_name in file:
            if dataset_name.endswith('table'):
                data_dict[dataset_name] = file[dataset_name][:]
            else:
                data_dict[dataset_name] = LazyArray(file_path, dataset_name)
    return data_dict


def struct_array_to_dataframe(data: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(data.reshape(-1))
    # convert strings
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.object_):
            df[column] = df[column].str.decode('utf-8')
    return df


class Database:
    """
    The first three dimensions of data:
    1. for the same "external" conditions
    2. for the same shape of particles
    3. for one simulation
    """
    def __init__(self, file_name: str):
        self.file_name = file_name
        dic = read_hdf5_to_dict(file_name)

        # configuration: 3 + 2 = 5 dim
        self.configuration: LazyArray = dic['configuration']

        # descent_curve: 3 + 1 = 4 dim
        self.descent_curve: LazyArray = dic['descent_curve']

        # state_table summary each state in a struct scalar: 3 dim
        self.state_table: np.ndarray = dic['state_table']

        self.particle_shape_table: np.ndarray = dic['particle_shape_table']
        self.simulation_table: np.ndarray = dic['simulation_table']

        self.summary = struct_array_to_dataframe(self.simulation_table)

    def __repr__(self):
        return str(self.summary)

    def property(self, prop: str) -> np.ndarray:
        """
        :param prop: name of recorded property
        :return: 3 dim tensor
        """
        return self.state_table[prop]

    def apply(self, func_act_on_configuration, num_threads=1):
        """
        :param func_act_on_configuration: (abg: (3,), configuration: (N, 3)) -> scalar
        This function loads all configuration data. Mind your memory!
        """
        data = self.configuration[:]
        abg = np.stack((self.property('A'), self.property('B'), self.property('gamma')), axis=3)
        shape_3d = data.shape[:3]
        xyt = data.reshape(-1, data.shape[-2], data.shape[-1])
        abg = abg.reshape(-1, 3)
        results = ut.Map(num_threads)(func_act_on_configuration, list(zip(abg, xyt)))
        if isinstance(results[0], np.ndarray):
            result_array = np.stack(results).reshape(*shape_3d, *results[0].shape)
        else:
            result_array = np.array(results).reshape(shape_3d)
        return result_array

    def descent_curve_at(self, i: int, j: int) -> np.ndarray:
        """
        :return: 2d array, a set of normalized descent curves of one simulation.
        """
        dc = self.descent_curve[i, j, :, :]
        return dc / dc[:, 0:1]
