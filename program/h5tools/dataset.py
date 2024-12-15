import os
import threading

import h5py
import numpy as np

from . import h5tools as ht
from .utils import randomString


class Dataset:
    """
    Automatically initializes metadata and summary table.
    Provides methods to read and write metadata and summary table.
    set `summary_dtype=None` to disable summary table (and create it later by other functions like `merge_dicts`).
    """

    def __init__(self, file_name: str, metadata: np.ndarray, summary_dtype: list[tuple], summary_table_name: str):
        self.file_name = file_name
        self.summary_dtype = summary_dtype
        self.summary_table_name = summary_table_name
        with h5py.File(self.file_name, 'w') as f:
            f.attrs['metadata'] = metadata
            if summary_dtype is not None:
                f.create_dataset(summary_table_name, dtype=summary_dtype, shape=(0,), maxshape=(None,), chunks=True)

    def append_summary(self, metadata: np.ndarray):
        with h5py.File(self.file_name, 'a') as f:
            dset = f[self.summary_table_name]
            dset.resize(dset.shape[0] + 1, axis=0)
            dset[-1] = metadata[0]

    def read_metadata(self) -> np.ndarray:
        return ht.read_metadata_to_struct(self.file_name)

    def read_data(self) -> dict:
        return ht.read_hdf5_to_dict(self.file_name)

    def rename(self, new_file_name: str):
        os.rename(self.file_name, new_file_name)
        self.file_name = new_file_name

    def compress_file(self) -> (int, int):
        """
        :return: original_size (KB), compressed_size (KB)
        """
        temp_file_name = f"temp_{threading.get_ident()}_{randomString()}.h5"
        ht.compress_hdf5_file(self.file_name, temp_file_name)
        original_size = os.path.getsize(self.file_name) // 1024
        compressed_size = os.path.getsize(temp_file_name) // 1024
        os.remove(self.file_name)
        os.rename(temp_file_name, self.file_name)
        return original_size, compressed_size


class SimulationData(Dataset):
    def __init__(self, file_name: str, metadata: np.ndarray, summary_dtype: list[tuple], summary_table_name: str,
                 descent_curve_size: int):
        super().__init__(file_name, metadata, summary_dtype, summary_table_name)
        N = metadata[0]['N']
        with h5py.File(self.file_name, 'a') as f:
            f.create_dataset('configuration', shape=(0, N, 3), maxshape=(None, N, 3), chunks=True)
            f.create_dataset('descent_curve', shape=(0, descent_curve_size), maxshape=(None, descent_curve_size),
                             chunks=True)

    def append(self, summary: np.ndarray, data: dict):
        self.append_summary(summary)
        ht.append_dict_to_hdf5_head(self.file_name, data)


class ExperimentData(Dataset):
    def __init__(self, file_name: str, metadata: np.ndarray, summary_dtype: list[tuple], summary_table_name: str):
        super().__init__(file_name, metadata, summary_dtype, summary_table_name)

    def write(self, data: dict):
        ht.write_dict_to_hdf5(self.file_name, data)


def package_simulations_into_experiment(
        file_name: str,
        summary_table_name: str,
        experiment_metadata: np.ndarray,
        simulations: list[Dataset]) -> ExperimentData:
    """
    This function creates an integrated HDF5 file, but does not destroy old files.
    """

    # create ExperimentData object
    experiment_data = ExperimentData(file_name, experiment_metadata, None, None)

    # collect data
    dic = ht.merge_dicts([s.read_data() for s in simulations])

    # collect metadata
    metadata_list = np.array([s.read_metadata() for s in simulations])
    dic[summary_table_name] = metadata_list

    # write to disk
    experiment_data.write(dic)
    return experiment_data


def package_simulations_cwd(file_name='data.h5', summary_table_name='simulation_table'):
    # collect h5 files
    folder = os.getcwd()
    files = os.listdir(folder)
    h5_files = [os.path.abspath(os.path.join(folder, file)) for file in files if
                file.endswith('.h5') and file not in ['data.h5', file_name]]

    # collect data
    dic = ht.merge_dicts([ht.read_hdf5_to_dict(file) for file in h5_files])

    # collect metadata
    metadata_list = np.array([ht.read_metadata_to_struct(file) for file in h5_files])
    dic[summary_table_name] = metadata_list

    # write to disk
    ht.write_dict_to_hdf5(file_name, dic)

    # clear (unsafe)
    # for file in h5_files:
    #     os.remove(file)
