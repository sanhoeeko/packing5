import os
import re
import threading

import h5py
import numpy as np

from . import h5tools as ht
from .h5tools import pack_h5_files, stack_h5_datasets
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


class SimulationData(Dataset):
    def __init__(self, file_name: str, metadata: np.ndarray, summary_dtype: list[tuple], summary_table_name: str,
                 descent_curve_size: int):
        super().__init__(file_name, metadata, summary_dtype, summary_table_name)
        self.N = metadata[0]['N']
        self.descent_curve_size = descent_curve_size
        with h5py.File(self.file_name, 'a') as f:
            f.create_dataset('configuration', shape=(0, self.N, 3), maxshape=(None, self.N, 3), chunks=True)
            f.create_dataset('mean_gradient_curve', shape=(0, descent_curve_size), maxshape=(None, descent_curve_size),
                             chunks=True)
            f.create_dataset('max_gradient_curve', shape=(0, descent_curve_size), maxshape=(None, descent_curve_size),
                             chunks=True)
            f.create_dataset('energy_curve', shape=(0, descent_curve_size), maxshape=(None, descent_curve_size),
                             chunks=True)

    def append(self, summary: np.ndarray, data: dict):
        self.append_summary(summary)
        ht.append_dict_to_hdf5_head(self.file_name, data)


def isFilenameRegular(file_name: str) -> bool:
    return not (file_name.startswith('data') or file_name.startswith('analysis'))


def pack_simulations_cwd(file_name='data.h5', truncate=False):
    pattern = re.compile(r'^(.*?)_\d+\.h5$')

    def get_ensemble_names(files):
        res = []
        for file in files:
            match = pattern.match(file)
            if match: res.append(match.group(1))
        return set(res)

    if os.path.exists(file_name):
        raise FileExistsError('Target file already exists!')

    # pack into ensembles
    files = [file for file in os.listdir() if pattern.match(file) and isFilenameRegular(file)]
    ensemble_ids = get_ensemble_names(files)
    for eid in ensemble_ids:
        stack_h5_datasets(eid, truncate)
    # for file in files: os.remove(file)

    # pack into single file
    sum_files = [file for file in os.listdir() if
                 file.endswith('.h5') and file not in files and isFilenameRegular(file)]
    pack_h5_files(sum_files, file_name)
    for file in sum_files: os.remove(file)
    compress_file(file_name)


def auto_pack():
    if not os.path.exists('data.h5'):
        pack_simulations_cwd(truncate=True)


def compress_file(file_name) -> (int, int):
    """
    :return: original_size (KB), compressed_size (KB)
    """
    temp_file_name = f"temp_{threading.get_ident()}_{randomString()}.h5"
    ht.compress_hdf5_file(file_name, temp_file_name)
    original_size = os.path.getsize(file_name) // 1024
    compressed_size = os.path.getsize(temp_file_name) // 1024
    os.remove(file_name)
    os.rename(temp_file_name, file_name)
    return original_size, compressed_size
