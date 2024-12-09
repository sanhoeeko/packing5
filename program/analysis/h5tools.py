import h5py
import numpy as np
import pandas as pd


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


def read_hdf5_to_dict_lazy(file_path: str) -> dict:
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


def dict_to_analysis_hdf5(file_name: str, data_dict: dict):
    """
    :param data_dict: key: name of order parameter; value: tuple of (mean, ci)
    """
    with h5py.File(file_name, 'w') as hdf5_file:
        for key, value in data_dict.items():
            if isinstance(value, tuple):
                group = hdf5_file.create_group(key)
                group.create_dataset('mean', data=value[0], dtype=np.float32)
                group.create_dataset('ci', data=value[1], dtype=np.float32)
            else:
                hdf5_file.create_dataset(key, data=value, dtype=np.float32)


def add_array_to_hdf5(file_name: str, name: str, data: np.ndarray):
    with h5py.File(file_name, 'a') as hdf5_file:
        hdf5_file.create_dataset(name, data=data, dtype=data.dtype)


def invalid_value_of(array: np.ndarray):
    if array.dtype.fields is not None:
        return _get_struct_invalid_value(array.dtype)
    else:
        return _get_invalid_value(array.dtype)


def _get_struct_invalid_value(dtype):
    """
    Generate an invalid value for a given numpy struct dtype.

    Parameters:
    dtype (np.dtype): The numpy struct dtype.

    Returns:
    tuple: A tuple containing invalid values for each field in the struct.
    """
    invalid_value = [_get_invalid_value(dtype.fields[field][0]) for field in dtype.fields]
    return np.array([tuple(invalid_value)], dtype=dtype)


def _get_invalid_value(field_type):
    if np.issubdtype(field_type, np.integer):
        return np.int32(-1) if field_type == np.int32 else np.int64(-1)
    elif np.issubdtype(field_type, np.floating):
        return np.float32(np.nan) if field_type == np.float32 else np.nan
    elif np.issubdtype(field_type, np.str_) or np.issubdtype(field_type, np.bytes_):
        return '*' * field_type.itemsize
    else:
        raise ValueError(f"Unsupported field type: {field_type}")
