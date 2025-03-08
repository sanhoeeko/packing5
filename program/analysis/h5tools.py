import pandas as pd

from h5tools.h5tools import *


def struct_array_to_dataframe(data: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(data.reshape(-1))
    # convert strings
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.object_):
            df[column] = df[column].str.decode('utf-8')
    return df


def dict_to_analysis_hdf5(file_name: str, data_dict: dict):
    """
    :param data_dict: dict[dict[tuple]]:
    key: name of ensemble;
    value:
        key: name of order parameter;
        value: tuple of (mean, ci)
    """
    with h5py.File(file_name, 'w') as hdf5_file:
        for ensemble_name, ensemble_data in data_dict.items():
            ensemble_h5 = hdf5_file.create_group(ensemble_name)
            for key, value in ensemble_data.items():
                if isinstance(value, tuple):
                    group = ensemble_h5.create_group(key)
                    group.create_dataset('mean', data=value[0].astype(np.float32), dtype=np.float32)
                    group.create_dataset('ci', data=value[1].astype(np.float32), dtype=np.float32)
                else:
                    ensemble_h5.create_dataset(key, data=value, dtype=np.float32)


def add_array_to_hdf5(file_name: str, name: str, data: np.ndarray):
    with h5py.File(file_name, 'a') as hdf5_file:
        hdf5_file.create_dataset(name, data=data, dtype=data.dtype)


def add_property_to_hdf5(file_name: str, key: str, value):
    with h5py.File(file_name, 'a') as hdf5_file:
        hdf5_file.attrs[key] = value


def read_hdf5_groups_to_dicts(file_path: str) -> (dict, dict[dict]):
    data_dict = {}
    group_dict = {}
    with h5py.File(file_path, 'r') as file:
        for dataset_name in file:
            if hasattr(file[dataset_name], 'keys'):
                # if it is a group
                temp_dic = {}
                for key in file[dataset_name].keys():
                    temp_dic[key] = file[dataset_name][key][:]
                group_dict[dataset_name] = temp_dic
            else:
                if len(file[dataset_name].shape) == 0:
                    # if it is a scalar
                    data_dict[dataset_name] = file[dataset_name][()]
                else:
                    # if it is a dataset
                    data_dict[dataset_name] = file[dataset_name][:]
    return data_dict, group_dict


def extract_metadata(file_path: str) -> np.ndarray:
    metadata_list = []

    # Open the HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Iterate through all groups in the file
        def collect_metadata(name, obj):
            if isinstance(obj, h5py.Group):
                if 'metadata' in obj.attrs:
                    metadata_list.append(obj.attrs['metadata'])

        f.visititems(collect_metadata)

    # Combine all metadata into a single structured ndarray
    metadata_array = np.concatenate(metadata_list)
    return metadata_array


def filter_dataframe(df: pd.DataFrame, column_name, target_value):
    if pd.api.types.is_numeric_dtype(df[column_name]):
        # for float
        if np.issubdtype(df[column_name].dtype, np.floating):
            filtered_df = df[np.abs(df[column_name] - target_value) < 1e-6]
        # for int
        else:
            filtered_df = df[df[column_name] == target_value]
    elif pd.api.types.is_string_dtype(df[column_name]):
        # for string
        filtered_df = df[df[column_name] == target_value]
    else:
        raise ValueError(f"Unsupported data type for column '{column_name}'")

    return filtered_df
