import glob
import re

import h5py
import numpy as np


def read_hdf5_to_dict(file_path: str) -> dict:
    data_dict = {}
    with h5py.File(file_path, 'r') as file:
        for dataset_name in file:
            data_dict[dataset_name] = file[dataset_name][:]
    return data_dict


def read_metadata_to_struct(file_name: str) -> np.ndarray:
    with h5py.File(file_name, 'a') as f:
        try:
            return f.attrs['metadata']
        except KeyError:
            raise ValueError('This file has no metadata!')


def append_dict_to_hdf5_head(file_path: str, data: dict):
    """
    This method requires keys to exist in the HDF5 file.
    """
    with h5py.File(file_path, 'a') as f:
        for key, value in data.items():
            if key in f:
                dset = f[key]
                dset.resize(dset.shape[0] + 1, axis=0)
                dset[-1, :] = value
            else:
                raise KeyError(f"Dataset {key} not found in HDF5 file.")


def write_dict_to_hdf5(file_path: str, data: dict):
    """
    This method requires keys NOT to exist in the HDF5 file.
    """
    with h5py.File(file_path, 'a') as file:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                file.create_dataset(key, data=value)
            else:
                raise ValueError('Not a numpy array!')


def read_metadata_from_hdf5(hdf5_filename: str) -> dict:
    metadata = {}
    with h5py.File(hdf5_filename, 'r') as f:
        for key, value in f.attrs.items():
            metadata[key] = value
    return metadata


def write_metadata_to_hdf5(hdf5_filename: str, metadata: dict):
    with h5py.File(hdf5_filename, 'a') as f:
        for key, value in metadata.items():
            f.attrs[key] = value


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


def numerical_sort(value):
    parts = re.split(r'(\d+)', value)
    return [int(text) if text.isdigit() else text for text in parts]


def stack_h5_datasets(ensemble_id: str):
    file_pattern = f'{ensemble_id}_*.h5'
    output_file = f'{ensemble_id}.h5'

    # Find all files matching the pattern
    files = sorted(glob.glob(file_pattern), key=numerical_sort)

    if not files:
        raise FileNotFoundError(f"No files matching the pattern {file_pattern} found.")

    # Create the new file
    with h5py.File(output_file, 'w') as f_out:
        # Copy datasets and add a new dimension
        for file in files:
            with h5py.File(file, 'r') as f:
                for key, data in f.items():
                    if key not in f_out:
                        # Initialize dataset with a new dimension
                        shape = (len(files),) + data.shape
                        maxshape = (None,) + data.shape
                        f_out.create_dataset(key, shape=shape, maxshape=maxshape, dtype=f[key].dtype, chunks=True)
                    # Insert data into the appropriate slice
                    f_out[key][files.index(file), ...] = data[:]

        # Copy all attributes from the first file to the new file
        with h5py.File(files[0], 'r') as f_first:
            for key, value in f_first.attrs.items():
                f_out.attrs[key] = value


def pack_h5_files(files: list[str], output_filename: str):
    def copy_item(group, name, obj):
        # Copy datasets and attributes
        if isinstance(obj, h5py.Dataset):
            # Copy dataset
            group.create_dataset(name, data=obj[:])
        elif isinstance(obj, h5py.Group):
            # Recursively copy group
            sub_group = group.create_group(name)
            for sub_name, sub_obj in obj.items():
                copy_item(sub_group, sub_name, sub_obj)
        # Copy attributes
        for key, value in obj.attrs.items():
            group[name].attrs[key] = value

    with h5py.File(output_filename, 'w') as f_out:
        for file in files:
            group_name = file.split('.')[0]
            with h5py.File(file, 'r') as f_in:
                # Create a group in the output file named after the original file path
                group = f_out.create_group(group_name)
                for name, obj in f_in.items():
                    copy_item(group, name, obj)
                # Copy attributes
                for key, value in f_in.attrs.items():
                    group.attrs[key] = value


def compress_hdf5_file(input_file: str, output_file: str, compression='gzip', compression_opts=9):
    """
    Compress an existing HDF5 file and save it to a new file.
    input_file != output_file
    """

    def copy_attrs(source, dest):
        """
        Copy attributes from source to destination.
        """
        for key, value in source.attrs.items():
            dest.attrs[key] = value

    def copy_group(source, dest):
        """
        Recursively copy groups and datasets from source to destination.
        """
        for name, item in source.items():
            if isinstance(item, h5py.Group):
                group = dest.create_group(name)
                copy_attrs(item, group)
                copy_group(item, group)
            elif isinstance(item, h5py.Dataset):
                data = item[:]
                dataset = dest.create_dataset(name, data=data, compression=compression,
                                              compression_opts=compression_opts)
                copy_attrs(item, dataset)

    # input_file != output_file
    assert input_file != output_file, "Please using different file names."

    # Open the existing HDF5 file
    with h5py.File(input_file, 'r') as f_in:
        # Create a new compressed HDF5 file
        with h5py.File(output_file, 'w') as f_out:
            copy_attrs(f_in, f_out)
            copy_group(f_in, f_out)
