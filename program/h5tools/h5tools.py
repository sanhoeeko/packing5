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


def merge_dicts(dict_list: list[dict]):
    """
    Merges a list of dictionaries into a single dictionary. If multiple dictionaries have the same key:
    1. If the values are all np.ndarray, they are concatenated along a new axis (axis=0).
    2. If any value is not a np.ndarray, they are packed into a list as the new value.

    Parameters:
    dict_list (list of dict): List of dictionaries to be merged.

    Returns:
    dict: A merged dictionary with combined values.
    """
    merged_dict = {}
    arrays_to_stack = {}

    for d in dict_list:
        for key, value in d.items():
            if key in merged_dict:
                if isinstance(value, np.ndarray) and isinstance(merged_dict[key], np.ndarray):
                    # Collect arrays to stack later
                    if key not in arrays_to_stack:
                        arrays_to_stack[key] = [merged_dict[key]]
                    arrays_to_stack[key].append(value)
                else:
                    # Pack non-np.ndarray values into a list
                    if not isinstance(merged_dict[key], list):
                        merged_dict[key] = [merged_dict[key]]
                    merged_dict[key].append(value)
            else:
                # Add new key-value pair to the merged dictionary
                merged_dict[key] = value
                if isinstance(value, np.ndarray):
                    arrays_to_stack[key] = [value]

    # Stack collected arrays and update the merged dictionary
    for key, arrays in arrays_to_stack.items():
        merged_dict[key] = stack_and_fill_nan(arrays)

    return merged_dict


def stack_and_fill_nan(matrices: list[np.ndarray]):
    """
    Stack a list of matrices along a new dimension and fill with NaN.

    Parameters:
    matrices (list of np.ndarray): List of matrices, which may have misaligned dimensions.

    Returns:
    np.ndarray: Stacked matrices with NaN filling.
    """
    # Find the maximum size for each dimension
    max_shape = tuple(max(s) for s in zip(*[m.shape for m in matrices]))

    # Create a new dataset filled with NaN
    stacked_data = np.full((len(matrices), *max_shape), invalid_value_of(matrices[0]))

    for i, matrix in enumerate(matrices):
        slices = tuple(slice(0, s) for s in matrix.shape)
        stacked_data[i][slices] = matrix

    return stacked_data


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
