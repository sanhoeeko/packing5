import h5py
import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, file_name: str, summary_dtype: list[tuple]):
        self.file_name = file_name
        self.summary_dtype = summary_dtype
        with h5py.File(self.file_name, 'w') as f:
            f.create_dataset('summary_table', dtype=summary_dtype, shape=(0,), maxshape=(None,), chunks=True)

    def append_summary(self, metadata: np.ndarray):
        with h5py.File(self.file_name, 'a') as f:
            dset = f['summary_table']
            dset.resize(dset.shape[0] + 1, axis=0)
            dset[-1] = metadata[0]

    def read_summary_table(self):
        with h5py.File(self.file_name, 'r') as f:
            dataset = f['summary_table']
            data = dataset[:]
        df = pd.DataFrame(data)
        # convert strings
        for column in df.columns:
            if np.issubdtype(df[column].dtype, np.object_):
                df[column] = df[column].str.decode('utf-8')
        return df


class SimulationData(Dataset):
    def __init__(self, file_name: str, metadata: np.ndarray, summary_dtype: list[tuple], descent_curve_size: int):
        super().__init__(file_name, summary_dtype)
        N = metadata[0]['N']
        with h5py.File(self.file_name, 'a') as f:
            f.attrs['metadata'] = metadata
            f.create_dataset('configuration', shape=(0, N, 3), maxshape=(None, N, 3), chunks=True)
            f.create_dataset('descent_curve', shape=(0, descent_curve_size), maxshape=(None, descent_curve_size),
                             chunks=True)

    def get_metadata(self) -> np.ndarray:
        with h5py.File(self.file_name, 'a') as f:
            return f.attrs['metadata']

    def append(self, summary: np.ndarray, data: dict):
        self.append_summary(summary)
        with h5py.File(self.file_name, 'a') as f:
            for key, value in data.items():
                if key in f:
                    dset = f[key]
                    dset.resize(dset.shape[0] + 1, axis=0)
                    dset[-1, :] = value
                else:
                    raise KeyError(f"Dataset {key} not found in HDF5 file.")

    def read_data(self):
        pass


class ExperimentData(Dataset):
    def __init__(self, file_name: str, metadata: np.ndarray, summary_dtype: list[tuple]):
        super().__init__(file_name, summary_dtype)
        with h5py.File(self.file_name, 'a') as f:
            f.attrs['metadata'] = metadata
            f.create_dataset('simulation_data', shape=(0,), dtype=h5py.vlen_dtype(h5py.special_dtype(vlen=bytes)),
                             chunks=True)

    def append(self, simulation_data: SimulationData):
        self.append_summary(simulation_data.get_metadata())
        with h5py.File(self.file_name, 'a') as f:
            sim_data_summary = simulation_data.read_summary_table().to_numpy()
            sim_data_dict = {}
            for key in ['configurations', 'descent_curves']:
                sim_data_dict[key] = simulation_data.read_data(key)
            sim_data_serialized = np.array(sim_data_summary, dtype=h5py.string_dtype())
            sim_data_serialized = np.append(sim_data_serialized, [sim_data_dict])
            f['simulation_data'].resize(f['simulation_data'].shape[0] + 1, axis=0)
            f['simulation_data'][-1] = sim_data_serialized.tobytes()

    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as f:
            metadata = f['summary_table'][index]
            sim_data_serialized = f['simulation_data'][index].tobytes()
            sim_data_summary, sim_data_dict = np.frombuffer(sim_data_serialized, dtype=object)
            temp_metadata = {key: value for key, value in zip(f.attrs.keys(), metadata)}
            simulation_data = SimulationData(file_name=None, metadata=temp_metadata)
            simulation_data.summary_table = sim_data_summary
            for key, value in sim_data_dict.items():
                setattr(simulation_data, key, value)
            return simulation_data


def write_metadata_to_hdf5(hdf5_filename: str, metadata: dict):
    with h5py.File(hdf5_filename, 'a') as f:
        for key, value in metadata.items():
            f.attrs[key] = value


def read_metadata_from_hdf5(hdf5_filename: str) -> dict:
    metadata = {}
    with h5py.File(hdf5_filename, 'r') as f:
        for key, value in f.attrs.items():
            metadata[key] = value
    return metadata


def package_simulations_into_experiment(file_name: str, experiment_metadata, simulations: list[SimulationData]):
    experiment_data = ExperimentData(file_name, experiment_metadata, simulations[0].get_metadata().dtype)
    for sim_data in simulations:
        experiment_data.append(sim_data)
    return experiment_data
