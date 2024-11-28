import h5py
import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, file_name: str, metadata: list[tuple]):
        self.file_name = file_name
        with h5py.File(self.file_name, 'w') as f:
            for key, value in metadata:
                f.attrs[key] = value
            f.create_dataset('summary_table', data=np.empty((0, len(metadata))), maxshape=(None, len(metadata)))

    def append_metadata(self, metadata):
        with h5py.File(self.file_name, 'a') as f:
            summary_table = f['summary_table']
            new_row = np.array(metadata, dtype=np.float32).reshape(1, -1)
            summary_table.resize(summary_table.shape[0] + 1, axis=0)
            summary_table[-1, :] = new_row

    def read_summary_table(self):
        with h5py.File(self.file_name, 'r') as f:
            summary_table = f['summary_table'][:]
            column_names = list(f.attrs.keys())
        return pd.DataFrame(summary_table, columns=column_names)


class SimulationData(Dataset):
    def __init__(self, file_name: str, metadata: list[tuple], descent_curve_size: int):
        super().__init__(file_name, metadata)
        meta_dict = dict(metadata)
        N = meta_dict['N']
        with h5py.File(self.file_name, 'a') as f:
            f.create_dataset('configuration', shape=(0, N, 3), maxshape=(None, N, 3), chunks=True)
            f.create_dataset('descent_curve', shape=(0, descent_curve_size), maxshape=(None, descent_curve_size),
                             chunks=True)

    def append(self, metadata, data: dict):
        self.append_metadata(metadata)
        with h5py.File(self.file_name, 'a') as f:
            for key, value in data.items():
                if key in f:
                    dset = f[key]
                    dset.resize(dset.shape[0] + 1, axis=0)
                    dset[-1, :] = value
                else:
                    raise KeyError(f"Dataset {key} not found in HDF5 file.")


class ExperimentData(Dataset):
    def __init__(self, file_name, metadata):
        super().__init__(file_name, metadata)
        with h5py.File(self.file_name, 'a') as f:
            f.create_dataset('simulation_data', shape=(0,), dtype=h5py.vlen_dtype(h5py.special_dtype(vlen=bytes)),
                             chunks=True)

    def append(self, simulation_data):
        metadata = list(simulation_data.read_summary_table().iloc[-1])
        self.append_metadata(metadata)
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


def package_simulations_into_experiment(file_name, experiment_metadata, simulations):
    experiment_data = ExperimentData(file_name, experiment_metadata)
    for sim_data in simulations:
        experiment_data.append(sim_data)
    return experiment_data
