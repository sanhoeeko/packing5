"""
analysis.database: Data Access Layer
"""
import h5py
import numpy as np

from simulation.state import State
from . import utils as ut
from .h5tools import extract_metadata, struct_array_to_dataframe


class Database:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.file = h5py.File(self.file_name, 'r')
        self.summary_table_array = extract_metadata(self.file_name)
        self.summary = struct_array_to_dataframe(self.summary_table_array)
        self.ids = self.summary['id'].tolist()

    def __repr__(self):
        return str(self.summary)

    def __getitem__(self, index: int):
        return self.id(self.ids[index])

    def __iter__(self):
        for ensemble_id in self.ids:
            obj = self.id(ensemble_id)
            yield obj
            del obj  # Explicitly delete the PickledEnsemble object to release memory

    def id(self, ensemble_id: str):
        return PickledEnsemble(self.file[ensemble_id])

    def apply(self, func):
        """
        :param func: function act on an ensemble
        :return: tuple of lists: [result 1 for ensembles], [result 2 for ensembles], ...
        """
        result = list(map(func, self))
        return tuple(zip(*result))


class PickledEnsemble:
    def __init__(self, h5_group):
        self.configuration = h5_group['configuration']  # shape: (replica, rho, N, 3)
        self.descent_curve = h5_group['descent_curve']  # shape: (replica, rho, m)
        self.state_table = h5_group['state_table']  # struct array, shape: (replica, rho)
        self.metadata = h5_group.attrs['metadata']

    def __len__(self):
        return self.state_table.shape[0]

    def simulation_at(self, nth_replica: int):
        return PickledSimulation(self.metadata, self.state_table[nth_replica], self.descent_curve[nth_replica],
                                 self.configuration[nth_replica])

    def property(self, prop: str) -> np.ndarray:
        """
        :param prop: name of recorded property
        :return: 3 dim tensor
        """
        return self.state_table[prop]

    def apply(self, func_act_on_configuration, num_threads=1, from_to_nth_data=None):
        """
        :param func_act_on_configuration: (abg: (3,), configuration: (N, 3)) -> scalar
        :param from_to_nth_data: tuple of int, if data of too low or too high density is unneeded.
        This function loads all configuration data. Mind your memory!
        """
        if from_to_nth_data is None:
            data = self.configuration[:]
            abg = np.stack((self.property('A'), self.property('B'), self.property('gamma')), axis=3)
        else:
            data = self.configuration[:, :, from_to_nth_data[0]:from_to_nth_data[1], :, :]
            abg = np.stack((self.property('A'), self.property('B'), self.property('gamma')), axis=3)
            abg = abg[:, :, from_to_nth_data[0]:from_to_nth_data[1], :]
        shape_3d = data.shape[:3]
        xyt = data.reshape(-1, data.shape[-2], data.shape[-1])
        abg = abg.reshape(-1, 3)
        results = ut.Map(num_threads)(func_act_on_configuration, list(zip(abg, xyt)))
        if isinstance(results[0], np.ndarray):
            result_array = np.stack(results).reshape(*shape_3d, *results[0].shape)
        else:
            result_array = np.array(results).reshape(shape_3d)
        return result_array


class PickledSimulation:
    def __init__(self, metadata: np.ndarray, state_info: np.ndarray, descent_curve: np.ndarray, xyt: np.ndarray):
        self.n = ut.actual_length_of_1d_array(state_info)
        self.metadata = ut.struct_to_dict(metadata)
        # clip nan data
        self.state_info = state_info[:self.n]
        self.descent_curve = descent_curve[:self.n, :]
        self.xyt = xyt[:self.n, :, :]

    def __len__(self):
        return self.n

    def __getitem__(self, idx) -> dict:
        state_info = ut.struct_to_dict(self.state_info[idx])
        return {
            'metadata': {**self.metadata, **state_info},
            'descent_curve': self.descent_curve[idx, :],
            'xyt': self.xyt[idx, :, :]
        }

    def state_at(self, idx: int) -> State:
        """
        :param idx: can be negative. E.g., if it is -1, return the last state.
        """
        info = self.state_info[idx]
        xyt4 = np.zeros((self.metadata['N'], 4))
        xyt4[:, :3] = self.xyt[idx, :, :]
        return State(self.metadata['N'], self.metadata['n'], self.metadata['d'], info['A'], info['B'],
                     configuration=xyt4)

    def normalizedDescentCurve(self) -> np.ndarray:
        """
        :return: 2d array, a set of normalized descent curves of one simulation.
        """
        if self.metadata['if_cal_energy']:
            return self.descent_curve / self.descent_curve[:, 0:1]
        else:
            return self.descent_curve

    def stateDistance(self) -> np.ndarray:
        """
        :return: 1d array, showing how much the state variates during compression.
        """
        diff_xyt = np.diff(self.xyt, axis=0)
        return np.linalg.norm(diff_xyt, axis=(1, 2)) / np.sqrt(self.metadata['N'])
