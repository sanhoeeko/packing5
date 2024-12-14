"""
analysis.database: Data Access Layer
"""

import numpy as np

from simulation.state import State
from . import utils as ut
from .h5tools import LazyArray, read_hdf5_to_dict_lazy, struct_array_to_dataframe


class Database:
    """
    The first three dimensions of data:
    1. for the same "external" conditions
    2. for the same shape of particles
    3. for one simulation
    """

    def __init__(self, file_name: str):
        self.file_name = file_name
        dic = read_hdf5_to_dict_lazy(file_name)

        # configuration: 3 + 2 = 5 dim
        self.configuration: LazyArray = dic['configuration']

        # descent_curve: 3 + 1 = 4 dim
        self.descent_curve: LazyArray = dic['descent_curve']

        # state_table summary each state in a struct scalar: 3 dim
        self.state_table: np.ndarray = dic['state_table']

        self.particle_shape_table: np.ndarray = dic['particle_shape_table']
        self.simulation_table: np.ndarray = dic['simulation_table']

        # shapes
        self.shape = self.configuration.shape[:3]
        self.m_groups, self.n_parallels, self.l_max_states = self.shape

        self.summary = struct_array_to_dataframe(self.simulation_table)

    def __repr__(self):
        return str(self.summary)

    def id(self, state_id: str):
        try:
            index = self.summary[self.summary['id'] == state_id].index.tolist()[0]
        except:
            raise KeyError("Simulation ID not found!")
        i = index // self.shape[1]
        j = index % self.shape[1]
        return self.simulation_at(i, j)

    def simulation_at(self, i: int, j: int):
        return PickledSimulation(
            self.simulation_table[i, j], self.state_table[i, j, :],
            self.descent_curve[i, j, :, :], self.configuration[i, j, :, :, :]
        )

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


class PickledSimulation:
    def __init__(self, metadata: np.ndarray, state_info: np.ndarray, descent_curve: np.ndarray, xyt: np.ndarray):
        self.n = ut.actual_length_of_1d_array(state_info)
        self.metadata = ut.struct_to_dict(metadata)
        # clip nan filled
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
        return self.descent_curve / self.descent_curve[:, 0:1]

    def stateDistance(self) -> np.ndarray:
        """
        :return: 1d array, showing how much the state variates during compression.
        """
        diff_xyt = np.diff(self.xyt, axis=0)
        return np.linalg.norm(diff_xyt, axis=(1, 2)) / np.sqrt(self.metadata['N'])
