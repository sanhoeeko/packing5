"""
analysis.database: Data Access Layer
"""
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pandas as pd

import default
from . import utils as ut, mymath as mm

ut.setWorkingDirectory()

from simulation.state import State

from .h5tools import extract_metadata, struct_array_to_dataframe, filter_dataframe
from .orders import general_order_parameter, Delaunay
from .voronoi import Voronoi


class DatabaseBase:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.file = h5py.File(self.file_name, 'r')
        self.summary_table_array = extract_metadata(self.file_name)
        self.ids = self.summary_table_array['id'].tolist()
        self.summary = self.process_summary(struct_array_to_dataframe(self.summary_table_array))

    def __repr__(self):
        return self.summary.to_string()

    def __getitem__(self, index: int):
        return self.id(self.ids[index])

    def __iter__(self):
        """
        We don't load all data into memory. Here we use some transformation to "sort" data by [gamma]
        """
        if hasattr(self, 'summary'):
            for index, row in self.summary.iterrows():
                ensemble_id = row['id']
                obj = self.id(ensemble_id)
                yield obj
                del obj  # Explicitly delete the PickledEnsemble object to release memory
        else:
            for ensemble_id in self.ids:
                obj = self.id(ensemble_id)
                yield obj
                del obj  # Explicitly delete the PickledEnsemble object to release memory

    def id(self, ensemble_id: str):
        raise NotImplementedError  # to be inherited

    def process_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        lens = [x.n_density for x in self]
        df.insert(1, 'n_states', np.array(lens))
        df = df.sort_values(by=['gamma'])
        df.reset_index(drop=True, inplace=True)
        return df


class Database(DatabaseBase):
    def __init__(self, file_name: str):
        super().__init__(file_name)

    def id(self, ensemble_id: str):
        return PickledEnsemble(self.file[ensemble_id])

    def apply(self, func):
        """
        :param func: function act on an ensemble
        :return: tuple of lists: [result 1 for ensembles], [result 2 for ensembles], ...
        """
        result = list(map(func, self))
        return tuple(zip(*result))

    def subSummary(self, **kwargs) -> pd.DataFrame:
        """
        :param kwargs: key=value
        :return: ensembles whose key is value
        """
        df = self.summary
        for k, v in kwargs.items():
            df = filter_dataframe(df, k, v)
        return df

    def find(self, **kwargs) -> list['PickledEnsemble']:
        df = self.subSummary(**kwargs)
        print(df)
        ensemble_names = df['id']
        return [self.id(name) for name in ensemble_names]

    def search_max_gradient(self):
        gs = [e.max_gradient() for e in self]
        idx = np.argmax(gs)
        print(f"Maximum gradient: {gs[idx]}, at ensemble {self.ids[idx]}")


class PickledEnsemble:
    def __init__(self, h5_group):
        self.configuration = h5_group['configuration']  # shape: (replica, rho, N, 3)
        self.gradient_curve = h5_group['gradient_curve']  # shape: (replica, rho, m)
        self.energy_curve = h5_group['energy_curve']  # shape: (replica, rho, m)
        self.state_table = h5_group['state_table']  # struct array, shape: (replica, rho)
        self.metadata = h5_group.attrs['metadata']
        self.n_replica = self.state_table.shape[0]
        self.n_density = self.state_table.shape[1]

    def __len__(self):
        return self.n_replica

    def __iter__(self):
        for i in range(len(self)):
            obj = self.simulation_at(i)
            yield obj
            del obj  # Explicitly delete the PickledEnsemble object to release memory

    def __getitem__(self, index: int):
        return self.simulation_at(index)

    def simulation_at(self, nth_replica: int):
        return PickledSimulation(self.metadata, self.state_table[nth_replica], self.gradient_curve[nth_replica],
                                 self.energy_curve[nth_replica], self.configuration[nth_replica])

    @property
    def normalized_gradient_amp(self):
        """
        :return: g / n^2, where n is the number of disks. g is approximately proportional to n^2.
        """
        return self.state_table['gradient_amp'] / self.metadata['n'] ** 2

    def property(self, prop: str) -> np.ndarray:
        """
        :param prop: name of recorded property
        :return: 2 dim tensor
        """
        try:
            return self.state_table[prop]
        except ValueError:
            return getattr(self, prop)

    def max_gradient(self):
        if self.state_table.shape[1] == 0: return 0
        return np.max(self.property('gradient_amp'))

    def apply(self, func_act_on_configuration, num_threads=1, from_to_nth_data=None):
        """
        :param func_act_on_configuration: (abg: (3,), configuration: (N, 3)) -> scalar
        :param from_to_nth_data: tuple of int, if data of too low or too high density is unneeded.
        This function loads all configuration data. Mind your memory!
        """
        if from_to_nth_data is None:
            data = self.configuration[:]
            abg = np.stack((self.property('A'), self.property('B'), self.property('gamma')), axis=2)
        else:
            data = self.configuration[:, :, from_to_nth_data[0]:from_to_nth_data[1], :, :]
            abg = np.stack((self.property('A'), self.property('B'), self.property('gamma')), axis=2)
            abg = abg[:, :, from_to_nth_data[0]:from_to_nth_data[1], :]
        shape_3d = data.shape[:2]
        xyt = data.reshape(-1, data.shape[-2], data.shape[-1])
        abg = abg.reshape(-1, 3)
        results = ut.Map(num_threads)(func_act_on_configuration, list(zip(abg, xyt)))
        if isinstance(results[0], np.ndarray):
            result_array = np.stack(results).reshape(*shape_3d, *results[0].shape)
        else:
            result_array = np.array(results).reshape(shape_3d)
        return result_array

    def illegalMap(self) -> np.ndarray[np.int32]:
        print('here')

        def inner(i, j):
            xyt = ut.CArray(self.configuration[i, j])
            meta = self.state_table[i, j]
            return i, j, mm.isParticleTooClose(xyt) or mm.isParticleOutOfBoundary(xyt, meta['A'], meta['B'])

        mask = np.zeros((self.n_replica, self.n_density), np.int32)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(inner, i, j) for i in range(self.n_replica) for j in
                       range(self.n_density)]
            for future in futures:
                i, j, result = future.result()
                mask[i, j] = result

        return mask


class PickledSimulation:
    def __init__(self, metadata: np.ndarray, state_info: np.ndarray, gradient_curve: np.ndarray,
                 energy_curve: np.ndarray, xyt: np.ndarray):
        self.n = ut.actual_length_of_1d_array(state_info)
        self.metadata = ut.struct_to_dict(metadata)
        # clip nan data
        self.state_info = state_info[:self.n]
        self.gradient_curve = gradient_curve[:self.n, :]
        self.energy_curve = energy_curve[:self.n, :]
        self.xyt = xyt[:self.n, :, :]

    def __len__(self):
        return self.n

    def __getitem__(self, idx) -> dict:
        state_info = ut.struct_to_dict(self.state_info[idx])
        return {
            'metadata': {**self.metadata, **state_info},
            'gradient_curve': self.gradient_curve[idx, :],
            'energy_curve': self.energy_curve[idx, :],
            'xyt': self.xyt[idx, :, :]
        }

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def voronoi(self, idx):
        return Voronoi.fromStateDict(self[idx])

    def op(self, order_parameter_name: str) -> np.ndarray:
        """
        :return: numpy array of order parameter
        """
        return np.vectorize(self.op_at(order_parameter_name))(range(len(self)))

    def op_at(self, order_parameter_name: str):
        def inner(index: int):
            state = self[index]
            if default.if_using_legacy_delaunay:
                voro = Delaunay.legacy(state['metadata']['N'], state['metadata']['gamma'], ut.CArray(state['xyt']))
            else:
                voro = Voronoi.fromStateDict(state).delaunay(False)
                if voro is None: return np.float32(np.nan)
            return np.mean(general_order_parameter(
                order_parameter_name, state['xyt'], voro,
                (state['metadata']['A'], state['metadata']['B'], state['metadata']['gamma'])
            ))

        return inner

    def state_at(self, idx: int) -> State:
        """
        :param idx: can be negative. E.g., if it is -1, return the last state.
        """
        info = self.state_info[idx]
        xyt4 = np.zeros((self.metadata['N'], 4))
        xyt4[:, :3] = self.xyt[idx, :, :]
        return State(self.metadata['N'], self.metadata['n'], self.metadata['d'], info['A'], info['B'],
                     configuration=xyt4)

    def energyCurve(self):
        """
        :return: 2d array, a set of normalized descent curves of one simulation.
        """
        assert self.metadata['if_cal_energy']
        return self.energy_curve / self.energy_curve[:, 0:1]

    def gradientCurve(self) -> np.ndarray:
        return self.gradient_curve

    def stateDistance(self) -> np.ndarray:
        """
        :return: 1d array, showing how much the state variates during compression.
        """
        diff_xyt = np.diff(self.xyt, axis=0)
        return np.linalg.norm(diff_xyt, axis=(1, 2)) / np.sqrt(self.metadata['N'])
