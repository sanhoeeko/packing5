"""
analysis.database: Data Access Layer
"""
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pandas as pd

from . import utils as ut, mymath as mm
from .mask import parse_mask_expr

ut.setWorkingDirectory()

from simulation.state import State

from .h5tools import extract_metadata, struct_array_to_dataframe, filter_dataframe
from .voronoi import Voronoi


class DatabaseBase:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.file = h5py.File(self.file_name, 'r', locking=False, libver='latest')
        self._summary_table_array = extract_metadata(self.file_name)
        self.ids = self._summary_table_array['id'].tolist()
        self.summary = self.process_summary(struct_array_to_dataframe(self._summary_table_array))

    def __repr__(self):
        return self.summary.to_string()

    def __getitem__(self, index: int):
        return self.id(self.ids[index])

    def __len__(self):
        return len(self.summary)

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
        print(df.to_string())
        ensemble_names = df['id']
        return [self.id(name) for name in ensemble_names]

    def search_max(self, flag: str, normalization_power_of_n=0):
        """
        flag: mean_gradient_amp | max_gradient_amp | max_force | max_torque
        """
        indices, gs = zip(*[e.max_gradient(flag, normalization_power_of_n) for e in self])
        idx = np.argmax(gs)
        print(
            f"Maximum {flag}: {gs[idx]}, at ensemble {self.summary.iloc[idx].id}, data_index={indices[idx]}. "
            f"normalization_power_of_n={normalization_power_of_n}"
        )

    def search_max_gradient(self):
        self.search_max('mean_gradient_amp', normalization_power_of_n=0)
        self.search_max('max_gradient_amp', normalization_power_of_n=0)
        self.search_max('max_force', normalization_power_of_n=0)
        self.search_max('max_torque', normalization_power_of_n=2)


class PickledEnsemble:
    def __init__(self, h5_group):
        self.configuration = h5_group['configuration']  # shape: (replica, rho, N, 3)
        self.mean_gradient_curve = h5_group['mean_gradient_curve']  # shape: (replica, rho, m)
        self.max_gradient_curve = h5_group['max_gradient_curve']  # shape: (replica, rho, m)
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
        return PickledSimulation(
            self.metadata, self.state_table[nth_replica],
            self.mean_gradient_curve[nth_replica],
            self.max_gradient_curve[nth_replica],
            self.energy_curve[nth_replica],
            self.configuration[nth_replica]
        )

    def index_at(self, index: int) -> list[dict]:
        return [e[index] for e in self]

    @property
    def normalized_gradient_amp(self):
        """
        :return: g / n^2, where n is the number of disks. g is approximately proportional to n^2.
        """
        return self.state_table['mean_gradient_amp'] / self.metadata['n'] ** 2

    def property(self, prop: str) -> np.ndarray:
        """
        :param prop: name of recorded property
        :return: 2 dim tensor
        """
        try:
            return self.state_table[prop]
        except ValueError:
            return getattr(self, prop)

    def max_gradient(self, flag: str, normalization_power_of_n=0) -> ((int, int), np.float32):
        """
        flag: mean_gradient_amp | max_gradient_amp | max_force | max_torque
        :return: (max_indices, max_gradient_value)
        """
        if self.state_table.shape[1] == 0: return ((0, 0), 0)
        data = self.property(flag)
        idx = int(np.argmax(data))
        i, j = idx // data.shape[1], idx % data.shape[1]
        return (i, j), data[i, j] / self.metadata['n'] ** normalization_power_of_n

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
    def __init__(self, metadata: np.ndarray, state_info: np.ndarray, mean_gradient_curve: np.ndarray,
                 max_gradient_curve: np.ndarray, energy_curve: np.ndarray, xyt: np.ndarray):
        self.n = ut.actual_length_of_1d_array(state_info)
        self.metadata = ut.struct_to_dict(metadata)
        # clip nan data
        self.state_info = state_info[:self.n]
        self.mean_gradient_curve = mean_gradient_curve[:self.n, :]
        self.max_gradient_curve = max_gradient_curve[:self.n, :]
        self.energy_curve = energy_curve[:self.n, :]
        self.xyt = xyt[:self.n, :, :]

    def __len__(self):
        return self.n

    def __getitem__(self, idx) -> dict:
        state_info = ut.struct_to_dict(self.state_info[idx])
        return {
            'metadata': {**self.metadata, **state_info},
            'mean_gradient_curve': self.mean_gradient_curve[idx, :],
            'max_gradient_curve': self.max_gradient_curve[idx, :],
            'energy_curve': self.energy_curve[idx, :],
            'xyt': self.xyt[idx, :, :]
        }

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def op(self, order_parameter_name: str, num_threads=1, upper_h: float = None, upper_phi: float = None,
           option='None') -> np.ndarray:
        """
        :return: numpy array of order parameter
        """
        _, upper_index = ut.indexInterval(self.state_info['phi'], self.metadata['gamma'],
                                          None, upper_h, upper_phi)
        if num_threads != 1:
            with multiprocessing.Pool(processes=num_threads) as pool:
                result = pool.map(self.op_at_wrapper, [(order_parameter_name, i, option) for i in range(upper_index)])
            return np.array(result)
        else:
            return np.vectorize(self.op_at(order_parameter_name, option))(range(upper_index))

    def op_at_wrapper(self, args):
        """
        Wrapper function to unpack arguments for parallel processing.
        """
        order_parameter_name, index, option = args
        if order_parameter_name in ['AngleDist']:
            return self.irregular_op_at(order_parameter_name, option)(index)
        else:
            return self.op_at(order_parameter_name, option)(index)

    def op_at(self, order_parameter_name: str, option_and_mask='None'):
        from .analysis import OrderParameterFunc

        parts = option_and_mask.split(',')
        option = parts[0]
        if len(parts) == 1:
            mask = None
        elif len(parts) == 2:
            mask_expr = parts[1].strip()
            try:
                mask = parse_mask_expr(mask_expr)
            except ValueError as e:
                raise e
            except Exception as e:
                raise ValueError(f"Invalid mask expression '{mask_expr}': {str(e)}")
        else:
            raise ValueError(f"Invalid option_and_mask format: {option_and_mask}")

        def inner(index: int):
            state = self[index]
            result_struct_array = OrderParameterFunc([order_parameter_name], option, mask)(
                (
                    (state['metadata']['A'], state['metadata']['B'], state['metadata']['gamma']),
                    state['xyt'],
                )
            )
            return result_struct_array[order_parameter_name]

        return inner

    def irregular_op_at(self, order_parameter_name: str, option='None'):
        from .orders import StaticOrders

        if order_parameter_name == 'AngleDist':
            def inner(index: int):
                state = self[index]
                return StaticOrders.AngleDist(state['xyt'])
        else:
            raise NotImplementedError

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

    def voronoi_at(self, idx):
        return Voronoi.fromStateDict(self[idx])

    def energyCurve(self):
        """
        :return: 2d array, a set of normalized descent curves of one simulation.
        """
        assert self.metadata['if_cal_energy']
        return self.energy_curve / self.energy_curve[:, 0:1]

    def meanGradientCurve(self) -> np.ndarray:
        return self.mean_gradient_curve

    def maxGradientCurve(self) -> np.ndarray:
        return self.max_gradient_curve

    def indexInterval(self, phi_c: float = None, upper_h: float = None):
        return ut.indexInterval(self.state_info['phi'], self.metadata['gamma'], phi_c, upper_h)

    def propertyInterval(self, name: str, phi_c=None, upper_h=None):
        from_, to_ = self.indexInterval(phi_c, upper_h)
        return self.state_info[name][from_:to_]

    def stateDistance(self, phi_c=None, upper_h=None) -> np.ndarray:
        """
        :return: 1d array, showing how much the state variates during compression.
        """
        from_, to_ = self.indexInterval(phi_c, upper_h)
        diff_xyt = np.diff(self.xyt[from_:to_, :, :], axis=0)
        return np.linalg.norm(diff_xyt, axis=(1, 2)) / np.sqrt(self.metadata['N'])

    def delaunayAt(self, index: int):
        from .voronoi import Voronoi
        return Voronoi.fromStateDict(self[index]).delaunay()

    def get_delaunay_list(self, from_: int, to_: int, num_threads: int) -> list:
        if num_threads != 1:
            import multiprocessing
            with multiprocessing.Pool(processes=num_threads) as pool:
                delaunays = pool.map(self.delaunayAt, range(from_, to_))
        else:
            dics = [dic for dic in self][from_:to_]
            delaunays = [Voronoi.fromStateDict(dic).delaunay() for dic in dics]
        for d in delaunays:
            d.check()
        return delaunays

    def bondCreation(self, num_threads=1, phi_c=None, upper_h=None) -> np.ndarray:
        from_, to_ = self.indexInterval(phi_c, upper_h)
        delaunays = self.get_delaunay_list(from_, to_, num_threads)
        res = np.zeros((to_ - from_ - 1,), dtype=int)
        for i in range(len(delaunays) - 1):
            res[i] = delaunays[i + 1].difference(delaunays[i]).count()
        return res

    def eventStat(self, max_track_length: int, num_threads=1, phi_c=None, upper_h=None) -> np.ndarray[np.int32]:
        from_, to_ = self.indexInterval(phi_c, upper_h)
        delaunays = self.get_delaunay_list(from_, to_, num_threads)
        delaunays = [None] * from_ + delaunays  # shift the index
        res = np.zeros((max_track_length, to_ - from_ - 1), dtype=np.int32)
        for i in range(from_, to_ - 1):
            xyt_1 = ut.CArray(self[i]['xyt'])
            xyt_0 = ut.CArray(self[i + 1]['xyt'])
            events = delaunays[i + 1].events_compared_with(delaunays[i], xyt_1, xyt_0)
            for event in events:
                track_length = (event[0] + 1) // 2  # => ceil(event[0]/2)
                idx = min(track_length, max_track_length - 1)
                res[idx, i - from_] += 1
        return res

    def stableDefects(self, num_threads=1, phi_c=None, upper_h=None) -> np.ndarray:
        """
        :return: count of stable defects, defined by
            unchanged topological charge && topological defect && internal position
        """
        from_, to_ = self.indexInterval(phi_c, upper_h)
        delaunays = self.get_delaunay_list(from_, to_, num_threads)
        zs = [delaunay.z_number() for delaunay in delaunays]

        none_delaunays = [None] * from_ + delaunays  # shift the index
        bodies = []
        for i in range(from_, to_ - 1):
            xyt_c = ut.CArray(self[i]['xyt'])
            bodies.append((1 - none_delaunays[i].dist_hull(xyt_c)).astype(bool))

        res = np.zeros((to_ - from_ - 1,), dtype=int)
        for i in range(len(delaunays) - 1):
            mask = np.bitwise_and(zs[i + 1] - zs[i] == 0, zs[i] != 6, bodies[i])
            res[i] = np.sum(mask)
        return res
