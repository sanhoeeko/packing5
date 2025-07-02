import pickle as pkl
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from . import mymath as mm, utils as ut
from .database import Database, PickledEnsemble
from h5tools.h5analysis import dict_to_analysis_hdf5, add_array_to_hdf5, add_property_to_hdf5
from .kernel import ker
from .mymath import DirtyDataException
from .orders import OrderParameterList
from .voronoi import Voronoi


def OrderParameterFunc(order_parameter_list: list[str], option='None'):
    """
    parameters of inner function:
        abg = (A_upper_bound, B_upper_bound, gamma) for each state
        xyt = (N, 3) configuration
    option: 'None' | 'abs averaged' | 'only boundary' | 'only internal' | 'dist rank' | 'y rank'
            'dense 0' | 'dense 0|1' | ... <0 for sparse, 1 for dense, 2 for super-dense>
    :return: a function object for `Database.apply`
    """
    assert type(option) is str

    def inner(args) -> np.ndarray:
        abg: tuple = args[0]
        xyt: np.ndarray = args[1]
        Xi = OrderParameterList(order_parameter_list)(xyt, abg)
        if option == 'abs averaged':
            return ut.apply_struct(np.mean)(Xi)
        elif option in ['only boundary', 'only internal']:
            N = xyt.shape[0]
            phi = ut.phi(N, abg[2], abg[0], abg[1])
            mask = ut.InternalMask2(phi, abg[0], abg[1], xyt)
            if option == 'only boundary':
                mask = ~mask
            N_valid = np.sum(mask)
            return ut.apply_struct(lambda x: np.sum(x) / N_valid)(ut.mask_structured_array(Xi, mask))
        elif option == 'dist rank':
            # slow, because it calls Voronoi twice
            xyt_c = ut.CArray(xyt, dtype=np.float32)
            voro = Voronoi(abg[2], abg[0], abg[1], xyt_c.data).delaunay()
            mask = voro.SegmentDistRankMask(xyt_c)
            N_valid = np.sum(mask)
            return ut.apply_struct(lambda x: np.sum(x) / N_valid)(ut.mask_structured_array(Xi, mask))
        elif option in ['y rank', '~y rank']:
            N = xyt.shape[0]
            phi = ut.phi(N, abg[2], abg[0], abg[1])
            mask = ut.y_rank(N, phi, xyt)
            # mask = ut.y_rank_2(N, phi, abg[0], xyt)
            if option == '~y rank':
                mask = ~mask
            N_valid = np.sum(mask)
            return ut.apply_struct(lambda x: np.sum(x) / N_valid)(ut.mask_structured_array(Xi, mask))
        elif option.startswith('dense'):
            # slow, because it calls Voronoi twice
            N = xyt.shape[0]
            xyt_c = ut.CArray(xyt, dtype=np.float32)
            voro = Voronoi(abg[2], abg[0], abg[1], xyt_c.data).delaunay()
            dense = voro.dense(xyt_c)
            mask = np.zeros((N,), dtype=bool)
            density_flags = list(map(int, option.split(' ')[1].split('|')))
            for flag in density_flags:
                mask = np.bitwise_or(mask, dense == flag)
            N_valid = np.sum(mask)
            return ut.apply_struct(lambda x: np.sum(x) / N_valid)(ut.mask_structured_array(Xi, mask))
        elif option == 'None':
            return Xi
        else:
            raise ValueError("Unknown option!")

    return inner


def CorrelationFunc(order_a: str, order_b: str):
    """
    order_a, order_b: order parameter names.
    All means are taken over each simulation separately.
    args = (abs, xyt)
    abg = (A_upper_bound, B_upper_bound, gamma) for each state
    xyt = (N, 3) configuration
    :return: r, (a - mean(a)) * (b - mean(b)) / (std(a) * std(b))
    """
    if_seg_dist = False

    def inner(args) -> (np.ndarray, np.ndarray):
        abg: tuple = args[0]
        xyt: np.ndarray = args[1]
        if order_a == order_b:
            fields = OrderParameterFunc([order_a], 'None')(args)
            a_field = b_field = fields[order_a]
        else:
            fields = OrderParameterFunc([order_a, order_b], 'None')(args)
            a_field = fields[order_a]
            b_field = fields[order_b]
        a_field_c, b_field_c = ut.CArray(a_field), ut.CArray(b_field)
        mean_a, mean_b = np.mean(a_field), np.mean(b_field)
        std_a, std_b = np.std(a_field), np.std(b_field)
        xyt_c = ut.CArray(xyt, dtype=np.float32)
        N = xyt.shape[0]
        out_r = ut.CArrayFZeros((N * (N + 1) // 2,))
        out_corr = ut.CArrayFZeros((N * (N + 1) // 2,))
        ker.dll.correlation(xyt_c.ptr, a_field_c.ptr, b_field_c.ptr, out_r.ptr, out_corr.ptr, if_seg_dist, N,
                            abg[2], mean_a, mean_b, std_a, std_b)
        return out_r.data, out_corr.data

    return inner


def orderParameterCurve(ensemble: PickledEnsemble, order_parameter_names: list[str], x_axis_name: str,
                        num_threads=1, from_to=None) -> (np.ndarray, np.ndarray):
    """
    :return: (interpolated x: 1 dim, interpolated y: 3 dim)
    """

    if from_to is None:
        x_tensor = ensemble.property(x_axis_name)
    else:
        x_tensor = ensemble.property(x_axis_name)[:, :, from_to[0]:from_to[1]]
    y_tensor = ensemble.apply(OrderParameterFunc(order_parameter_names, 'abs averaged'),
                              num_threads=num_threads, from_to_nth_data=from_to)
    return x_tensor, y_tensor


def averageByReplica(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    :param x: interpolated 1 dim
    :param y: interpolated 2 dim
    :return: (interpolated x: 1 dim, mean y: 1 dim, CI radius y: 1 dim)
    """
    if y.dtype.fields is not None:
        ave_curve = ut.apply_struct(np.nanmean, axis=0)(y)
        ci_curve = ut.apply_struct(mm.CIRadius, axis=0, confidence=0.95)(y)
    else:
        ave_curve = np.nanmean(y, axis=0)
        ci_curve = mm.CIRadius(y, axis=0, confidence=0.95)
    return x, ave_curve, ci_curve


def averageEnergy(ensemble: PickledEnsemble, x_axis_name: str):
    """
    :return: xs, mean energy, energy CI
    """
    x_tensor = ensemble.property(x_axis_name)
    return averageByReplica(x_tensor[0], ensemble.property('energy'))


def orderParameterAnalysis(database: Database, order_parameters: list[str], x_axis_name: str, averaged=False,
                           num_threads=1, from_to=None, out_file='analysis.h5'):
    dic = {}
    for ensemble in database:
        if ensemble.n_density < 1: continue
        sub_dic = {}
        if averaged:
            x, y_mean, y_ci = averageByReplica(
                *orderParameterCurve(ensemble, order_parameters, x_axis_name, num_threads, from_to)
            )
            for order_parameter in order_parameters:
                sub_dic[order_parameter] = (y_mean[order_parameter], y_ci[order_parameter])
            sub_dic[x_axis_name] = x[0, :]
        else:
            x, y_tensor = orderParameterCurve(ensemble, order_parameters, x_axis_name, num_threads, from_to)
            for order_parameter in order_parameters:
                sub_dic[order_parameter] = y_tensor[order_parameter]
            sub_dic[x_axis_name] = x[0, :]
        dic[ensemble.metadata['id'][0].decode('utf-8')] = sub_dic

    # add x-axis
    dict_to_analysis_hdf5(out_file, dic)
    add_property_to_hdf5(out_file, 'x_axis_name', x_axis_name)

    # add metadata
    add_array_to_hdf5(out_file, 'summary_table', database._summary_table_array)


def calAllOrderParameters(database: Database, x_axis_name: str, averaged=False, num_threads=4,
                          out_file='analysis.h5'):
    order_parameters = ['Phi4', 'Phi6', 'S_local', 'S_global', 'EllipticPhi6', 'MeanSegmentDist', 'defect']
    try:
        orderParameterAnalysis(
            database, order_parameters, x_axis_name, averaged, num_threads, out_file=out_file
        )
    except DirtyDataException as e:
        print(e)
        print("Exception is caught. Try to calculate in smaller range.")
        orderParameterAnalysis(
            database, order_parameters, x_axis_name, averaged, num_threads,
            from_to=(0, e.nth_state), out_file=out_file
        )


def CorrelationOverEnsemble(order_a: str, order_b: str):
    """
    order_a, order_b: order parameter names.
    All means are taken over each simulation separately.
    :return: r, (a - mean(a)) * (b - mean(b)) / (std(a) * std(b))
    """
    if_seg_dist = False

    def inner(abg, xyts: list) -> (list[np.ndarray], list[np.ndarray]):
        a_fields = []
        b_fields = []
        for xyt in xyts:
            if order_a == order_b:
                fields = OrderParameterFunc([order_a], 'None')((abg, xyt))
                a_fields.append(fields[order_a])
                b_fields.append(fields[order_a])
            else:
                fields = OrderParameterFunc([order_a, order_b], 'None')((abg, xyt))
                a_fields.append(fields[order_a])
                b_fields.append(fields[order_b])
        A = np.array(a_fields)
        B = np.array(b_fields)
        mean_a, mean_b = np.mean(A), np.mean(B)
        std_a, std_b = np.std(A), np.std(B)
        rs = []
        corrs = []
        for xyt, a_field, b_field in zip(xyts, a_fields, b_fields):
            a_field_c, b_field_c = ut.CArray(a_field), ut.CArray(b_field)
            xyt_c = ut.CArray(xyt, dtype=np.float32)
            N = xyt.shape[0]
            out_r = ut.CArrayFZeros((N * (N + 1) // 2,))
            out_corr = ut.CArrayFZeros((N * (N + 1) // 2,))
            ker.dll.correlation(xyt_c.ptr, a_field_c.ptr, b_field_c.ptr, out_r.ptr, out_corr.ptr, if_seg_dist, N,
                                abg[2], mean_a, mean_b, std_a, std_b)
            rs.append(out_r.data)
            corrs.append(out_corr.data)
        return rs, corrs

    return inner


def AngularCorrelationOverEnsemble(abg, xyts: list) -> (list[np.ndarray], list[np.ndarray]):
    """
    order_a, order_b: order parameter names.
    All means are taken over each simulation separately.
    :return: r, <cos(θi-θj)>
    """
    if_seg_dist = False

    rs = []
    corrs = []
    for xyt in xyts:
        xyt_c = ut.CArray(xyt, dtype=np.float32)
        N = xyt.shape[0]
        out_r = ut.CArrayFZeros((N * (N + 1) // 2,))
        out_corr = ut.CArrayFZeros((N * (N + 1) // 2,))
        ker.dll.angularCorrelation(xyt_c.ptr, out_r.ptr, out_corr.ptr, if_seg_dist, N, abg[2])
        rs.append(out_r.data)
        corrs.append(out_corr.data)
    return rs, corrs


def GeneralCalculation(filenames: list[str], calculation: Callable, save=False, test=True, output_name='data',
                       aggregate_method='sum', horizontal_axis: np.ndarray = None):
    """
    :param calculation: calculation(simu: PickledSimulation) -> np.ndarray
    :param save: True => save a pickle | False => visualization right now
    :param test: True => small amount of data | False => all data
    :param aggregate_method: sum | average
    """
    op_gamma = []
    gammas = [2.2] if test else np.arange(1.1, 3, 0.1)
    ensembles_per_file = 1 if test else 5
    db0 = Database(filenames[0])
    for gamma in gammas:
        ops = []
        for filename in filenames:
            db = Database(filename)
            e = db.find(gamma=gamma)[0]
            for j in range(ensembles_per_file):
                ops.append(calculation(e[j]))
        if aggregate_method == 'sum':
            op = sum(ops)
        elif aggregate_method == 'average':
            op = sum(ops) / len(ops)
        else:
            raise ValueError("Unknown aggregate method!")
        op_gamma.append(op)
    if save:
        with open(f'{output_name}.pkl', 'wb') as f:
            pkl.dump(op_gamma, f)
    else:
        for op, gamma in zip(op_gamma, gammas):
            s = db0.find(gamma=gamma)[0][0]
            if horizontal_axis is None:
                x_axis = s.propertyInterval('phi', upper_h=1.2)[:len(op)]
                plt.scatter(x_axis, op)
            else:
                x_axis = horizontal_axis
                plt.scatter(x_axis, op[-len(x_axis):])
            plt.show()
