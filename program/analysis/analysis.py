from typing import Union

import numpy as np

from . import mymath as mm, utils as ut
from .database import Database, PickledEnsemble
from .h5tools import dict_to_analysis_hdf5, add_array_to_hdf5, add_property_to_hdf5
from .mymath import DirtyDataException
from .orders import OrderParameterList


def OrderParameterFunc(order_parameter_list: list[str], weighted: bool, abs_averaged: bool):
    """
    parameters of inner function:
    abg = (A_upper_bound, B_upper_bound, gamma) for each state
    xyt = (N, 3) configuration
    :return: a function object for `Database.apply`
    """

    def inner(args) -> np.ndarray:
        abg: tuple = args[0]
        xyt: np.ndarray = args[1]
        Xi = OrderParameterList(order_parameter_list)(xyt, abg, weighted)
        return ut.apply_struct(np.mean)(ut.apply_struct(np.abs)(Xi)) if abs_averaged else Xi

    return inner


def CorrelationFunc(order_a: str, order_b: str, normal_a: float, normal_b: float, weighted: bool, averaged: bool):
    """
    order_a, order_b: order parameter names.
    All means are taken over each simulation separately.
    :return: mean((a - mean(a)) * (b - mean(b)))
    """

    def inner(args) -> Union[np.float32, np.ndarray]:
        fields = OrderParameterFunc([order_a, order_b], weighted, False)(args)
        a_field = fields[order_a]
        b_field = fields[order_b]
        cor_field = (a_field - normal_a) * (b_field - normal_b)
        if averaged:
            return np.mean(cor_field)
        else:
            return cor_field

    return inner


def orderParameterCurve(ensemble: PickledEnsemble, order_parameter_names: list[str], x_axis_name: str,
                        weighted=False, num_threads=1, from_to=None) -> (np.ndarray, np.ndarray):
    """
    :return: (interpolated x: 1 dim, interpolated y: 3 dim)
    """

    if from_to is None:
        x_tensor = ensemble.property(x_axis_name)
    else:
        x_tensor = ensemble.property(x_axis_name)[:, :, from_to[0]:from_to[1]]
    y_tensor = ensemble.apply(OrderParameterFunc(order_parameter_names, weighted, True),
                              num_threads=num_threads, from_to_nth_data=from_to)
    return x_tensor, y_tensor


def averageByReplica(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    :param x: interpolated 1 dim
    :param y: interpolated 3 dim
    :return: (interpolated x: 1 dim, mean y: 2 dim, CI radius y: 2 dim)
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


def orderParameterAnalysis(database: Database, order_parameters: list[str], x_axis_name: str,
                           weighted=False, num_threads=1, from_to=None):
    out_file = 'analysis.h5'
    dic = {}
    for ensemble in database:
        # if ensemble.data_length() == 0: continue
        if ensemble.n_density < 100: continue
        sub_dic = {}
        x, y_mean, y_ci = averageByReplica(
            *orderParameterCurve(ensemble, order_parameters, x_axis_name, weighted, num_threads, from_to)
        )
        for order_parameter in order_parameters:
            sub_dic[order_parameter] = (y_mean[order_parameter], y_ci[order_parameter])
        dic[ensemble.metadata['id'][0].decode('utf-8')] = sub_dic

    # add x-axis
    dict_to_analysis_hdf5(out_file, dic)
    dic['x_axis'] = x
    add_property_to_hdf5(out_file, 'x_axis_name', x_axis_name)

    # add metadata
    # add_array_to_hdf5(out_file, 'state_table', database.state_table)
    add_array_to_hdf5(out_file, 'summary_table', database.summary_table_array)


def calAllOrderParameters(database: Database, x_axis_name: str, weighted=False, num_threads=4):
    order_parameters = ['Phi4', 'Phi6', 'S_local', 'S_global', 'EllipticPhi6', 'MeanSegmentDist', 'StdSegmentDist']
    try:
        orderParameterAnalysis(
            database, order_parameters, x_axis_name, weighted, num_threads
        )
    except DirtyDataException as e:
        print(e)
        print("Exception is caught. Try to calculate in smaller range.")
        orderParameterAnalysis(
            database, order_parameters, x_axis_name, weighted, num_threads, from_to=(0, e.nth_state)
        )
