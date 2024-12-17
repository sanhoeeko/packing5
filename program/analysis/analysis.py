from typing import Union

import numpy as np

from . import mymath as mm
from .database import Database
from .h5tools import dict_to_analysis_hdf5, add_array_to_hdf5, add_property_to_hdf5
from .mymath import DirtyDataException
from .orders import OrderParameter


def OrderParameterFunc(order_parameter_name: str, weighted: bool, abs_averaged: bool):
    """
    parameters of inner function:
    abg = (A_upper_bound, B_upper_bound, gamma) for each state
    xyt = (N, 3) configuration
    :return: a function object for `Database.apply`
    """

    def inner(args) -> Union[np.float32, np.ndarray]:
        abg: tuple = args[0]
        xyt: np.ndarray = args[1]
        Xi = OrderParameter(order_parameter_name)(xyt, abg, weighted)
        return np.mean(np.abs(Xi)) if abs_averaged else Xi

    return inner


def CorrelationFunc(order_a: str, order_b: str, normal_a: float, normal_b: float, weighted: bool, averaged: bool):
    """
    order_a, order_b: order parameter names.
    All means are taken over each simulation separately.
    :return: mean((a - mean(a)) * (b - mean(b)))
    """

    def inner(args) -> Union[np.float32, np.ndarray]:
        a_field = OrderParameterFunc(order_a, weighted, False)(args)
        b_field = OrderParameterFunc(order_b, weighted, False)(args)
        cor_field = (a_field - normal_a) * (b_field - normal_b)
        if averaged:
            return np.mean(cor_field)
        else:
            return cor_field

    return inner


def interpolatedOrderParameterCurves(database: Database, order_parameter_name: str, x_axis_name: str,
                                     weighted=False, num_threads=1, from_to=None) -> (np.ndarray, np.ndarray):
    """
    :return: (interpolated x: 1 dim, interpolated y: 3 dim)
    """
    if from_to is None:
        x_tensor = database.property(x_axis_name)
    else:
        x_tensor = database.property(x_axis_name)[:, :, from_to[0]:from_to[1]]
    y_tensor = database.apply(OrderParameterFunc(order_parameter_name, weighted, True),
                              num_threads=num_threads, from_to_nth_data=from_to)
    min_parameter_variation = np.min(np.abs(np.diff(x_tensor, axis=2)))
    return mm.interpolate_tensor(x_tensor, y_tensor, min_parameter_variation, num_threads)


def averageByGamma(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    :param x: interpolated 1 dim
    :param y: interpolated 3 dim
    :return: (interpolated x: 1 dim, mean y: 2 dim, CI radius y: 2 dim)
    """
    ave_curve = np.nanmean(y, axis=1)
    ci_curve = mm.CIRadius(y, axis=1, confidence=0.95)
    return x, ave_curve, ci_curve


def averageEnergy(database: Database, x_axis_name: str):
    """
    :return: xs, mean energy, energy CI
    """
    x_tensor = database.property(x_axis_name)
    min_parameter_variation = np.min(np.abs(np.diff(x_tensor, axis=2)))
    return averageByGamma(*mm.interpolate_tensor(x_tensor, database.property('energy'), min_parameter_variation))


def orderParameterAnalysisInterpolated(database: Database, order_parameters: list[str], x_axis_name: str,
                                       weighted=False, num_threads=1, from_to=None):
    out_file = 'analysis.h5'
    dic = {}
    x = None
    for order_parameter in order_parameters:
        x, y_mean, y_ci = averageByGamma(
            *interpolatedOrderParameterCurves(database, order_parameter, x_axis_name, weighted, num_threads, from_to)
        )
        dic[order_parameter] = (y_mean, y_ci)
        print("Successfully calculated order parameter:", order_parameter)

    # add x-axis
    dic['x_axis'] = x
    dict_to_analysis_hdf5(out_file, dic)
    add_property_to_hdf5(out_file, 'x_axis_name', x_axis_name)

    # add metadata
    add_array_to_hdf5(out_file, 'state_table', database.state_table)
    add_array_to_hdf5(out_file, 'simulation_table', database.simulation_table)
    if hasattr(database, 'particle_shape_table'):
        add_array_to_hdf5(out_file, 'particle_shape_table', database.particle_shape_table)


def calAllOrderParameters(database: Database, x_axis_name: str, weighted=False, num_threads=4):
    order_parameters = ['Phi4', 'Phi6', 'S_local', 'S_global']
    try:
        orderParameterAnalysisInterpolated(
            database, order_parameters, x_axis_name, weighted, num_threads
        )
    except DirtyDataException as e:
        print(e)
        print("Exception is caught. Try to calculate in smaller range.")
        orderParameterAnalysisInterpolated(
            database, order_parameters, x_axis_name, weighted, num_threads, from_to=(0, e.nth_state)
        )
