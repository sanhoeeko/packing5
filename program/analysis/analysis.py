from typing import Union

import numpy as np

from . import mymath as mm, utils as ut
from .database import Database
from .h5tools import dict_to_analysis_hdf5, add_array_to_hdf5
from .voronoi import Voronoi


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
        xyt_c = ut.CArray(xyt)
        Xi = getattr(Voronoi(abg[2], abg[0], abg[1], xyt_c.data).delaunay(weighted), order_parameter_name)(xyt_c)
        return np.mean(np.abs(Xi)) if abs_averaged else Xi

    return inner


def CorrelationFunc(order_a: str, order_b: str, weighted: bool):
    """
    order_a, order_b: order parameter names.
    All means are taken over each simulation separately.
    :return: mean((a - mean(a)) * (b - mean(b)))
    """

    def inner(args) -> np.float32:
        a_field = OrderParameterFunc(order_a, weighted, False)(args)
        b_field = OrderParameterFunc(order_b, weighted, False)(args)
        a_mean = np.mean(a_field, axis=len(a_field.shape) - 1, keepdims=True)
        b_mean = np.mean(a_field, axis=len(a_field.shape) - 1, keepdims=True)
        cor_field = (a_field - a_mean) * (b_field - b_mean)
        return np.mean(cor_field)

    return inner


def interpolatedOrderParameterCurves(database: Database, order_parameter_name: str, x_axis_name: str,
                                     weighted=False, num_threads=1) -> (np.ndarray, np.ndarray):
    """
    :return: (interpolated x: 1 dim, interpolated y: 3 dim)
    """
    x_tensor = database.property(x_axis_name)
    y_tensor = database.apply(OrderParameterFunc(order_parameter_name, weighted, True), num_threads=num_threads)
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


def orderParameterAnalysisInterpolated(database: Database, order_parameters: list[str], x_axis_name: str,
                                       weighted=False, num_threads=1):
    out_file = 'analysis.h5'
    dic = {}
    x = None
    for order_parameter in order_parameters:
        x, y_mean, y_ci = averageByGamma(
            *interpolatedOrderParameterCurves(database, order_parameter, x_axis_name, weighted, num_threads)
        )
        dic[order_parameter] = (y_mean, y_ci)
    dic[x_axis_name] = x
    dict_to_analysis_hdf5(out_file, dic)
    add_array_to_hdf5(out_file, 'state_table', database.state_table)
    add_array_to_hdf5(out_file, 'particle_shape_table', database.particle_shape_table)
    add_array_to_hdf5(out_file, 'simulation_table', database.simulation_table)
