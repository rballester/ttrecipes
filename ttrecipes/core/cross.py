# -----------------------------------------------------------------------------
# Authors:      Rafael Ballester-Ripoll <rballester@ifi.uzh.ch>
#
# Copyright:    ttrecipes project (c) 2017
#               VMMLab - University of Zurich
# -----------------------------------------------------------------------------

from __future__ import (absolute_import, division,
                        print_function, unicode_literals, )
from future.builtins import range

import numpy as np
import tt

import ttrecipes as tr


def cross(ticks_list, fun, mode="array", qtt=False, eps=1e-3, verbose=False, **kwargs):
    """
    Create a TT from a function and a list of discretized axes (the ticks). This function is mostly a convenience
    wrapper for ttpy's multifuncrs2

    :param ticks_list: a list of vectors
    :param fun: the black-box procedure
    :param mode: if "parameters", :param: `fun` takes its N inputs as N parameters. If "array" (default), :param: `fun` takes a single input, namely a P x N array, and returns an iterable with P elements. Mode "array" has *much* less overhead, which makes a difference especially with many function evaluations
    :param qtt: if True, QTT indexing is used, i.e. each axis is reshaped to 2 x ... x 2 and then all dimensions interleaved (all axes must have the same number of ticks, a power of 2). Default is False
    :param eps:
    :param verbose:
    :param kwargs: these will be passed to ttpy's multifuncrs2
    :return:

    """

    assert mode in ("array", "parameters")

    N = len(ticks_list)
    if qtt:
        I = len(ticks_list[0])
        L = int(np.log2(I))
        if 2**L != int(I):
            raise ValueError('For QTT cross-approximation, the number of ticks must be a power of two along all axes')
        if not all([len(ticks_list[n]) == I for n in range(N)]):
            raise ValueError('For QTT cross-approximation, all axes must have the same number of ticks')
        shape = [2]*(N*L)
    else:
        shape = [len(ticks) for ticks in ticks_list]

    def indices_to_coordinates(Xs):
        """
        Map integer indices (tensor entries) to coordinates via a given ticks_list

        :param indices: a P x N matrix of integers with ndim columns
        :param ticks_list:
        :return coordinates: a P x N matrix

        """

        Xs = Xs.astype(int)
        if qtt:
            Xs = tr.core.idx_from_qtt(Xs, I=I)
        result = np.empty(Xs.shape)
        for j in range(N):
            result[:, j] = np.asarray(ticks_list[j])[Xs[:, j]]
        return result

    if mode == "parameters":
        def f(Xs):
            values = []
            coordinates = indices_to_coordinates(Xs)
            for x in coordinates:
                values.append(fun(*x))
            return np.array(values)
    elif mode == "array":
        def f(Xs):
            coordinates = indices_to_coordinates(Xs)
            return fun(coordinates)

    grids = tr.core.meshgrid(shape)
    if verbose:
        print("Cross-approximating a {}D function with target error {}...".format(N, eps))
    return tt.multifuncrs2(grids, f, eps=eps, verb=verbose, **kwargs)
