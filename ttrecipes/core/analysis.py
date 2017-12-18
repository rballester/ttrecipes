"""
Various processing and optimization tasks in the TT format
"""

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
from tt.optimize import tt_min
import copy

import ttrecipes as tr


def best_subspace(t, ndim=1, target='max', mode='cross', eps=1e-6, verbose=False, **kwargs):
    """
    Find an axis-aligned subspace of a certain dimensionality that has the highest/lowest variance. Example applications:
    - In visualization, to find interesting subspaces
    - In factor fixing (sensitivity analysis), to find the set of parameters (and their values) that will minimize the uncertainty of a model

    TODO: only uniform independently distributed inputs are supported now

    :param t: a TT
    :param ndim: dimensionality of the subspace sought (default is 2)
    :param target: if 'max' (default), the highest variance will be sought; if 'min', the lowest one
    :param mode: 'cross' (default) or 'kronecker'
    :param verbose:
    :param kwargs: arguments for the cross-approximation
    :return: (a) a list of indices, with slice(None) in the free subspace's dimensions, and (b) the variance of that subspace

    """

    assert mode in ('cross', 'kronecker')
    assert target in ('max', 'min')

    # Build up a tensor that contains variances of all possible subspaces of any dimensionality, using the formula E(X^2) - E(X)^2
    if mode == 'cross':
        cores = tt.vector.to_list(tt.multifuncrs2([t], lambda x: x ** 2, eps=eps, verb=verbose, **kwargs))
    else:
        cores = tt.vector.to_list((t*t).round(0))
    cores = [np.concatenate([np.mean(core, axis=1, keepdims=True), core], axis=1) for core in cores]
    part1 = tt.vector.from_list(cores)  # E(X^2)

    cores = tt.vector.to_list(t)
    cores = [np.concatenate([np.mean(core, axis=1, keepdims=True), core], axis=1) for core in cores]
    part2 = tt.vector.from_list(cores)
    if mode == 'cross':
        part2 = tt.multifuncrs2([part2], lambda x: x ** 2, eps=eps, verb=verbose, **kwargs)
    else:
        part2 = (part2*part2).round(0)  # E(X)^2

    variances = (part1 - part2).round(0)

    # Filter out encoded subspaces that do not have the target dimensionality
    mask = tt.vector.to_list(tr.core.hamming_eq_mask(t.d, t.d-ndim))
    mask = [np.concatenate([core[:, 0:1, :], np.repeat(core[:, 1:, :], sh, axis=1)], axis=1) for core, sh in zip(mask, t.n)]
    mask = tt.vector.from_list(mask)

    # Find and return the best candidate
    if target == 'max':
        prod = tt.vector.round(variances*mask, eps=eps)
        val, point = tt_min.min_tens(-prod, verb=verbose)
        val = -val
    else:
        shift = -1e3*tt_min.min_tens(-variances, verb=False, rmax=1)[0]
        variances_shifted = variances - tt.vector.from_list([np.ones([1, sh+1, 1]) for sh in t.n])*shift
        val, point = tt_min.min_tens(variances_shifted*mask, verb=verbose)
        val += shift
    nones = np.where(np.array(point) == 0)[0]
    point = [p - 1 for p in point]
    for i in nones:
        point[i] = slice(None)
    return point, val


def moments(t, modes, order, centered=False, normalized=False, eps=1e-3, verbose=False, **kwargs):
    """
    Given an N-dimensional TT and a list of M modes, returns a TT of dimension N - M that contains the k-th order moments along these modes

    :param t: a TT
    :param modes: a list of M integers
    :param order: an integer
    :param centered: if True the moments will be computed about their mean. Default is False
    :param normalized: if True the moments will be divided by sigma^order. Default is False
    :param eps: accuracy for cross-approximation (default is 1e-3)
    :return: a TT of dimension N - M

    """

    N = t.d
    assert np.all(0 <= np.array(modes))
    assert np.all(np.array(modes) < N)
    if not hasattr(modes, '__len__'):
        modes = [modes]
    assert len(modes) == len(set(modes))  # Modes may not be repeated
    assert 1 <= len(modes) <= N

    if centered or normalized:
        central_cores = []
        cores = tt.vector.to_list(t)
        for n in range(N):
            if n in modes:
                central_cores.append(np.repeat(np.mean(cores[n], axis=1, keepdims=True), cores[n].shape[1], axis=1))
            else:
                central_cores.append(cores[n])
        central = t - tt.vector.from_list(central_cores)
    if centered:
        if order == 1:
            moments = copy.deepcopy(central)
        else:
            moments = tt.multifuncrs2([central], lambda x: x**order, eps=eps, verb=verbose, **kwargs)
    else:
        if order == 1:
            moments = copy.deepcopy(t)
        else:
            moments = tt.multifuncrs2([t], lambda x: x**order, eps=eps, verb=verbose, **kwargs)
    cores = tt.vector.to_list(moments)
    for mode in modes:
        cores[mode] = np.mean(cores[mode], axis=1, keepdims=True)
    moments = tt.vector.from_list(cores)
    if normalized:
        central = tt.multifuncrs2([central], lambda x: x**2, eps=eps, verb=verbose, **kwargs)
        cores = tt.vector.to_list(central)
        for mode in modes:
            cores[mode] = np.mean(cores[mode], axis=1, keepdims=True)
        variances = tt.vector.from_list(cores)
        moments = tt.multifuncrs2([moments, variances], lambda x: x[:, 0] / (x[:, 1] ** (order/2.)), eps=eps, verb=verbose, **kwargs)
    return tr.core.squeeze(moments)


def means(t, modes, **kwargs):
    """
    Convenience function for the first moment (see :func: `moments`)
    """

    return moments(t, modes, order=1, centered=False, normalized=False, **kwargs)


def variances(t, modes, **kwargs):
    """
    Convenience function for the second centered unnormalized moment (see :func: `moments`)
    """

    return moments(t, modes, order=2, centered=True, normalized=False, **kwargs)