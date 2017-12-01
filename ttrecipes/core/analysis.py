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
        variances_shifted = variances - tt.vector.from_list([np.ones([1, sh+1, 1]) for sh in s.shape])*shift
        val, point = tt_min.min_tens(variances_shifted*mask, verb=verbose)
        val += shift
    nones = np.where(np.array(point) == 0)[0]
    point = [p - 1 for p in point]
    for i in nones:
        point[i] = slice(None)
    return point, val
