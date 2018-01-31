# -----------------------------------------------------------------------------
# Authors:      Rafael Ballester-Ripoll <rballester@ifi.uzh.ch>
#
# Copyright:    ttrecipes project (c) 2017-2018
#               VMMLab - University of Zurich
#
# ttrecipes is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ttrecipes is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with ttrecipes.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

from __future__ import (absolute_import, division,
                        print_function, unicode_literals, )
from future.builtins import range

import numpy as np
import tt

import ttrecipes as tr


def cross(ticks_list, fun, mode="array", qtt=False, callback=None, return_n_samples=False, eps=1e-3, verbose=False,
**kwargs):
    """
    Create a TT from a function and a list of discretized axes (the ticks). This function is mostly a convenience
    wrapper for ttpy's multifuncrs2

    :param ticks_list: a list of vectors
    :param fun: the black-box procedure
    :param mode: if "parameters", :param: `fun` takes its N inputs as N parameters. If "array" (default), :param: `fun` takes a single input, namely a P x N array, and returns an iterable with P elements. Mode "array" has *much* less overhead, which makes a difference especially with many function evaluations
    :param qtt: if True, QTT indexing is used, i.e. each axis is reshaped to 2 x ... x 2 and then all dimensions interleaved (all axes must have the same number of ticks, a power of 2). Default is False
    :param callback: if not None, this function will be regularly called with a value in [0, 1] that estimates the fraction of the cross-approximation that has been completed. Default is None
    :param return_n_samples: if True, return also the number of samples taken
    :param eps:
    :param verbose:
    :param kwargs: these will be passed to ttpy's multifuncrs2
    :return: a TT, or (TT, n_samples) if return_n_samples is True

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

    if 'nswp' not in kwargs:
        nswp = 10  # ttpy's default
    else:
        nswp = kwargs['nswp']
    total_calls = nswp*2*(N*3 - 2)
    global n_calls
    n_calls = 0
    global n_samples
    n_samples = 0

    def indices_to_coordinates(Xs):
        """
        Map integer indices (tensor entries) to coordinates via a given ticks_list

        :param Xs: a P x N matrix of integers with ndim columns
        :return coordinates: a P x N matrix

        """

        global n_calls
        n_calls += 1
        if callback is not None:
            callback(n_calls / float(total_calls))
        global n_samples
        n_samples += len(Xs)

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
    result = tt.multifuncrs2(grids, f, eps=eps, verb=verbose, **kwargs)
    if verbose:
        print('Function evaluations: {}'.format(n_samples))
        print('The resulting tensor has ranks {} and {} elements'.format([r for r in result.r], len(result.core)))
    if return_n_samples:
        return result, n_samples
    return result
