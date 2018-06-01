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

import time
import numpy as np
import tt

import ttrecipes as tr


def cross(ticks_list, fun, mode="array", qtt=False, callback=None, return_n_samples=False, stats=False, eps=1e-3, verbose=False,
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
    :param stats: if True, display an error summary over the acquired samples. Default is False
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

        Xs = Xs.astype(int)
        if qtt:
            Xs = tr.core.idx_from_qtt(Xs, I=I)
        result = np.empty(Xs.shape)
        for j in range(N):
            result[:, j] = np.asarray(ticks_list[j])[Xs[:, j]]
        return result

    def check_values(Xs, coordinates, values):
        where = np.where(np.isnan(values))[0]
        if len(where) > 0:
            raise ValueError('NaN detected in cross-approximation: indices = {}, coords = {}'.format(Xs[where[0], :], coordinates[where[0], :]))

        where = np.where(np.isinf(values))[0]
        if len(where) > 0:
            raise ValueError('Infinite detected in cross-approximation: indices = {}, coords = {}'.format(Xs[where[0], :], coordinates[where[0], :]))

    global n_samples
    n_samples = 0
    if stats:
        all_Xs = []
        all_values = []

    if mode == "parameters":
        def f(Xs):
            global n_samples
            values = []
            coordinates = indices_to_coordinates(Xs)
            for x in coordinates:
                values.append(fun(*x))
            values = np.array(values)
            check_values(Xs, coordinates, values)
            n_samples += len(Xs)
            if stats:
                all_Xs.extend(list(Xs))
                all_values.extend(list(values))
            return values
    elif mode == "array":
        def f(Xs):
            global n_samples
            coordinates = indices_to_coordinates(Xs)
            values = fun(coordinates)
            check_values(Xs, coordinates, values)
            n_samples += len(Xs)
            if stats:
                all_Xs.extend(list(Xs))
                all_values.extend(list(values))
            return values

    grids = tr.core.meshgrid(shape)
    if verbose:
        print("Cross-approximating a {}D function with target error {}...".format(N, eps))
        start = time.time()
    result = tt.multifuncrs2(grids, f, eps=eps, verb=verbose, **kwargs)
    if verbose:
        total_time = time.time() - start
        print('Function evaluations: {} in {} seconds (time/evaluation: {})'.format(n_samples, total_time, total_time /
        n_samples))
        print('The resulting tensor has ranks {} and {} elements'.format([r for r in result.r], len(result.core)))
    if stats:
        import matplotlib.pyplot as plt
        all_Xs = np.array(all_Xs)
        all_values = np.array(all_values)
        if len(all_values) > 10000:  # To keep things light
            idx = np.random.choice(len(all_values), 10000, replace=False)
            all_Xs = all_Xs[idx, ...]
            all_values = all_values[idx]
        reco = tr.core.sparse_reco(result, all_Xs)
        n = all_values.size
        norm_diff = np.linalg.norm(all_values - reco)
        eps = norm_diff / np.linalg.norm(all_values)
        rmse = norm_diff / np.sqrt(n)
        psnr = 20 * np.log10((all_values.max() - all_values.min()) / (2 * rmse))
        rsquared = 1 - norm_diff**2 / np.var(all_values)
        fig = plt.figure()
        plt.suptitle(
            'eps = {}, '.format(eps) +
            'rmse = {}\n'.format(rmse) +
            'PSNR = {}, '.format(psnr) +
            'R^2 = {}'.format(rsquared)
        )
        fig.add_subplot(121)
        plt.scatter(all_values, reco)
        plt.xlabel('Groundtruth')
        plt.ylabel('Learned')
        line = np.linspace(all_values.min(), all_values.max(), 100)
        plt.plot(line, line, color='black')
        fig.add_subplot(122)
        plt.hist(reco-all_values, 25, facecolor='green', alpha=0.75)
        plt.xlabel('Error')
        plt.ylabel('Count')
        plt.show()
    if return_n_samples:
        return result, n_samples
    return result
