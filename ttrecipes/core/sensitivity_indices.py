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
import scipy as sp
import scipy.signal
import tt

import ttrecipes as tr


def sobol_tt(t, pdf=None, premultiplied=False, eps=1e-6, verbose=False, **kwargs):
    """
    Create a Sobol tensor train, i.e. a 2^N TT tensor that compactly encodes all Sobol' indices of a surrogate according to a pdf. For example, element S_{2} is encoded by
     [0, 1, 0, ..., 0], element S_{12} is encoded by [1, 1, 0, ..., 0], and so on

    :param t: a TT
    :param pdf: a TT containing the joint PDF of the N input variables. It does not need to sum 1. If None (default), independent uniformly distributed variables will be assumed. If a list of marginals (vectors), independent variables will be assumed (i.e. separable PDF -> rank-1 TT)
    :param premultiplied: if False (default), `t` is assumed to encode a TT surrogate or analytical function. If True, it is assumed to be the surrogate times the PDF.
    :param eps: default is 1e-6
    :param verbose:
    :param kwargs: these will be used for the cross-approximation
    :return: a 2^N TT with all Sobol indices (if the PDF is not separable, some may be negative)

    """

    N = t.d
    if hasattr(pdf, '__len__'):
        pdf = tr.core.separable_tt(pdf)
    if pdf is None:
        pdf = tr.core.constant_tt(shape=t.n, fill=1./np.prod(t.n))  # A constant function that sums 1

    if premultiplied:
        tpdf = t
    else:
        if np.max(pdf.r) == 1:
            tpdf = t*pdf
        else:
            tpdf = tt.multifuncrs2([t, pdf], lambda x: x[:, 0] * x[:, 1], y0=t, eps=eps, verb=verbose, **kwargs)
    t2 = tt.vector.from_list([np.concatenate([np.sum(core, axis=1, keepdims=True), core], axis=1) for core in tt.vector.to_list(tpdf)])

    pdf2 = tt.vector.from_list([np.concatenate([np.sum(core, axis=1, keepdims=True), core], axis=1) for core in tt.vector.to_list(pdf)])

    def fun(x):
        x[x[:, 1] == 0, 1] = float('inf')
        result = (x[:, 0]**2 / x[:, 1])
        return result
    t_normalized_sq = tt.multifuncrs2([t2, pdf2], fun, y0=t2, eps=eps, verb=verbose, **kwargs)

    sobol = tt.vector.from_list([np.concatenate([core[:, 0:1, :], np.sum(core[:, 1:, :], axis=1, keepdims=True) - core[:, 0:1, :]], axis=1) for core in tt.vector.to_list(t_normalized_sq)])
    sobol *= (1. / (tr.core.sum(sobol) - sobol[[0, ]*N]))
    correction = tt.vector.from_list([np.array([1, 0])[np.newaxis, :, np.newaxis], ]*N)  # Set first index to 0 for convenience
    return (sobol - correction*sobol[[0, ]*N]).round(eps=0)


def semivalues(game, ps, p=None, eps=1e-10):
    """
    Compute all N semivalues for each of N players. Each semivalue 1, ..., N has cost O(N^3 R) + O(N^2 R^2),
    where R is the game's rank

    References:

    - "Semivalues and applications" (R. Lucchetti), http://www.gametheory.polimi.it/uploads/4/1/4/6/41466579/dauphine_june15_2015.pdf
    - "A Note on Values and Multilinear Equations" (A. Roth), http://web.stanford.edu/~alroth/papers/77_NRLQ_NoteValuesMultilinearExt.pdf

    :param game: a 2^N TT
    :param ps: 'shapley', 'banzhaf-coleman', 'binomial', a function, or an array of N values: for each n, probability that a player joins any coalition of size n that does not include him/her. It must satisfy that ps[n] > 0 for all n (regularity) and \sum_{n=0}^{N-1} \binom(N-1}{n} ps[n] = 1 (it is a probability)
    :param p: parameter in (0, 1). Used when `ps` is 'binomial', ignored otherwise
    :param eps: default is 1e-6
    :return: array with N semivalues

    """

    N = game.d
    if ps == 'shapley':
        ps = [1./(N*sp.special.binom(N-1, n)) for n in range(N)]
    elif ps == 'banzhaf-coleman':
        ps = [1./(2**(N-1))]*N
    elif ps == 'binomial':
        ps = [p**n * (1-p)**(N-n-1) for n in range(N)]
    elif hasattr(ps, '__call__'):
        ps = [ps(n) for n in range(N)]
    if np.abs(np.sum([sp.special.binom(N-1, n)*ps[n] for n in range(N)]) - 1) > 1e-10:
        raise ValueError('The `ps` must be a probability')
    if not all([ps[n] >= 0 for n in range(N)]):
        raise ValueError('The `ps` must be regular')

    cores = [np.concatenate([core[:, 0:2, :], core[:, 1:2, :] - core[:, 0:1, :]], axis=1) for core in tt.vector.to_list(game)]
    game = tt.vector.from_list(cores)

    ws = tr.core.hamming_weight_state(N)
    ws_cores = tt.vector.to_list(ws)
    ws_cores[-1][np.arange(N), np.arange(N), 0] = ps
    ws_cores[-2] = np.einsum('ijk,km->ijm', ws_cores[-2], np.sum(ws_cores[-1][:, :-1, :], axis=1))  # Absorb last core
    ws_cores = ws_cores[:-1]
    ws_cores = [np.concatenate([core, core[:, 0:1, :]], axis=1) for core in ws_cores]
    ws = tt.vector.from_list(ws_cores)
    ws = tt.vector.round(ws, eps=eps)

    result = []
    for n in range(N):
        idx = [slice(0, 2)]*N
        idx[n] = slice(2, 3)
        result.append(np.asscalar(tr.core.tt_dot(game[idx], ws[idx])))
    return np.asarray(result)


def mean_dimension(st):
    """
    Compute the mean dimension from a Sobol' tensor

    Reference: "Valuation of Mortgage Backed Securities Using Brownian Bridges to Reduce Effective Dimension", R. Caflisch, W. Morekoff, and A. Owen (1997)

    :param st: a Sobol TT
    :return: a real >= 1

    """

    h = tr.core.hamming_weight(st.d)
    return tr.core.tt_dot(st, h)


def order_contribution(st, order, mode='eq'):
    """
    Compute the relative variance contributed by terms of [up to] a fixed order

    :param st: a Sobol TT
    :param order: a positive integer
    :param mode: if 'eq' (default), only the order is considered. If 'le', all terms lower or equal than the order are aggregated
    :return: a real between 0 and 1

    """

    if order == 0:
        return 0
    assert mode in ('eq', 'le')

    N = st.d
    if mode == 'eq':
        weighted = st * tr.core.hamming_eq_mask(N, order)
    else:
        weighted = st * tr.core.hamming_le_mask(N, order)
    return np.squeeze(tr.core.squeeze(tt.vector.from_list([np.sum(core, axis=1, keepdims=True) for core in tt.vector.to_list(weighted)])).full())


def effective_dimension(st, threshold=0.95, mode='superposition'):
    """
    Compute the effective dimension from the Sobol' indices in one of three senses:
    - Superposition: the minimal integer such that all contributions up to this order add up to the requested relative variance threshold
    - Truncation: the size of the smallest set of variables that explains up to the threshold
    - Successive: the minimal integer such that all contributions of tuples of this length (or less) exceed the threshold. The length of a tuple is the length of the longest non-empty substring of its binary representation. For example, the length of [0, 1, 0, 1, 0] is 3, and the length of [0, 0, 0] is 0.

    References:
    - "Valuation of Mortgage Backed Securities Using Brownian Bridges to Reduce Effective Dimension", by Caflisch, R.E. and Morokoff, W.J. and Owen, A.B. (1997)
    - "Variance Reduction via Lattice Rules", by P. L'Ecuyer and C. Lemieux (2000)

    :param st: a Sobol TT
    :param threshold: the variance to surpass (a real between 0 and 1)
    :param mode: 'superposition' (default), 'truncation', or 'successive'
    :return: an integer (the dimension) and a float (the variance attained)

    """

    assert 0 <= threshold <= 1

    N = st.d
    if mode == 'superposition':
        order = 1
        accum = 0
        for order in range(1, N):
            accum += order_contribution(st, order)
            if accum >= threshold:
                break
        return order, accum
    elif mode == 'truncation':
        closed = tr.core.to_lower(st)
        for order in range(1, N+1):
            best = tr.core.largest_k_tuple(closed, order)
            if best[1] >= threshold:
                return order, best[1]
    elif mode == 'successive':
        for order in range(1, N+1):
            contribution = tr.core.tt_dot(st, tr.core.lness_le_mask(st.d, order)).item()
            if contribution >= threshold:
                return order, contribution
    else:
        raise ValueError("Mode must be either 'superposition', 'truncation' or 'successive'")


def dimension_distribution(st):
    """
    Computes a vector with the Sobol' contributions for order 1, ..., N.

    Done via tensor contraction of the Sobol' tensor with the Hamming weight state, which has one more dimension. This last dimension is not contracted, and results in the final vector

    Reference: A. Owen, "The dimension distribution and quadrature test functions" (2003)

    :param st: a Sobol TT
    :return: a vector of N elements
    """

    def partial_dot_right(cores1, cores2, mu, Rprod):
        """
        Advance the dot product computation towards the right
        """
        Ucore = tr.core.ttm(cores1[mu], Rprod, mode=0, transpose=True)
        Vcore = cores2[mu]
        return np.dot(tr.core.left_unfolding(Ucore).T, tr.core.left_unfolding(Vcore))

    N = st.d
    cores1 = tt.vector.to_list(st)
    cores2 = tt.vector.to_list(tr.core.hamming_weight_state(N))
    d = len(cores1)
    Rprod = np.array([[1]])
    for mu in range(d):
        Rprod = partial_dot_right(cores1, cores2, mu, Rprod)
    return np.squeeze(Rprod)[1:]
