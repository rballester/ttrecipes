"""
Methods for sparse compression and decompression
"""

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
import time


def sparse_reco(t, Xs):
    """
    Reconstruct a sparse set of samples from a TT

    :param t:
    :param Xs: a P x N matrix of integers
    :return: a P-vector

    """

    P = Xs.shape[0]
    Xs = Xs.astype(int)
    lefts = np.ones([P, 1])
    for i, core in enumerate(tt.vector.to_list(t)):
        lefts = np.einsum('jk,kjl->jl', lefts, core[:, Xs[:, i]])
    return np.squeeze(lefts)


def sparse_covariance(Xs, ys, nrows):
    hashed_cols = np.einsum('i,ji', np.random.randint(0, np.iinfo(np.int).max, [Xs.shape[1]-1]), Xs[:, 1:])
    u, v = np.unique(hashed_cols, return_inverse=True)
    D = np.zeros([nrows, len(u)])
    D[Xs[:, 0], v] = ys
    return D.dot(D.T)


def full_times_sparse(F, Xs, ys):
    hashed_cols = np.einsum('i,ji', np.random.randint(0, np.iinfo(np.int).max, [Xs.shape[1]-1]), Xs[:, 1:])
    u, idx, v = np.unique(hashed_cols, return_index=True, return_inverse=True)
    D = np.zeros([F.shape[1], len(u)])
    D[Xs[:, 0], v] = ys
    FD = F.dot(D)
    new_row = np.mod(np.arange(FD.size), FD.shape[0])
    newcols = np.repeat(Xs[idx, 1:][:, np.newaxis, :], FD.shape[0], axis=1)
    newcols = np.reshape(newcols, [len(idx)*FD.shape[0], -1])
    return np.hstack([new_row[:, np.newaxis], newcols]), FD.flatten(order='F')


def sparse_tt_svd(Xs, ys, eps, shape=None, rmax=np.iinfo(np.int32).max, verbose=False):
    """
    Sparse TT-SVD

    :param Xs: sample coordinates
    :param ys: sample values
    :param eps: prescribed accuracy (resulting relative error is guaranteed to be not larger than this)
    :param shape: input tensor shape. If not specified, a tensor will be chosen such that Xs fits in
    :param rmax: optionally, cap all ranks above this value
    :param verbose:
    :return: a TT

    """

    def mysvd(Xs, ys, nrows, delta, rmax):

        start = time.time()
        cov = sparse_covariance(Xs, ys, nrows)
        if verbose:
            print('Time (sparse_covariance):', time.time() - start)

        start = time.time()
        w, v = np.linalg.eigh(cov)
        if verbose:
            print('Time (eigh):', time.time() - start)
        w[w < 0] = 0
        w = np.sqrt(w)
        svd = [v, w]

        # Sort eigenvalues and eigenvectors in decreasing importance
        idx = np.argsort(svd[1])[::-1]
        svd[0] = svd[0][:, idx]
        svd[1] = svd[1][idx]

        S = svd[1]**2
        where = np.where(np.cumsum(S[::-1]) <= delta**2)[0]
        if len(where) == 0:
            rank = max(1, int(np.min([rmax, len(S)])))
        else:
            rank = max(1, int(np.min([rmax, len(S) - 1 - where[-1]])))
        left = svd[0]
        left = left[:, :rank]

        start = time.time()
        Xs, ys = full_times_sparse(left.T, Xs, ys)
        if verbose:
            print('Time (product):', time.time() - start)

        return left, Xs, ys

    N = Xs.shape[1]

    if shape is None:
        shape = np.amax(Xs, axis=0) + 1
    assert N == len(shape)
    assert np.all(shape > np.amax(Xs, axis=0))
    shape = np.array(shape)
    delta = eps / np.sqrt(N - 1) * np.linalg.norm(ys)

    cores = []
    curshape = shape.copy()
    for n in range(1, N):
        left, Xs, ys = mysvd(Xs, ys, curshape[0], delta=delta, rmax=rmax)
        cores.append(np.reshape(left, [left.shape[0] // shape[n - 1], shape[n - 1], left.shape[1]]))
        rank = left.shape[1]
        curshape[0] = rank
        if n == N-1:
            break

        # Merge the two first indices (sparse reshape)
        Xs = np.hstack([Xs[:, 0:1]*curshape[1] + Xs[:, 1:2], Xs[:, 2:]])
        tmp = curshape[0]*curshape[1]
        curshape = curshape[1:]
        curshape[0] = tmp

    lastcore = np.zeros(curshape)
    lastcore[list(Xs.T)] = ys
    cores.append(lastcore[:, :, np.newaxis])

    return tt.vector.from_list(cores)
