"""
Generate a TT using TT-SVD
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
import ttrecipes as tr


def truncated_svd(M, delta=None, eps=None, rmax=None, left_ortho=True, verbose=False):
    """
    Decompose a matrix M (size (m x n) in two factors U and V (sizes m x r and r x n)

    :param M: a matrix
    :param delta: if provided, maximum error norm
    :param eps: if provided, maximum relative error
    :param rmax: optionally, maximum r
    :param left_ortho: if True (default), U will be orthonormal. If False, V will
    :param verbose:
    :return: U, V

    """

    if delta is not None and eps is not None:
        raise ValueError('Provide either `delta` or `eps`')
    if delta is None and eps is not None:
        delta = eps*np.linalg.norm(M)
    if delta is None and eps is None:
        delta = 0
    if rmax is None:
        rmax = np.iinfo(np.int32).max
    assert rmax >= 1

    start = time.time()
    if M.shape[0] <= M.shape[1]:
        cov = M.dot(M.T)
        singular_vectors = 'left'
    else:
        cov = M.T.dot(M)
        singular_vectors = 'right'
    if verbose:
        print('Time (sparse_covariance):', time.time() - start)

    if np.linalg.norm(cov) < 1e-14:
        return np.zeros([M.shape[0], 1]), np.zeros([1, M.shape[1]])

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

    if singular_vectors == 'left':
        if left_ortho:
            M2 = left.T.dot(M)
        else:
            M2 = ((1. / svd[1][:rank])[:, np.newaxis]*left.T).dot(M)
            left = left*svd[1][:rank]
    else:
        if left_ortho:
            M2 = M.dot(left * (1. / svd[1][:rank])[np.newaxis, :])
            left, M2 = M2, left.dot(np.diag(svd[1][:rank])).T
        else:
            M2 = M.dot(left)
            left, M2 = M2, left.T
    if verbose:
        print('Time (product):', time.time() - start)

    return left, M2


def tt_svd(X, eps, rmax=np.iinfo(np.int32).max, verbose=False):
    """
    Compress a full dense tensor into a TT using the TT-SVD algorithm. Left singular vectors of each unfolding M are computed via eigenvalue decomposition of M.T.dot(M), or via M.dot(eig(M.dot(M.T))) (whatever is cheaper)

    :param X: a dense ndarray
    :param eps: prescribed accuracy (resulting relative error is guaranteed to be not larger than this)
    :param rmax: optionally, cap all ranks above this value
    :param verbose:
    :return: a TT

    """

    N = X.ndim
    delta = eps / np.sqrt(N - 1) * np.linalg.norm(X)
    cores = []
    shape = X.shape
    M = np.reshape(X.astype(float), [shape[0], -1])
    for n in range(1, N):
        left, M = truncated_svd(M, delta=delta, rmax=rmax, verbose=verbose)
        M = np.reshape(M, (M.shape[0] * shape[n], M.shape[1] // shape[n]))
        cores.append(np.reshape(left, [left.shape[0] // shape[n - 1], shape[n - 1], left.shape[1]]))
    cores.append(np.reshape(M, [M.shape[0] // shape[N - 1], shape[N - 1], 1]))
    return tt.vector.from_list(cores)


def full(t, keep_end_ranks=False):
    """
    NumPy's einsum() does quite a good job at TT reconstruction, especially for bigger tensors
    """

    if t.d > 26:
        return t.full()
    str = ','.join([chr(ord('a') + n) + chr(ord('A') + n) + chr(ord('a') + n + 1) for n in range(t.d)])
    str += '->' + 'a' + ''.join([chr(ord('A') + n) for n in range(t.d)]) + chr(ord('a') + t.d)
    result = np.einsum(str, *tt.vector.to_list(t), optimize=True)
    if not keep_end_ranks:
        if t.r[0] == 1:
            result = result[0, ...]
        if t.r[-1] == 1:
            result = result[..., 0]
    return result


def round(t, eps, rmax=np.iinfo(np.int32).max, verbose=False):
    N = t.d
    shape = t.n
    cores = tt.vector.to_list(t)
    start = time.time()
    tr.core.orthogonalize(cores, N-1)
    if verbose:
        print('Orthogonalization time:', time.time() - start)
    delta = eps / np.sqrt(N - 1) * np.linalg.norm(cores[-1])
    for mu in range(N-1, 0, -1):
        M = tr.core.right_unfolding(cores[mu])
        left, M = truncated_svd(M, delta=delta, rmax=rmax, left_ortho=False, verbose=verbose)
        cores[mu] = np.reshape(M, [-1, shape[mu], cores[mu].shape[2]], order='F')
        cores[mu-1] = np.einsum('ijk,kl', cores[mu-1], left, optimize=True)
    return tt.vector.from_list(cores)
