"""
Generate a TT using TT-SVD
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
import time
import ttrecipes as tr


def mysvd(M, delta, rmax, left_ortho=True, verbose=False):

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
        left, M = mysvd(M, delta=delta, rmax=rmax, verbose=verbose)
        M = np.reshape(M, (M.shape[0] * shape[n], M.shape[1] // shape[n]))
        cores.append(np.reshape(left, [left.shape[0] // shape[n - 1], shape[n - 1], left.shape[1]]))
    cores.append(np.reshape(M, [M.shape[0] // shape[N - 1], shape[N - 1], 1]))
    return tt.vector.from_list(cores)


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
        left, M = mysvd(M, delta=delta, rmax=rmax, left_ortho=False, verbose=verbose)
        cores[mu] = np.reshape(M, [-1, shape[mu], cores[mu].shape[2]], order='F')
        cores[mu-1] = np.einsum('ijk,kl', cores[mu-1], left, optimize=True)
    return tt.vector.from_list(cores)
