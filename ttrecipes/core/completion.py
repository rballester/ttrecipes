"""
Methods to interpolate a tensor train over missing values
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
import scipy

import ttrecipes as tr


def substitution(Xs, ys, shape=None, x0=None, rmax=1, niters=10, verbose=False):
    """
    A slow (and probably the simplest) method to complete a tensor:
        - Impose that its error is 0 by adding a rank-P correction. The result becomes rank R+P
        - Do TT-round (go back to rank R)
        - Repeat

    Similar to "Breaking the Curse of Dimensionality Using Decompositions of Incomplete Tensors", N. Vervliet, O. Debals, L. Sorber, L. de Lathauwer (2014)

    :param Xs:
    :param ys:
    :param shape: initial solution (a TT tensor). If None, a random tensor will be used
    :param rmax: an integer (default is 1)
    :param niters: number of iterations
    :param verbose:

    """

    if shape is None:
        shape = np.amax(Xs, axis=0)+1
    P = Xs.shape[0]
    N = Xs.shape[1]
    if x0 is None:
        x0 = tr.core.random_tt(shape, rmax)

    t = x0
    for iter in range(niters):
        error = tr.core.sparse_reco(t, Xs) - ys
        if verbose:
            print("Error: {}".format(np.linalg.norm(error) / np.linalg.norm(ys)))
        cores = []
        firstcore = np.zeros((1, shape[0], P))
        firstcore[np.zeros(P, dtype=np.int), Xs[:, 0], np.arange(P)] = error
        cores.append(firstcore)
        for i in range(1, N - 1):
            core = np.zeros((P, shape[i], P))
            core[np.arange(P), Xs[:, i], np.arange(P)] = 1
            cores.append(core)
        lastcore = np.zeros((P, shape[N - 1], 1))
        lastcore[np.arange(P), Xs[:, N - 1], np.zeros(P, dtype=np.int)] = 1
        cores.append(lastcore)
        residual = tt.vector.from_list(cores)
        t -= residual
        t = tt.vector.round(t, eps=0, rmax=rmax)
    return t


def categorical_ALS(Xs, ys, ws=None, shape=None, x0=None, ranks=1, nswp=10, verbose=False):
    """
    Complete an N-dimensional TT from P samples using alternating least squares (ALS).

    We assume only low-rank structure, and no smoothness/spatial locality. Such assumption requires that there is at least one sample for each tensor hyperslice. This is usually the case for categorical variables, since it is meaningless to have a class or label for which no instances exist. For continuous variables this is trickier. Possible ways to circumvent this include:
        - Add a smoothness objective or prior to the optimization
        - Tensorize the input grid (see quantized tensor train, QTT)
        - Decrease the completion density, so that each hyperslice contains more samples

    Note that this method may not converge (or be extremely slow to do so) if the number of available samples is below or near a certain proportion of the overall tensor. Such proportion, unfortunately, depends on the data set and its true rank structure

    Reference: "Riemannian optimization for high-dimensional tensor completion", M. Steinlechner (2015)

    :param Xs: a P x N matrix
    :param ys: a vector with P elements
    :param ws: a vector with P elements, with the weight of each sample (if None, 1 is assumed)
    :param shape: list of N integers. If None, the smallest shape that accommodates `Xs` will be chosen
    :param x0: initial solution (a TT tensor). If None, a random tensor will be used
    :param ranks: an integer (or list). Default is 1. Ignored if x0 is given
    :param nswp: number of ALS sweeps. Default is 10
    :param verbose:

    """

    if ws is None:
        ws = np.ones(len(ys))
    if shape is None:
        shape = np.amax(Xs, axis=0)+1
    P = Xs.shape[0]
    N = Xs.shape[1]
    if not hasattr(ranks, '__len__'):
        ranks = [1] + [ranks]*(N-1) + [1]
    if x0 is None:
        x0 = tr.core.random_tt(shape, ranks)
    # All tensor slices must contain at least one sample point
    for dim in range(N):
        if np.unique(Xs[:, dim]).size != x0.n[dim]:
            raise ValueError('One groundtruth sample is needed for every tensor slice')

    normys = np.linalg.norm(ys)
    cores = tt.vector.to_list(x0)

    # Memoized product chains for all groundtruth points
    # lefts will be initialized on the go
    lefts = [np.ones([1, P, x0.r[i]]) for i in range(N)]
    # rights, however, needs to be initialized now
    rights = [None] * N
    rights[-1] = np.ones([1, P, 1])
    for dim in range(N-2, -1, -1):
        rights[dim] = np.einsum('ijk,kjl->ijl', cores[dim+1][:, Xs[:, dim+1], :], rights[dim+1])

    def optimize_core(cores, mu, direction):
        sse = 0
        for index in range(cores[mu].shape[1]):
            idx = np.where(Xs[:, mu] == index)[0]
            leftside = lefts[mu][0, idx, :]
            rightside = rights[mu][:, idx, 0]
            lhs = np.transpose(rightside, [1, 0])[:, :, np.newaxis]
            rhs = leftside[:, np.newaxis, :]
            A = np.reshape(lhs*rhs, [len(idx), -1], order='F')*ws[idx, np.newaxis]
            b = ys[idx]*ws[idx]
            sol, residuals = scipy.linalg.lstsq(A, b)[0:2]
            if residuals.size == 0:
                residuals = np.linalg.norm(A.dot(sol) - b) ** 2
            cores[mu][:, index, :] = np.reshape(sol, cores[mu][:, index, :].shape, order='C')
            sse += residuals
        # Update product chains for next core
        if direction == 'right':
            lefts[mu+1] = np.einsum('ijk,kjl->ijl', lefts[mu], cores[mu][:, Xs[:, mu], :])
        else:
            rights[mu-1] = np.einsum('ijk,kjl->ijl', cores[mu][:, Xs[:, mu], :], rights[mu])
        return sse

    for swp in range(nswp):

        # Sweep: left-to-right
        if verbose:
            print("Sweep: {}{{->}}".format(swp), end='')
        for mu in range(N-1):
            optimize_core(cores, mu, direction="right")

        # Sweep: right-to-left
        if verbose:
            print(" {}{{<-}}".format(swp), end='')
        for mu in range(N-1, 0, -1):
            sse = optimize_core(cores, mu, direction="left")

        if verbose:
            print(" | eps: {}".format(np.sqrt(sse) / normys))

    return tt.vector.from_list(cores)


def continuous_ALS(Xs, ys, ws=None, shape=None, x0=None, ranks=1, ranks2=1, nswp=10, verbose=False):
    """
    TT completion for smooth variables: each core is sought within the subspace spanned by a few leading orthogonal polynomials (polynomial chaos expansion, PCE). This is equivalent to fixing the bases in the "extended tensor train" format (ETT), first used in I. Oseledets, "Tensor-train decomposition" (2011). See also: Bigoni et al., "Spectral tensor-train decomposition" (2014).

    Usage: see `categorical_ALS`. The extra parameter `ranks2` (list or integer, default is 1) specifies the core ranks along the spatial dimension; i.e. the size of the truncated basis)

    """

    if ws is None:
        ws = np.ones(len(ys))
    if shape is None:
        shape = np.amax(Xs, axis=0)+1
    P = Xs.shape[0]
    N = Xs.shape[1]
    if not hasattr(ranks, '__len__'):
        ranks = [1] + [ranks]*(N-1) + [1]
    if not hasattr(ranks2, '__len__'):
        ranks2 = [ranks2]*N
    if x0 is None:
        x0 = tr.core.random_tt(ranks2, ranks)
    Us = tr.core.generate_bases('legendre', shape, ranks2)
    normys = np.linalg.norm(ys)
    cores = tt.vector.to_list(x0)
    for mu in range(N):
        cores[mu] = np.einsum('ijk,lj->ilk', cores[mu], Us[mu])  # Put cores in TT (not ETT) format
    tr.core.orthogonalize(cores, 0)

    # Memoized product chains for all groundtruth points
    # lefts will be initialized on the go
    lefts = [np.ones([1, P, x0.r[i]]) for i in range(N)]
    # rights, however, needs to be initialized now
    rights = [None] * N
    rights[-1] = np.ones([1, P, 1])
    for dim in range(N-2, -1, -1):
        rights[dim] = np.einsum('ijk,kjl->ijl', cores[dim+1][:, Xs[:, dim+1], :], rights[dim+1])

    def optimize_core(cores, mu, direction):
        sse = 0
        Rl = cores[mu].shape[0]
        Rr = cores[mu].shape[2]
        S = ranks2[mu]
        A = np.zeros([P, S*Rr*Rl])
        b = np.zeros(P)
        counter = 0
        for index in range(cores[mu].shape[1]):
            idx = np.where(Xs[:, mu] == index)[0]
            Q = len(idx)
            if Q == 0:
                continue
            leftside = lefts[mu][0, idx, :]  # Q x Rl
            rightside = rights[mu][:, idx, 0]  # Rr x Q
            factorside = np.repeat(Us[mu][index:index+1, :], Q, axis=0)[:, :, np.newaxis, np.newaxis]  # Q x D x 1 x 1
            lhs = np.transpose(rightside, [1, 0])[:, np.newaxis, :, np.newaxis]  # Q x 1 x Rr x 1
            rhs = leftside[:, np.newaxis, np.newaxis, :]  # Q x 1 x 1 x Rl
            kronecker = factorside*lhs*rhs  # Q x S x Rr x Rl
            A[counter:counter + Q] = np.reshape(kronecker, [Q, -1], order='F')  # Q x (D Rl Rr)
            b[counter:counter + Q] = ys[idx]
            counter += Q
        A = A*ws[:, np.newaxis]
        b = b*ws
        sol, residuals = scipy.linalg.lstsq(A, b)[0:2]
        if residuals.size == 0:
            residuals = np.linalg.norm(A.dot(sol) - b)**2
        sse += residuals
        cores[mu] = np.transpose(np.reshape(sol, [S, Rr, Rl], order='F'), [2, 0, 1])
        tmp = np.reshape(np.transpose(cores[mu], [1, 0, 2]), [S, -1])
        U, S, V = np.linalg.svd(tmp)
        print()
        print(S)
        cores[mu] = np.einsum('ijk,lj->ilk', cores[mu], Us[mu])  # Decompress this new core: ETT -> TT
        # Update product chains for next core
        if direction == 'right':
            tr.core.left_orthogonalize(cores, mu)
            lefts[mu+1] = np.einsum('ijk,kjl->ijl', lefts[mu], cores[mu][:, Xs[:, mu], :])
        else:
            tr.core.right_orthogonalize(cores, mu)
            rights[mu-1] = np.einsum('ijk,kjl->ijl', cores[mu][:, Xs[:, mu], :], rights[mu])
        return sse

    for swp in range(nswp):

        # Sweep: left-to-right
        if verbose:
            print("Sweep: {}{{->}}".format(swp), end='')
        for mu in range(N-1):
            optimize_core(cores, mu, direction="right")

        # Sweep: right-to-left
        if verbose:
            print(" {}{{<-}}".format(swp), end='')
        for mu in range(N-1, 0, -1):
            sse = optimize_core(cores, mu, direction="left")

        if verbose:
            print(" | eps: {}".format(np.sqrt(sse) / normys))

    return tt.vector.from_list(cores)
