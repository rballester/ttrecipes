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
import time
import tt
import scipy
import scipy.sparse

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
        firstcore = np.zeros([1, shape[0], P])
        firstcore[np.zeros(P, dtype=np.int), Xs[:, 0], np.arange(P)] = error
        cores.append(firstcore)
        for i in range(1, N - 1):
            core = np.zeros([P, shape[i], P])
            core[np.arange(P), Xs[:, i], np.arange(P)] = 1
            cores.append(core)
        lastcore = np.zeros([P, shape[N - 1], 1])
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
        - Add a smoothness objective or prior to the optimization (see `pce_interpolation`)
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
    Xs = Xs.astype(int)
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

    if verbose:
        print('Completing a {}D tensor of size {} using {} samples...'.format(N, list(shape), P))

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


def pce_interpolation(Xs, ys, ws=None, shape=None, x0=None, ranks=None, ranks2=None, maxswp=50, tol=1e-3, verbose=False):
    """
    TT completion for smooth variables: each core is sought within the subspace spanned by a few leading orthogonal polynomials (polynomial chaos expansion, PCE). This is equivalent to fixing the bases in the "extended tensor train" format (ETT), first used in I. Oseledets, "Tensor-train decomposition" (2011). See also: Bigoni et al., "Spectral tensor-train decomposition" (2014), a version that uses cross-approximation, unlike completion as this function does.

    :param ranks2: integer or list of integers, default is None. Core ranks along the spatial dimension; i.e. the size of the truncated basis

    """

    if ws is None:
        ws = np.ones(len(ys))
    Xs = Xs.astype(int)
    if shape is None:
        shape = np.amax(Xs, axis=0)+1
    P = Xs.shape[0]
    N = Xs.shape[1]
    if x0 is not None:
        ranks = x0.r
    if ranks is None and ranks2 is None:
        ranks = ranks2 = max(1, int(np.real(np.roots([N - 2, 2, 0, -P / 2])[-1])))  # Simple heuristic: highest integer r such that, if ranks = ranks2 = r, the overall number of elements in the tensor won't exceed P / 2
    if not hasattr(ranks, '__len__'):
        ranks = [1] + [ranks]*(N-1) + [1]
    if not hasattr(ranks2, '__len__'):
        ranks2 = [ranks2]*N
    if x0 is None:
        x0 = tr.core.random_tt(shape, ranks)
    Us = tr.core.generate_bases('legendre', shape, ranks2)
    normys = np.linalg.norm(ys)
    cores = tt.vector.to_list(x0)
    # for mu in range(N):
    #     cores[mu] = np.einsum('ijk,lj->ilk', cores[mu], Us[mu])  # Put cores in TT (not ETT) format
    tr.core.orthogonalize(cores, 0)

    if verbose:
        print('Interpolating a {}D tensor of size {} using {} samples and ranks {}...'.format(N, list(shape), P, [r for r in x0.r]))

    # Memoized product chains for all groundtruth points
    # lefts will be initialized on the go
    lefts = [np.ones([1, P, x0.r[i]]) for i in range(N)]
    # rights, however, needs to be initialized now
    rights = [None] * N
    rights[-1] = np.ones([1, P, 1])
    for dim in range(N-2, -1, -1):
        rights[dim] = np.einsum('ijk,kjl->ijl', cores[dim+1][:, Xs[:, dim+1], :], rights[dim+1])

    def optimize_core(cores, mu, direction):

        if True:
            Rl = cores[mu].shape[0]
            S = ranks2[mu]
            Rr = cores[mu].shape[2]

            part1 = lefts[mu][0, :, :]
            part2 = rights[mu][:, :, 0]
            import time
            global time1
            time1 = 0
            global time2
            time2 = 0

            def matvec(v):
                global time1
                start = time.time()
                deco = np.einsum('ijk,aj->iak', np.reshape(v, [Rl, S, Rr]), Us[mu], optimize=True)
                deco = deco[:, Xs[:, mu], :]
                result = np.einsum('ai,ka,iak->a', part1, part2, deco, optimize=True)
                time1 += (time.time() - start)
                return result*ws

            deco2 = np.einsum('ai,ka->iak', part1, part2, optimize=True)
            idx = np.argsort(Xs[:, mu])
            jumps = np.concatenate([np.atleast_1d([0]), np.where(np.diff(Xs[idx, mu]))[0]+1])
            deco2 = deco2[:, idx, :]
            unique = np.unique(Xs[:, mu])

            def rmatvec(v):
                global time2
                start = time.time()
                v *= ws
                zero = np.add.reduceat(deco2*v[np.newaxis, idx, np.newaxis], indices=jumps, axis=1)
                result = np.einsum('ijk,ja->iak', zero, Us[mu][unique, :], optimize=True)
                time2 += (time.time() - start)
                return result

            A = scipy.sparse.linalg.LinearOperator((P, Rl*S*Rr), matvec=matvec, rmatvec=rmatvec)
            b = ys*ws
            solution = scipy.sparse.linalg.lsmr(A, b)
            sol = solution[0]
            residual = solution[3]**2

            cores[mu] = np.reshape(sol, [Rl, S, Rr])
            cores[mu] = np.einsum('ijk,lj->ilk', cores[mu], Us[mu], optimize=True)
        else:
            kronecker = np.einsum('pl,rp,ps->plrs', lefts[mu][0, :, :], rights[mu][:, :, 0], Us[mu][Xs[:, mu], :], optimize=False)
            A = np.reshape(kronecker, [P, -1])*ws[:, np.newaxis]
            # print(A.shape)
            # print(np.linalg.svd(A)[1])
            b = ys*ws
            sol, residual = scipy.linalg.lstsq(A, b)[0:2]
            if residual.size == 0:
                residual = np.linalg.norm(A.dot(sol) - b)**2
            cores[mu] = np.transpose(np.reshape(sol, [ranks2[mu], cores[mu].shape[2], cores[mu].shape[0]], order='F'), [2, 0, 1])
            cores[mu] = np.einsum('ijk,lj->ilk', cores[mu], Us[mu])  # Decompress this new core: ETT -> TT
        # Update product chains for next core
        if direction == 'right':
            tr.core.left_orthogonalize(cores, mu)
            lefts[mu+1] = np.einsum('ijk,kjl->ijl', lefts[mu], cores[mu][:, Xs[:, mu], :])
        else:
            tr.core.right_orthogonalize(cores, mu)
            rights[mu-1] = np.einsum('ijk,kjl->ijl', cores[mu][:, Xs[:, mu], :], rights[mu])
        return residual

    lasteps = float('inf')
    swptime = time.time()
    for swp in range(maxswp):

        # Sweep: left-to-right
        if verbose:
            print("Sweep: {: >3}".format(swp), end='', flush=True)
        for mu in range(N-1):
            if verbose:
                print(('\rSweep: {: >3} [{: >'+str(mu)+'}{: >'+str(N-1-mu)+'}]').format(swp, 'x', ''), flush=True, end='')
            optimize_core(cores, mu, direction="right")

        # Sweep: right-to-left
        for mu in range(N-1, 0, -1):
            sse = optimize_core(cores, mu, direction="left")
            if verbose:
                print(('\rSweep: {: >3} [{: >'+str(mu)+'}{: >'+str(N-1-mu)+'}]').format(swp, 'x', ''), flush=True, end='')
            eps = np.sqrt(sse) / normys

        if verbose:
            print("\rSweep: {: >3} | eps: {:.16f} | time: {:.4f}".format(swp, eps, time.time() - swptime), flush=True)
        swptime = time.time()
        if (lasteps >= eps and (lasteps - eps) / eps < tol) or eps < 1e-14:
            break
        lasteps = eps

    return tt.vector.from_list(cores)
