"""
Utilities for evaluating and manipulating TTs and their cores
"""

# -----------------------------------------------------------------------------
# Authors:      Rafael Ballester-Ripoll <rballester@ifi.uzh.ch>
#               Enrique G. Paredes <egparedes@ifi.uzh.ch>
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
import copy
import tt
import scipy.interpolate
import scipy.fftpack
import scipy.sparse as sps
import ttrecipes as tr


def load(file, verbose=False):
    """Load a TT vector from a Numpy dict file (.npz)

    :param file: filename or file object with the TT vector
    :param verbose: Show information messages. Defaults to False
    :return: a TT vector

    """
    with np.load(file) as data:
        assert 'd' in data.keys() and 'n' in data.keys() and 'r' in data.keys()
        d = data['d'].item()
        n = data['n']
        r = data['r']

        if verbose:
            print("Loading {d}-dimensional TT ...".format(d=d))
            print("\t - sizes: {n}".format(n=n))
            print("\t - ranks: {r}".format(r=r))

        cores = [None] * d
        for i in range(d):
            cores[i] = data["core_{}".format(i)]

    return tt.vector.from_list(cores)


def save(t, file, compressed=False):
    """Save a TT vector to a Numpy dict file (.npz)

    :param t: a TT vector
    :param file: filename or file object with the TT vector
    :param compressed: Use the compressed format. Defaults to False

    """
    save_np = np.savez_compressed if compressed else np.savez
    core_files = dict([("core_{}".format(i), core)
                       for i, core in enumerate(tt.vector.to_list(t))])
    save_np(file, d=np.asarray([t.d]), n=t.n, r=t.r, **core_files)


def reshape(a, sz):
    return np.reshape(a, sz, order="F")


def ql_decomposition(U):
    """
    Decomposes U = Q*L with Q orthogonal and L lower-triangular
    """

    r, q = scipy.linalg.rq(U.T, mode='economic')
    return q.T, r.T


def is_identity(U, eps=1e-10):
    U_norm2 = np.sum(U ** 2)
    U_norm2_minusid = U_norm2 - np.sum(np.diag(U) ** 2) + np.sum((np.diag(U) - 1) ** 2)
    if U_norm2_minusid / U_norm2 < eps:
        return True
    return False


def is_semiorthonormal(U, **kwargs):
    """
    Checks if either the rows or columns of U are orthonormal
    """

    if U.shape[0] > U.shape[1]:
        return is_identity(U.T.dot(U), **kwargs)
    else:
        return is_identity(U.dot(U.T), **kwargs)


def generate_coords(Is, P, replace=True):
    """
    Generate uniformly-distributed random_sampling points over a tensor grid

    :param Is: tensor shape
    :param P: how many points to generate
    :param replace: Whether the points should be generated with replacement (they may repeat). Default is True
    Returns :param: P *distinct* points over a regular grid defined by :param axes:.

    """

    if replace:
        Xs = np.random.randint(0, np.prod(Is), P)
    else:
        import random
        Xs = random.sample(range(np.prod(Is)), P)
    return np.asarray(np.unravel_index(Xs, Is), dtype=np.int).T


def ticks_list(Xs, shape=64):
    """
    Compute a list of ticks that match a sample matrix Xs

    :param Xs: a P x N matrix of floats
    :param shape: integer (or list), default is 64
    :return:
    """

    N = Xs.shape[1]
    if not hasattr(shape, '__len__'):
        shape = [shape] * N
    assert len(shape) == N

    return [np.linspace(left, right, I) for left, right, I in zip(np.amin(Xs, axis=0), np.amax(Xs, axis=0), shape)]


def indices_to_coordinates(Xs, ticks_list):
    """
    Map integer indices (tensor entries) to space coordinates

    :param Xs: a P x N matrix of integers
    :param ticks_list: a list of vectors
    :return coordinates: a P x N matrix of floats

    """

    N = len(ticks_list)
    assert Xs.shape[1] == N
    Xs = Xs.astype(int)

    coordinates = np.zeros(Xs.shape)
    for dim in range(N):
        coordinates[:, dim] = ticks_list[dim][Xs[:, dim]].astype(float)
    return coordinates


def coordinates_to_indices(coordinates, ticks_list):
    """
    Map space coordinates to integer indices (tensor entries)

    :param coordinates: a P x N matrix of floats
    :param ticks_list: a list of vectors
    :return Xs: a P x N matrix of integers

    """

    N = len(ticks_list)
    assert coordinates.shape[1] == N

    interps = [scipy.interpolate.interp1d(axis, np.arange(
        len(axis)), kind='nearest') for axis in ticks_list]
    Xs = np.empty(coordinates.shape, dtype=np.int)
    for j in range(N):
        Xs[:, j] = interps[j](coordinates[:, j])
    return Xs


def generate_bases(name, Is, Ds=None, orthonormal=False):
    """
    Generate factor matrices that

    :param name:
    :param Is: list of spatial dimensions
    :param Ds: list of number of bases to calculate per dimension
    :param orthonormal: whether to orthonormalize the bases
    :return: a list of matrices, with the bases as columns
    """

    if Ds is None:
        Ds = list(Is)
    if not hasattr(Ds, '__len__'):
        Ds = [Ds, ]*len(Is)
    assert len(Is) == len(Ds)

    if name == "dct":
        Us = [scipy.fftpack.dct(np.eye(I), norm="ortho")[:, :D] for I, D in zip(Is, Ds)]
    else:
        Us = []
        for I, D in zip(Is, Ds):
            eval_points = np.linspace(-1+(1./I), 1-(1./I), I)
            # eval_points = np.linspace(-1, 1, I)
            if name == "legendre":
                U = np.polynomial.legendre.legval(eval_points, np.eye(I, D)).T
            elif name == "chebyshev":
                U = np.polynomial.chebyshev.chebval(eval_points, np.eye(I, D)).T
            elif name == "hermite":
                U = np.polynomial.hermite.hermval(eval_points, np.eye(I, D)).T
            else:
                raise ValueError("Unrecognized basis function")
            Us.append(U)
    if orthonormal:
        Us = [U / np.sqrt(np.sum(U*U, axis=0)) for U in Us]
    return Us


def generate_laplacian(size, periodic=False):
    """
    Create a sparse M such that M*x approximates the Laplacian of x for any x of the given size

    :param size: integer
    :param periodic: if True, x is assumed to be periodic. Default is False
    :return: a scipy.sparse.lil_matrix of shape :param: `size` x `size` (if periodic) and :param: `size-2` x `size` otherwise

    """

    if periodic:
        L = sps.lil_matrix((size, size))
        inds = np.arange(size)
        L[inds, inds] = -2
        L[inds, np.mod(inds+1, size)] = 1
        L[inds, np.mod(inds-1, size)] = 1
    else:
        L = sps.lil_matrix((size-2, size))
        inds = np.arange(size-2)
        L[inds, inds+1] = -2
        L[inds, inds] = 1
        L[inds, inds+2] = 1
    return L


def ttm(core, U, mode, transpose=False):
    """
    Tensor-times-matrix (TTM) along a single mode

    :param core: an ND array
    :param U: the factor
    :param mode:
    :param transpose: if False (default) the contraction is performed
     along U's columns, else along its rows
    :return:

    """

    inds = (mode, ) + tuple(np.delete(np.arange(core.ndim), mode))
    core2 = np.transpose(core, inds)
    shape = core2.shape
    core2 = reshape(core2, [core2.shape[0], -1])
    if transpose:
        core2 = U.T.dot(core2)
    else:
        core2 = U.dot(core2)
    core2 = reshape(core2, (core2.shape[0], ) + shape[1:])
    core2 = np.transpose(core2, np.argsort(inds))
    return core2


def sum_and_compress(tts, rounding=1e-8, rmax=np.iinfo(np.int32).max, verbose=False):
    """
    Sum a sequence of tensors in the TT format (of the same size), by computing a pyramid-like merging with a two-at-a-time summing step followed by truncation (TT-round)

    :param tts: A generator (or list) of TT-tensors, all with the same shape
    :param rounding: Intermediate tensors will be rounded to this value when climbing up the hierarchy
    :return: The summed TT-tensor

    """

    d = dict()
    result = None
    for i, elem in enumerate(tts):
        if result is None:
            result = 0 * tt.ones(elem.n)
        if verbose and i % 100 == 0:
            print("sum_and_compress: {}-th element".format(i))
        climb = 0  # For going up the tree
        add = elem
        while climb in d:
            if verbose:
                print("Hierarchy level:", climb, "- We sum", add.r, "and", d[climb].r)
            add = tr.core.round(d[climb] + add, eps=rounding, rmax=rmax)
            d.pop(climb)
            climb += 1
        d[climb] = add
    for key in d:
        result += d[key]
    return tr.core.round(result, eps=rounding, rmax=rmax)


def tt_dot(t1, t2):
    """
    Compute the dot (scalar) product between two TTs

    Reference:
    Alg. 3 in "Computation of the Response Surface in the Tensor Train data format" (S. Dolgov, B. N. Khoromskij, A. Litvinenko and H. G. Matthies), SIAM Journal on Uncertainty Quantification, 2014
    """

    def partial_dot_left(cores1, cores2, mu, Lprod):
        """
        Advance the dot product computation towards the left
        """
        d = len(cores1)
        assert len(cores2) == d
        assert 0 <= mu < d
        Ucore = ttm(cores1[mu], Lprod, mode=2, transpose=True)
        Vcore = cores2[mu]
        return np.dot(right_unfolding(Ucore), right_unfolding(Vcore).T)

    cores1 = tt.vector.to_list(t1)
    cores2 = tt.vector.to_list(t2)
    d = len(cores1)
    assert len(cores2) == d
    assert all([cores1[dim].shape[1] == cores2[dim].shape[1] for dim in range(d)])
    Rprod = np.array([[1]])
    for mu in range(d-1, -1, -1):
        Rprod = partial_dot_left(cores1, cores2, mu, Rprod)
    return np.squeeze(Rprod)


def left_unfolding(core):  # rs[mu] ns[mu] x rs[mu+1]
    return reshape(core, [-1, core.shape[2]])


def right_unfolding(core):  # rs[mu] x ns[mu] rs[mu+1]
    return reshape(core, [core.shape[0], -1])


def left_orthogonalize(cores, mu, recursive=False):
    """
    Makes the mu-th core left-orthogonal and pushes the R factor to its right core
    Note: this may change the size of the cores

    :param recursive: continue until the right end of the train
    """
    assert 0 <= mu < len(cores)-1
    coreL = left_unfolding(cores[mu])
    Q, R = np.linalg.qr(coreL, mode='reduced')
    cores[mu] = reshape(Q, cores[mu].shape[:-1] + (Q.shape[1], ))
    rightcoreR = right_unfolding(cores[mu+1])
    cores[mu+1] = reshape(np.dot(R, rightcoreR), (R.shape[0], ) + cores[mu+1].shape[1:])
    if recursive and mu < len(cores)-2:
        left_orthogonalize(cores, mu+1)
    return R


def right_orthogonalize(cores, mu, recursive=False):
    """
    Makes the mu-th core right-orthogonal and pushes the R factor to its left core
    Note: this may change the size of the cores

    :param recursive: continue until the left end of the train
    """
    assert 1 <= mu < len(cores)
    coreR = right_unfolding(cores[mu])
    L, Q = scipy.linalg.rq(coreR, mode='economic', check_finite=False)
    cores[mu] = reshape(Q, (Q.shape[0], ) + cores[mu].shape[1:])
    leftcoreL = left_unfolding(cores[mu-1])
    cores[mu-1] = reshape(np.dot(leftcoreL, L), cores[mu-1].shape[:-1] + (L.shape[1], ))
    # cores[mu-1] = reshape(np.dot(leftcoreL, R), cores[mu-1].shape)
    if recursive and mu > 1:
        right_orthogonalize(cores, mu-1)
    return L


def orthogonalize(cores, mu):
    """
    Apply left and right orthogonalizations to make the tensor mu-orthogonal

    :returns L, R:
    """
    L = np.array([[1]])
    R = np.array([[1]])
    for i in range(0, mu):
        R = left_orthogonalize(cores, i)
    for i in range(len(cores)-1, mu, -1):
        L = right_orthogonalize(cores, i)
    return R, L


def insert_dummies(t, modes, shape=1):
    """
    Inserts dummy dimensions, i.e. modes along which the tensor does not change

    :param t: input TT
    :param modes: indices where the dummy dimensions should be after insertion
    :param shape: new dimensions' sizes
    :return: a longer TT

    """

    print(t)
    N = t.d
    if not hasattr(modes, '__len__'):
        modes = [modes]
    if not hasattr(shape, '__len__'):
        shape = [shape]*len(modes)
    M = N + len(modes)
    assert len(modes) == len(shape)

    cores = tt.vector.to_list(t)
    newcores = [None]*M
    for n, core in zip(np.delete(np.arange(M), modes), cores):
        newcores[n] = core
    lastrank = 1
    counter = 0
    for n, sh in zip(modes, shape):
        lastrank = 1
        if n > 0:
            lastrank = newcores[n-1].shape[2]
        newcores[n] = np.repeat(np.eye(lastrank)[:, np.newaxis, :], sh, axis=1)
        counter += 1
    if newcores[0].shape[0] > 1:
        newcores[0] = newcores[0][0:1, ...]
    if newcores[-1].shape[2] > 1:
        newcores[-1] = newcores[-1][..., 0:1]
        # if n == 0:
        #     rleft = 1
        # else:
        #     rleft = newcores[n-1].shape[2]
        # if n == M-1:
        #     rright = 1
        # else:
        #     rright = newcores[n+1].shape[0]
        # newcores[n] = np.ones([rleft, sh, rright])
    return tt.vector.from_list(newcores)


def squeeze(t, modes=None):
    """
    Removes singleton dimensions

    :param t: A TT tensor
    :param modes: which modes to delete. By default, all that have size 1
    :return: Another TT tensor, without dummy (singleton) indices
    """

    if modes is None:
        modes = np.where(t.n == 1)[0]
    assert np.all(t.n[modes] == 1)

    cores = tt.vector.to_list(t)
    newcores = []
    curcore = None
    for mu in range(len(cores)):
        if cores[mu].shape[1] == 1 and mu in modes:
            if curcore is None:
                curcore = cores[mu]
            else:
                curcore = np.reshape(np.dot(curcore[:, 0, :], right_unfolding(cores[mu])), [curcore.shape[0], cores[mu].shape[1], -1], order='F')
        else:
            if curcore is None:
                curcore = cores[mu]
            else:
                curcore = np.reshape(np.dot(curcore[:, 0, :], right_unfolding(cores[mu])), [curcore.shape[0], cores[mu].shape[1], -1], order='F')
            newcores.append(curcore)
            curcore = None
    if curcore is not None:
        if len(newcores) > 0:
            assert newcores[-1].shape[1] > 1
            newcores[-1] = ttm(newcores[-1], curcore, mode=2, transpose=True)
        else:
            newcores.append(curcore)
    return tt.vector.from_list(newcores)


def choose(t, modes):
    """
    Dimensions in :param modes: get evaluated at their second index; dimensions not in :param modes: at their first

    :param t:
    :param modes:
    :return: the element evaluated
    """

    if not hasattr(modes, '__len__'):
        modes = [modes]
    modes = np.array(modes).astype(int)
    index = [0, ] * t.d
    for mode in modes:
        index[mode] = 1
    return t[index]


def random_tt(shape, ranks=1):
    """
    Generate a TT with random_sampling cores

    :param shape:
    :param ranks: an integer or list
    :return:

    """

    if not hasattr(ranks, "__len__"):
        ranks = [1, ] + [ranks, ] * (len(shape) - 1) + [1, ]
    assert len(ranks) == len(shape)+1
    assert ranks[0] == 1
    assert ranks[-1] == 1

    cores = []
    for i in range(len(shape)):
        cores.append(np.random.rand(ranks[i], shape[i], ranks[i + 1]))
    return tt.vector.from_list(cores)


def constant_tt(shape, fill=1):
    """
    Generate a TT filled with a constant value

    :param shape:
    :param fill: default is 1
    :return: a TT

    """

    cores = [np.ones([1, shape[0], 1]) * fill]
    for i in range(1, len(shape)):
        cores.append(np.ones([1, shape[i], 1]))
    return tt.vector.from_list(cores)


def separable_tt(ticks_list):
    """
    Build a separable TT from a list of discretized axes

    :param ticks_list: a list of vectors
    :return: the separable TT

    """

    return tt.vector.from_list([ticks[np.newaxis, :, np.newaxis] for ticks in ticks_list])


def gaussian_tt(shape, sigmas):
    """
    Build a multivariate Gaussian in the TT format with the given shape and sigmas
    """

    ticks_list = [np.exp(-np.linspace(-sh/2., sh/2., sh)**2 / (2*sig**2)) / np.sqrt(2*np.pi*sig**2) for sh, sig in zip(shape, sigmas)]
    return separable_tt(ticks_list)


def meshgrid(xi):
    """
    Similar to NumPy meshgrid

    :param xi: list of N axes, each must be a vector (or an integer n, then np.arange(n) is used)

    :return: A list of N TT tensors, equivalent to NumPy's meshgrid() for the given shape

    """

    grids = []
    ticks_list = []
    for ticks in xi:
        if not hasattr(ticks, '__len__'):
            ticks = np.arange(ticks, dtype=np.int)
        ticks_list.append(ticks)
    for dim in range(len(xi)):
        grid = [np.ones([1, len(ticks), 1], dtype=np.int) for ticks in ticks_list]
        grid[dim] = ticks_list[dim][np.newaxis, :, np.newaxis]
        grids.append(tt.vector.from_list(grid))
    return grids


def sum(t, modes=None, keepdims=False):
    """
    Sum all elements of a TT

    :param t:
    :param modes:
    :return: a scalar

    """

    N = t.d
    allsum = False
    if modes is None:
        allsum = True
        modes = np.arange(N)
    if not hasattr(modes, '__len__'):
        modes = [modes]

    cores = tt.vector.to_list(t)
    for n in modes:
        cores[n] = np.sum(cores[n], axis=1, keepdims=True)
    t = tt.vector.from_list(cores)
    if allsum:
        return np.asscalar(t.full())
    else:
        if keepdims:
            return t
        else:
            return tr.core.squeeze(t, modes=modes)


def mean(t):
    """
    Average all elements of a TT

    :param t:
    :return: a scalar

    """

    return tr.core.sum(t) / np.prod(t.n)


def sum_repeated(Xs, ys):
    """
    Returns a version of Xs that discards duplicates, and a version of ys that adds the contributions of all discarded Xs
    """

    b = np.ascontiguousarray(Xs).view(np.dtype((np.void, Xs.dtype.itemsize * Xs.shape[1])))
    _, idx, i = np.unique(b, return_index=True, return_inverse=True)
    zero = np.zeros(len(idx))
    np.add.at(zero, i, ys)
    return Xs[idx], zero


def convolve_along_axis(A, B, axes, mode='full'):
    """
    Convolve two arrays A and B along a pair of axes. The final number of dimensions is A.ndim + B.ndim - 1. The resulting dimensions are arranged similarly to NumPy's `tensordot` function, i.e. A's followed by B's

    :param A: an ndarray
    :param B: an ndarray
    :param axes: two integers
    :param mode: either 'full', 'same' or 'valid'. See NumPy's `convolve`
    :return: the convolved array. Dimensions: the ones from A (the axes[0]-th dimension has a different size, unless `mode` is 'same'), followed from the ones from B (with its axes[1]-th missing)

    """

    if mode not in ['full', 'same', 'valid']:
        raise ValueError("Choose one of the following modes: 'full', 'same', or 'valid'")

    # Put the target axis at the beginning
    idxA = np.concatenate([np.atleast_1d(axes[0]), np.delete(np.arange(A.ndim), axes[0])])
    idxB = np.concatenate([np.atleast_1d(axes[1]), np.delete(np.arange(B.ndim), axes[1])])
    At = np.transpose(A, idxA)
    Bt = np.transpose(B, idxB)

    # Pad and FFT arrays along their 1st axes
    Ash0 = At.shape[0]
    Bsh0 = Bt.shape[0]
    minsize = min(Ash0, Bsh0)
    maxsize = max(Ash0, Bsh0)
    At = np.concatenate([At, np.zeros([Bsh0 - 1] + list(At.shape[1:]))], axis=0)
    Bt = np.concatenate([Bt, np.zeros([Ash0 - 1] + list(Bt.shape[1:]))], axis=0)
    Atf = np.fft.fft(At, axis=0)
    Btf = np.fft.fft(Bt, axis=0)

    # Convolution theorem: multiply element-wise (but only along the target axes!)
    Atf = np.reshape(Atf, list(At.shape) + [1]*(Bt.ndim-1))
    Btf = np.reshape(Btf, [Btf.shape[0]] + [1]*(At.ndim-1) + list(Btf.shape[1:]))
    conv = Atf*Btf
    conv = np.fft.ifft(conv, axis=0)
    if mode == 'same':
        conv = conv[(minsize-1)//2:(minsize-1)//2+maxsize]
    elif mode == 'valid':
        if minsize == 1:
            end = None
        else:
            end = -(minsize-1)
        conv = conv[minsize-1:end]

    # Move the convolved axis back into its original position
    conv = np.transpose(conv, np.concatenate([(np.argsort(idxA)), np.arange(A.ndim, A.ndim+B.ndim-1)]))
    return np.real(conv)


def minimize(t, verbose=False, **kwargs):
    """
    Wrapper for ttpy's great min_tens function
    """

    val, point = tt.optimize.tt_min.min_tens(t, verb=verbose, **kwargs)
    return val, point


def maximize(t, verbose=False, **kwargs):
    """
    Wrapper for ttpy's great min_tens function
    """

    val, point = tt.optimize.tt_min.min_tens(-t, verb=verbose, **kwargs)
    return -val, point


def idx_to_qtt(Xs, I):
    """
    Convert multidimensional indices to QTT indexing with interleaving

    :param Xs: a P x N matrix of integers, each >= 0 and < I
    :param I: the size of each axis
    :return: a P x (log2(I)*N) matrix of binary integers

    """

    L = int(np.log2(I))
    assert I == 2**L
    Xs = np.array(np.unravel_index(Xs, [2]*L)).T
    Xs = np.transpose(Xs, [1, 0, 2])
    Xs = np.reshape(Xs, [Xs.shape[0], -1], order='F')
    return Xs


def idx_from_qtt(Xs, I):
    """
    Convert multidimensional indices from QTT indexing with interleaving

    :param Xs: a P x (log2(I)*N) matrix of binary integers
    :param I: the size of each axis
    :return: a P x N matrix

    """

    L = int(np.log2(I))
    assert I == 2**L
    N = Xs.shape[1] // L
    Xs = np.reshape(Xs, [Xs.shape[0], L, N])
    Xs = np.transpose(Xs, [1, 0, 2])
    Xs = np.einsum('ijk,i->jk', Xs, 2**np.arange(L)[::-1])
    return Xs


def derive(t, modes=None, order=1):
    """
    Given a TT, return its n-th order derivative along one or several modes. Each axis will lose `order` ticks

    :param t:
    :param modes: an integer, or a list of integers. By default, all modes are derived
    :param order: an integer >= 0. Default is 1
    :return: the derived TT

    """

    N = t.d
    if modes is None:
        modes = np.arange(N)
    if not hasattr(modes, '__len__'):
        modes = [modes]*N
    assert len(modes) <= N
    assert np.min(modes) >= 0 and np.max(modes) < N

    cores = tt.vector.to_list(t)
    for n in range(N):
        if n in modes:
            cores[n] = np.diff(cores[n], n=order, axis=1)
    return tt.vector.from_list(cores)


def transpose(t, order, **kwargs):
    """
    Like NumPy's transpose(). Uses cross-approximation
    """

    ticks_list = [np.arange(t.n[n]) for n in range(t.d)]
    ticks_list = [ticks_list[o] for o in order]

    def fun(Xs):
        Xs = Xs.copy()
        Xs = Xs[:, np.argsort(order)]
        return tr.core.sparse_reco(t, Xs)

    return tr.core.cross(ticks_list=ticks_list, fun=fun, **kwargs)


def transpose(t, order, eps):
    N = t.d
    idx = np.empty(len(order))
    idx[order] = np.arange(len(order))

    k = 0
    while True:
        # Find next inversion
        nk = k
        while (nk < N - 1) and (idx[nk] < idx[nk + 1]):
            nk += 1
        if nk == N - 1:
            break
        print(k, nk)
        k = nk
        idx[[k, k + 1]] = idx[[k + 1, k]]
        t = shift_mode(t, k, shift=1, eps=eps)
        k = max(k - 1, 0)
    return t


def shift_mode(t, n, shift, eps=1e-3, mode='swap', **kwargs):
    """
    Shift a mode back or forth within a TT

    :param t:
    :param n: which mode to move
    :param shift: how many positions to move. If positive move right, if negative move left
    :param eps: prescribed relative error tolerance. If 'same' (default), ranks will be kept no larger than the original
    :return: the transposed tensor

    """

    N = t.d
    assert 0 <= n + shift < N
    assert mode in ('cross', 'swap')

    if shift == 0:
        return copy.deepcopy(t)

    if mode == 'cross':
        order = list(range(N))
        del order[n]
        if shift < 0:
            order.insert(n+shift, n)
        else:
            order.insert(n+shift, n)
        print('order:', order)
        return transpose(t, order, eps=eps, **kwargs)

    cores = copy.deepcopy(tt.vector.to_list(t))
    tr.core.orthogonalize(cores, n)
    sign = np.sign(shift)
    for i in range(n, n + shift, sign):
        if sign == 1:
            c1 = i
            c2 = i+1
            left_ortho = True
        else:
            c1 = i-1
            c2 = i
            left_ortho = False
        R1 = cores[c1].shape[0]
        R2 = cores[c1].shape[2]
        R3 = cores[c2].shape[2]
        I1 = cores[c1].shape[1]
        I2 = cores[c2].shape[1]
        sc = np.einsum('iaj,jbk->iabk', cores[c1], cores[c2], optimize=False)
        sc = np.transpose(sc, [0, 2, 1, 3])
        sc = np.reshape(sc, [sc.shape[0]*sc.shape[1], sc.shape[2]*sc.shape[3]])
        if eps == 'same':
            left, right = tr.core.truncated_svd(sc, eps=0, rmax=R2, left_ortho=left_ortho)
        elif eps >= 0:
            left, right = tr.core.truncated_svd(sc, eps=eps/np.sqrt(np.abs(shift)), left_ortho=left_ortho)
        else:
            raise ValueError("Relative error '{}' not recognized".format(eps))
        newR2 = left.shape[1]
        cores[c1] = np.reshape(left, [R1, I2, newR2])
        cores[c2] = np.reshape(right, [newR2, I1, R3])
    return tt.vector.from_list(cores)


def stack(ts, axis, **kwargs):
    """
    Similar to NumPy's stack function
    """

    assert len(axis) == 1

    for i in range(len(ts)):
        ts[i] = tr.core.insert_dummies(ts[i], axis)
    return tr.core.concatenate(ts, axis=axis, **kwargs)


def concatenate(ts, axis, eps=0, rmax=np.iinfo(np.int32).max, verbose=False):
    """
    Given a list of TT tensors, stack them along a specified dimension axis

    :param ts: a list of TTs of the same dimension N
    :param axis: an integer. All TTs must have the same shape, except along this axis
    :param eps: default is 0
    :param rmax: default is infinite
    :param verbose:
    :return: a TT, the concatenation

    """

    def concatenate_pair(t1, t2, axis):

        if t1 is None:
            return copy.deepcopy(t2)
        N = t1.d
        if t2.d != N:
            raise ValueError('For concatenation, both tensors must have the same dimensionality')
        check = list(range(N))
        del check[axis]
        if not all(t1.n[check] == t2.n[check]):
            raise ValueError('For concatenation, both tensors must have equal sizes along all (but one) modes')
        cores1 = tt.vector.to_list(t1)
        cores2 = tt.vector.to_list(t2)
        cores = []
        for n in range(N):
            if n != axis:
                core = np.zeros([cores1[n].shape[0] + cores2[n].shape[0], cores1[n].shape[1], cores1[n].shape[2] + cores2[n].shape[2]])
                core[:cores1[n].shape[0], :, :cores1[n].shape[2]] = cores1[n]
                core[cores1[n].shape[0]:, :, cores1[n].shape[2]:] = cores2[n]
            else:
                core = np.zeros([cores1[n].shape[0] + cores2[n].shape[0], cores1[n].shape[1] + cores2[n].shape[1], cores1[n].shape[2] + cores2[n].shape[2]])
                core[:cores1[n].shape[0], :cores1[n].shape[1], :cores1[n].shape[2]] = cores1[n]
                core[cores1[n].shape[0]:, cores1[n].shape[1]:, cores1[n].shape[2]:] = cores2[n]
            if n == 0:
                core = np.sum(core, axis=0, keepdims=True)
            elif n == N-1:
                core = np.sum(core, axis=2, keepdims=True)
            cores.append(core)
        return tt.vector.from_list(cores)

    # Tree-like procedure, as in sum_and_compress
    d = dict()
    result = ts[0]
    for i in range(1, len(ts)):
        if verbose and i % 100 == 0:
            print("concatenate: {}-th element".format(i))
        climb = 0  # For going up the tree
        new = ts[i]
        while climb in d:
            if verbose:
                print("Hierarchy level:", climb, "- We concatenate", new.n, "and", d[climb].n)
            new = concatenate_pair(d[climb], new, axis=axis)
            if eps > 0 or rmax is not None:
                new = tt.vector.round(new, eps, rmax=rmax)
            d.pop(climb)
            climb += 1
        d[climb] = new
    for key in sorted(d.keys())[::-1]:
        result = concatenate_pair(result, d[key], axis=axis)
    return result


def get_row(ttm, row):
    """
    Read a row from a TT matrix

    :param ttm: a TT matrix
    :param row: a list encoding a multiindex
    :return: a TT vector
    """

    slices = [slice(r, r+n*m, n) for r, n, m in zip(row, ttm.n, ttm.m)]
    return ttm.tt[slices]


def get_col(ttm, col):
    """
    Read a column from a TT matrix

    :param ttm: a TT matrix
    :param col: a list encoding a multiindex
    :return: a TT vector
    """

    slices = [slice(c*n, c*n+n) for c, n, m in zip(col, ttm.n, ttm.m)]
    return ttm.tt[slices]
