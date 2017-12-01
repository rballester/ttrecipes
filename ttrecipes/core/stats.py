"""
Statistical tools in the TT format: random sampling schemes, marginal and joint PDFs, etc.
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


def random_sampling(pdf, P=1):
    """
    Generate P points from a joint PDF distribution in the TT format. We use Gibbs sampling

    :param pdf: a TT (does not have to sum 1)
    :param P: how many samples to draw (default: 1)
    :return Xs: a matrix of size P x pdf.d

    """

    def from_matrix(M):
        """
        Treat each row of M as a pdf and select a column per row according to it
        """

        M = M.astype(np.float) / np.sum(M, axis=1)[:, np.newaxis]
        M = np.hstack([np.zeros([M.shape[0], 1]), M])
        M = np.cumsum(M, axis=1)
        thresh = np.random.rand(M.shape[0])
        M -= thresh[:, np.newaxis]
        shiftand = np.logical_and(M[:, :-1] <= 0, M[:, 1:] > 0)  # Find where the sign switches
        return np.where(shiftand)[1]

    N = pdf.d
    Xs = np.zeros([P, N])
    rights = [np.array([1])]
    cores = tt.vector.to_list(pdf)
    for core in cores[::-1]:
        rights.append(np.dot(np.sum(core, axis=1), rights[-1]))
    rights = rights[::-1]
    lefts = np.ones([P, 1])

    for mu in range(N):
        fiber = np.einsum('ijk,k->ij', cores[mu], rights[mu + 1])
        per_point = np.einsum('ij,jk->ik', lefts, fiber)
        rows = from_matrix(per_point)
        Xs[:, mu] = rows
        lefts = np.einsum('ij,jik->ik', lefts, cores[mu][:, rows, :])

    return Xs


def LHS(shape, P, balance=True):
    """
    Uses latin hypercube sampling (LHS) to get P points (they may be repeated) from a grid of the given shape

    :param shape:
    :param P: The number of points to draw
    :param balance: If True, try to put an as equal as possible number of samples on each slice. Otherwise, only guarantee that each slice contains at least 1 sample
    :return: a P x N matrix

    """

    N = len(shape)
    if P < np.max(shape):
        raise ValueError("LHS on this tensor needs at least {} samples".format(np.max(shape)))

    # Strategy: a) one (or as many as possible) passes ensuring all slices are populated; b) the rest randomly; c) shuffle
    indices = np.empty((P, N), dtype=np.int)
    for i, sh in enumerate(shape):
        if balance:
            part1 = np.repeat(np.arange(sh), (P // sh))
            part2 = np.random.choice(sh, P - len(part1), replace=False)
        else:
            part1 = np.random.permutation(sh)[:(P % sh)]
            part2 = np.random.randint(0, sh, P - len(part1))
        indices[:, i] = np.concatenate([part1, part2])
        np.random.shuffle(indices[:, i])
    return indices


def marginals(t):
    """
    Return all N marginals from an N-dimensional PDF

    :param t: a TT representing a PDF
    :return: a list of vectors, each of which sums 1

    """

    N = t.d
    cores = tt.vector.to_list(t)
    summed = [np.sum(core, axis=1, keepdims=True) for core in cores]
    result = []
    for i in range(N):
        tmp = summed[i]
        summed[i] = cores[i]
        result.append(np.squeeze(tr.core.squeeze(tt.vector.from_list(summed)).full()))
        result[-1] /= np.sum(result[-1])  # Must sum 1
        summed[i] = tmp
    return result


def fit_to_marginals(t, marginals):
    """
    Given a copula in the TT format and a list of associated marginal PDFs (vectors), compute the equivalent joint PDF
    """

    N = t.d
    cores = tt.vector.to_list(t * (1./tr.core.sum(t)))
    curs = tr.core.marginals(t)
    newcores = []
    for i, core in enumerate(cores):
        marg = marginals[i]/np.sum(marginals[i])
        cmarg = np.concatenate([np.array([0]), np.cumsum(marg)])
        cur = curs[i]
        I = t.n[i]
        ccur = np.concatenate([np.array([0]), np.cumsum(cur)])
        interp = scipy.interpolate.interp1d(ccur, np.linspace(0, 1, I+1), assume_sorted=True, fill_value='extrapolate')(cmarg)
        ccore = np.concatenate([np.zeros([core.shape[0], 1, core.shape[2]]), np.cumsum(core, axis=1)], axis=1)
        interpcore = scipy.interpolate.interp1d(np.linspace(0, 1, I+1), ccore, axis=1, assume_sorted=True, fill_value='extrapolate')
        interpcore = interpcore(interp)
        newcores.append(np.diff(interpcore, axis=1))
    return tt.vector.from_list(newcores)


def corr_to_pdf(corr, marginals, fraction=1e-4, eps=1e-6, verbose=False, **kwargs):
    """
    Build a Gaussian copula for a given correlation matrix, and turn it into a PDF that additionally satisfies the given marginals

    TODO work in progress...

    :param corr: an N x N matrix
    :param marginals: a list of N vectors
    :param fraction: the bounds will be set so that this proportion of the normal PDFs will be left out. Default is 1e-4
    :return: two TTs: the PDF and its copula

    """

    N = len(marginals)
    # corr_Z = 2*np.sin(np.pi/6*corr)  # NORTA transform
    corr_Z = corr
    chol_corr_Z = np.linalg.cholesky(corr_Z).T
    chol_corr_Z_inv = np.linalg.inv(chol_corr_Z)

    def fun(Xs):
        N = Xs.shape[1]
        Xs_g = Xs.dot(chol_corr_Z_inv)
        result = np.exp(-1./2 * np.einsum('ij,ij->i', Xs_g, Xs_g))/np.sqrt((2*np.pi)**N)
        return result

    normdist = scipy.stats.norm(0, 1)
    ticks_list = [np.linspace(normdist.ppf(fraction/2), normdist.ppf(1-fraction/2), len(marginals[i])) for i in range(N)]
    copula = tr.core.cross(ticks_list, fun, mode='array', eps=eps, verbose=verbose, **kwargs)

    pdf = tr.core.fit_to_marginals(copula, marginals).round(eps)
    pdf *= (1./tr.core.sum(pdf))
    return pdf, copula
