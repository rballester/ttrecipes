"""
Tools to work with probability density functions (PDFs) in the TT format: marginals, copulas and joint PDFs, etc.
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
import scipy

import ttrecipes as tr


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
