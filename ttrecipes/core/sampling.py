"""
Sampling schemes for completion/surrogate modeling
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


def random_sampling(pdf, P=1):
    """
    Generate P points (with replacement) from a joint PDF distribution represented by a TT tensor. We use Gibbs sampling

    :param pdf: a TT (does not have to sum 1, it will be normalized)
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
