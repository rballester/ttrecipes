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
                        print_function, unicode_literals)
from unittest import TestCase
import numpy as np
import tt

import ttrecipes as tr


class TestCompletion(TestCase):

    def test_categorical(self):
        N = 4
        Is = [5, ]*N
        ranks = 3
        gt = tr.core.random_tt(Is, ranks)
        P = int(np.prod(gt.n)/10)
        Xs = tr.core.LHS(Is, P)
        ys = np.array([gt[x] for x in Xs])
        completed = tr.core.categorical_ALS(Xs, ys, shape=Is, ranks=ranks, verbose=False)
        reco = np.array([completed[x] for x in Xs])
        self.assertLessEqual(np.linalg.norm(reco - ys)/np.linalg.norm(ys), 0.25)

    def test_continuous(self):
        N = 4
        Is = [15, ]*N
        Rs = [1] + [4, ]*(N-1) + [1]
        Ss = [3, ]*N
        cores = tt.vector.to_list(tr.core.random_tt(Ss, Rs))
        Us = tr.core.generate_bases("legendre", Is, Ss)
        cores = [np.einsum('ijk,lj->ilk', c, U) for c, U in zip(cores, Us)]
        gt = tt.vector.from_list(cores)
        P = int(np.prod(Is) / 10)
        Xs = tr.core.LHS(gt.n, P)
        ys = tr.core.sparse_reco(gt, Xs)
        completed = tr.core.pce_interpolation(Xs, ys, shape=Is, ranks=Rs, ranks2=Ss, maxswp=10, verbose=True)
        reco = tr.core.sparse_reco(completed, Xs)
        self.assertLessEqual(np.linalg.norm(reco - ys) / np.linalg.norm(ys), 1e-5)
