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
from future.builtins import range

from unittest import TestCase
import numpy as np
import tt

import ttrecipes as tr


class TestSparse(TestCase):

    def test_auxiliary(self):

        for i in range(10):

            P = 100
            N = np.random.randint(1, 5)
            Is = np.random.randint(1, 3, N)
            Xs = [np.random.randint(0, I, P) for I in Is]
            Xs = np.array(Xs).T
            ys = np.random.rand(P)
            Xs, ys = tr.core.sum_repeated(Xs, ys)

            # sparse_covariance()
            D = np.zeros(Is)
            D[list(Xs.T)] = ys
            D = np.reshape(D, [D.shape[0], -1])
            gt = D.dot(D.T)
            reco = tr.core.sparse_covariance(Xs, ys, nrows=Is[0])
            self.assertAlmostEqual(np.linalg.norm(gt - reco) / np.linalg.norm(gt), 0)

            # full_times_sparse()
            F = np.random.rand(Is[0], Is[0])
            D = np.zeros(Is)
            D[list(Xs.T)] = ys
            D = np.reshape(D, [Is[0], -1])
            gt = F.dot(D)
            gt = np.reshape(gt, Is)
            Xs, ys = tr.core.full_times_sparse(F, Xs, ys)
            reco = np.zeros(Is)
            reco[list(Xs.T)] = ys
            self.assertAlmostEqual(np.linalg.norm(gt - reco) / np.linalg.norm(gt), 0)

    def test_sparse(self):

        N = 16
        P = 2**N
        Xs = np.array(np.unravel_index(np.arange(P), [2, ] * N)).T
        ys = 1. / (np.sum(Xs, axis=1) + 1)  # Hilbert tensor

        for i in range(1, 5):

            eps = 10**(-i)
            t = tr.core.sparse_tt_svd(Xs, ys, shape=[2, ] * N, rmax=5, verbose=False, eps=eps)
            reco = t.full()
            gt = np.zeros([2, ] * N)
            gt[list(Xs.T)] = ys
            error = np.linalg.norm(reco - gt) / np.linalg.norm(gt)
            self.assertLessEqual(error, eps)

            t = tt.vector(gt, eps=eps)
            reco = t.full()
            error_tt = np.linalg.norm(reco - gt) / np.linalg.norm(gt)
            self.assertLessEqual((error - error_tt) / error_tt, 1.5)
