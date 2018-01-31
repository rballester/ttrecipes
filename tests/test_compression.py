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

import ttrecipes as tr


class TestCompression(TestCase):

    def test_tt_svd(self):
        ntries = 1000
        for i in range(ntries):
            N = np.random.randint(2, 5)
            Is = np.random.randint(2, 8, N)
            X = np.random.rand(*list(Is))
            eps = np.random.rand()*0.1
            X_t = tr.core.tt_svd(X, eps)
            reco = X_t.full()
            rel = np.linalg.norm(reco - X) / np.linalg.norm(X)
            self.assertLessEqual(rel, eps)

    def test_round(self):
        ntries = 100
        for i in range(ntries):
            N = np.random.randint(3, 5)
            shape = [np.random.randint(1, 15) for n in range(N)]
            ranks = [1] + [np.random.randint(1, 10) for n in range(N-1)] + [1]
            t = tr.core.random_tt(shape, ranks)
            eps = np.random.rand()
            t2 = tr.core.round(t, eps=eps, verbose=False)
            gt = t.full()
            reco = t2.full()
            reps = np.linalg.norm(reco - gt) / np.linalg.norm(gt)
            self.assertLessEqual(reps, eps)
