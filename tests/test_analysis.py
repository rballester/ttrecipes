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

import ttrecipes as tr


class TestAnalysis(TestCase):

    def test_best_subspace(self):

        N = 5
        ntries = 100

        def fun(Xs):
            return np.sum(np.sin(Xs*2)*np.array([[-2, 2, 1, 4, -2]]), axis=1)**2

        t = tr.core.cross(ticks_list=[(np.linspace(0, 1, 64)) for i in range(N)], fun=fun, mode='array')
        best_subspace = tr.core.best_subspace(t, ndim=2)[0]
        best_var = np.var(t[best_subspace].full())

        # Try several random_sampling slices and verify that their variance is never larger
        for i in range(ntries):
            ind = list(np.random.randint(0, 64, t.d))
            for where in np.random.choice(N, 2):
                ind[where] = slice(None)
            self.assertLessEqual(np.var(t[ind].full()), best_var)
