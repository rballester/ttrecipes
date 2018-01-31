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


class TestSampling(TestCase):

    def test_random_sampling(self):

        N = 3
        I = 8
        P = 100

        # A joint PDF
        # Rs = [1, ] + [10, ]*(N-1) + [1, ]
        # pdf = tt.vector.from_list([np.random_sampling.rand(Rs[n], I, Rs[n+1]) for n in range(N)])
        # Rs = np.ones(N+1)
        # pdf = tt.vector.from_list([np.exp(-np.linspace(-7.5, 7.5, I)**2)[np.newaxis, :, np.newaxis]]*N)
        pdf = np.zeros((I,) * N)
        for i in range(I):
            pdf[i, :, i] = 1
        pdf = tt.vector(pdf, eps=0)

        Xs = tr.core.random_sampling(pdf, P)
        self.assertAlmostEqual(np.linalg.norm(Xs[:, 0] - Xs[:, 2]), 0)
