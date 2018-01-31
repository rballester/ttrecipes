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
import scipy.stats

import ttrecipes as tr


class TestCopula(TestCase):

    def test_copula(self):

        # Parameters
        P = 10000
        N = 2
        I = 64
        fraction = 1e-3

        # Correlation matrix
        corr = np.zeros([2, 2])
        corr[0, 0] = 1
        corr[1, 1] = 1
        corr[0, 1] = 0.5
        corr[1, 0] = 0.5

        targetdists = [scipy.stats.uniform, scipy.stats.lognorm(0.25, 0.5)]

        marginals = [targetdists[i].pdf(np.linspace(targetdists[i].ppf(fraction / 2), targetdists[i].ppf(1 - fraction / 2), I)) for i in range(N)]
        marginals = [m/m.sum() for m in marginals]
        pdf, copula = tr.core.corr_to_pdf(corr, marginals)

        # Draw some samples according to the PDF we obtained
        samples = tr.core.random_sampling(pdf, P=P)

        # Empirical correlation
        result_corr = np.corrcoef(samples, rowvar=False)

        # Check correlation is as expected
        self.assertLessEqual(np.linalg.norm(result_corr - corr) / np.linalg.norm(corr), 0.05)

        # Check all marginals are as expected
        result_marginals = tr.core.marginals(pdf)
        for i in range(N):
            self.assertLessEqual(np.linalg.norm(result_marginals[i] - marginals[i]) / np.linalg.norm(marginals[i]), 1e-3)
