# -----------------------------------------------------------------------------
# Authors:      Rafael Ballester-Ripoll <rballester@ifi.uzh.ch>
#
# Copyright:    ttrecipes project (c) 2017
#               VMMLab - University of Zurich
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
