# -----------------------------------------------------------------------------
# Authors:      Rafael Ballester-Ripoll <rballester@ifi.uzh.ch>
#
# Copyright:    TensorChart project (c) 2016-2017
#               VMMLab - University of Zurich
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
