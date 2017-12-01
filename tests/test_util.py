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
import tt
np.random.seed(0)
import ttrecipes as tr


class TestUtil(TestCase):

    def test_tt_sum(self):

        P = 10

        def create_generator():
            for i in range(P):
                yield i*tt.ones([1])

        generator = create_generator()
        result = tr.core.sum_and_compress(generator)
        self.assertAlmostEqual(np.squeeze(result.full()), (P-1)*(P/2))

    def test_convolve_along_axis(self):
        for i in range(100):
            N1 = np.random.randint(1, 5)
            N2 = np.random.randint(1, 5)
            Is = np.random.randint(1, 8, N1)
            Js = np.random.randint(1, 8, N2)
            A = np.random.rand(*list(Is))
            B = np.random.rand(*list(Js))
            axes = [np.random.randint(N1), np.random.randint(N2)]
            idx1 = [np.random.randint(0, sh) for sh in A.shape]
            idx1[axes[0]] = slice(None)
            idx2 = [np.random.randint(0, sh) for sh in B.shape]
            idx2[axes[1]] = slice(None)
            for mode in ['full', 'same', 'valid']:
                gt = np.convolve(A[idx1], B[idx2], mode=mode)
                reco = tr.core.convolve_along_axis(A, B, axes=axes, mode=mode)
                idx2_copy = idx2.copy()
                del idx2_copy[axes[1]]
                reco = reco[idx1 + idx2_copy]
                self.assertAlmostEqual(np.linalg.norm(reco - gt) / np.linalg.norm(gt), 0)
