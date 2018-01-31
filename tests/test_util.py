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

    def test_shift_mode(self):
        N = 6
        Is = np.random.randint(2, 10, N)
        Rs = [1] + [3]*(N-1) + [1]
        t = tr.core.random_tt(shape=Is, ranks=Rs)
        eps = 1e-3
        for dim in range(N):
            tshift = tr.core.shift_mode(t, dim, -dim, eps=eps)  # Put dim at the beginning
            tshift2 = tr.core.shift_mode(tshift, 0, dim, eps=eps)  # Swap back dim where it was
            self.assertLessEqual(tt.vector.norm(tshift2 - t) / tt.vector.norm(t), 2*eps)

    def test_concatenate(self):
        N = np.random.randint(4, 10)
        n = np.random.randint(N)
        Is1 = [4]*N
        Is2 = [4]*N
        Is2[n] = 6
        Rs = [1] + [3]*(N-1) + [1]
        t1 = tr.core.random_tt(shape=Is1, ranks=Rs)
        t2 = tr.core.random_tt(shape=Is2, ranks=Rs)
        t = tr.core.concatenate([t1, t2], axis=n)
        self.assertAlmostEqual(tt.vector.norm(t1 - t[[slice(None)]*n + [slice(Is1[n])] + [slice(None)]*(N-n-1)]), 0)
        self.assertAlmostEqual(tt.vector.norm(t2 - t[[slice(None)]*n + [slice(Is1[n], None)] + [slice(None)]*(N-n-1)]), 0)
