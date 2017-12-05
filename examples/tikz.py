# -----------------------------------------------------------------------------
# Authors:      Rafael Ballester-Ripoll <rballester@ifi.uzh.ch>
#
# Copyright:    ttrecipes project (c) 2017
#               VMMLab - University of Zurich
# -----------------------------------------------------------------------------

from __future__ import (absolute_import, division,
                        print_function, unicode_literals, )
import numpy as np
import tt
import ttrecipes as tr
np.random.seed(1)


X = np.random.randn(3, 4, 3, 4)
t = tt.vector(X, eps=0.0)
output_file = 'tikz_example.tex')
selected = [1, 3, 0, 1]

tr.tikz.draw(t=t, output_file=output_file, rank_labels=True, core_sep=5.5, colormap='autumn', figure_decorator=False, letter='T', selected=selected)
