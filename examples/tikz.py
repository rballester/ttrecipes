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
                        print_function, unicode_literals, )
import numpy as np
import tt
import ttrecipes as tr
import ttrecipes.tikz
np.random.seed(1)


X = np.random.randn(3, 4, 3, 4)
t = tt.vector(X, eps=0.0)
output_file = 'tikz_example.tex'
selected = [1, 3, 0, 1]

tr.tikz.draw(t=t, output_file=output_file, rank_labels=True, core_sep=5.5, colormap='autumn', figure_decorator=False, letter='T', selected=selected)
