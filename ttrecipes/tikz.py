"""
TT visualization using tikz
"""

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
import matplotlib.pyplot as plt


def draw(t, output_file, rank_labels=True, rank_values=False, core_labels=True, letter='T', figure_decorator=True, core_sep=5, slice_sep_x=0.15, slice_sep_y=0.1, factor=0.2, colormap='viridis', selected=None):
    """
    Programmatically depict a TT in the form of a tikzpicture (saved as a text file)

    :param t: a TT
    :param output_file: a .tex to save the figure's code
    :param rank_labels: if True (default), rank names are shown
    :param rank_values: if True, rank values are shown. Default is false
    :param core_labels: if True (default), each core name is shown
    :param letter: if core_labels, this letter will be used for each core. Default is 'T'
    :param figure_decorator: if True (default), the tikzfigure is placed into a "figure" environment with an informative caption
    :param core_sep: gap between cores
    :param slice_sep_x: x gap between slices in a core
    :param slice_sep_y: y gap between slices in a core
    :param factor: scales everything (except fonts)
    :param colormap: defines the matplotlib's colormap to use (default is 'viridis')
    :param selected: a list, each element denotes one core slice to highlight (if None, all will be)

    """

    if rank_values and not rank_labels:
        raise ValueError('\"rank_values\" requires \"rank_labels\"')

    N = t.d
    Is = t.n
    Rs = t.r

    if selected is None:
        selected = [None]*N
    for i in range(N):
        if selected[i] is None:
            selected[i] = np.arange(Is[i])
        if not hasattr(selected[i], '__len__'):
            selected[i] = [selected[i]]

    # Design decisions
    fill_opacity = 1./3
    cmap = plt.cm.get_cmap(colormap, N*1.33)

    string = ''
    if figure_decorator:
        string += '\\begin{figure}\n'
        string += '\\centering\n'
    string += '\\begin{tikzpicture}\n'

    core_corner = np.array([0., 0.])
    for n in range(N):
        for i in range(Is[n]-1, -1, -1):

            if i == Is[n]-1:
                if not rank_labels:
                    above_label = ''
                elif n == N-1 and Rs[N] == 1:
                    above_label = 'node[above, opacity=1]{{1}}'
                else:
                    if rank_values:
                        value = ' = {}'.format(Rs[n+1])
                    else:
                        value = ''
                    above_label = 'node[above, opacity=1]{{$R_{}{}$}}'.format(n+1, value)
            else:
                above_label = ''
            if i == 0:
                if core_labels:
                    below_label = 'node[below, opacity=1, font=\\large]{{$\\mathcal{{{}}}^{{({})}}$}}'.format(letter, n+1)
                else:
                    below_label = ''
                if rank_values:
                    rotation = 'above, rotate=90'
                else:
                    rotation = 'left'
                if not rank_labels:
                    left_label = ''
                elif n == 0 and Rs[n] == 1:
                    left_label = 'node[{}, opacity=1]{{1}}'.format(rotation)
                else:
                    if rank_values:
                        value = ' = {}'.format(Rs[n])
                    else:
                        value = ''
                    left_label = 'node[{}, opacity=1]{{$R_{}{}$}}'.format(rotation, n, value)
            else:
                below_label = ''
                left_label = ''
            # if selected is None or selected[n] is None or i == selected[n]:
            if i in selected[n]:
                opacity = 1
            else:
                opacity = 0.25

            slice_corner = core_corner + np.array([i*slice_sep_x, i*slice_sep_y - factor*Rs[n]/2.])
            color = np.array(cmap(n))
            color[:3] = 1 - (1-color[:3])*fill_opacity
            string += '\definecolor{{fillcolor}}{{rgb}}{{{:.3f},{:.3f},{:.3f}}}'.format(*list(color))

            string += '\\draw [draw=black, fill=fillcolor, fill opacity={:.3f}] ({:.3f},{:.3f}) --{} ({:.3f},{:.3f}) -- ({:.3f},{:.3f}) --{} ({:.3f},{:.3f}) --{} ({:.3f},{:.3f});\n'.format(opacity, slice_corner[0], slice_corner[1], below_label, slice_corner[0] + factor*Rs[n+1], slice_corner[1], slice_corner[0] + factor*Rs[n+1], slice_corner[1] + factor*Rs[n], above_label, slice_corner[0], slice_corner[1] + factor*Rs[n], left_label, slice_corner[0], slice_corner[1])

        core_corner[0] += factor*(Rs[n+1] + core_sep)

    string += '\\end{tikzpicture}\n'
    if figure_decorator:
        if rank_values:
            total_elements = ', compressed using {} elements'.format(np.sum([Rs[n]*Is[n]*Rs[n+1] for n in range(N)]))
        else:
            total_elements = ''
        string += '\\caption{{A {} tensor train{}.'.format('$' + ' \\times '.join([str(I) for I in Is]) + '$', total_elements)
        if selected is not None:
            string += ' Multiplying the highlighted slices yields the element $\\mathcal{{T}}[{}]$.'.format(','.join([str(s) for s in selected]))
        string += '}}\n\\end{figure}\n'

    with open(output_file, 'w') as f:
        print(string, file=f)
