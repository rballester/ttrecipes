"""
Visualization of TT tensors using matplotlib
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
import tt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
import copy
import collections
import six

import ttrecipes as tr


def navigation(t, names=None, ticks_list=None, output_name='y', coords=None, ys=None, gt_range=-1, dims=None, diagrams=None, focus='center'):
    """
    Create an interactive chart to navigate a TT surrogate model. The chart contains different approximated subspaces
    that pass through a focus (the "focus focus"). The user can click and drag on the plots to vary this focus

    Other features:
    - Buttons to bring you to the surrogate's maximum, minimum, random point, or closest groundtruth point (if given)
    - Double click on a plot to find the associated subspace with maximal (left button) resp. minimal (right button)
    variance

    :param t: an N-dimensional TT
    :param names: N strings with the variable names. If None (default), generic names will be given
    :param ticks_list: N vectors containing the ticks for each axis. If None (default), [0, ..., In-1] will be used
    for each tensor size In
    :param output_name: the output variable's name. Default is 'y'
    :param coords: optionally, a P x N matrix of groundtruth values to show along the surrogate
    :param ys: the groundtruth values. Must be provided if `coords` is given
    :param dims: how the dimensions are grouped together to define subspaces (list of lists). If None, N fibers will
    be shown
    :param gt_range: if 0 or above, groundtruth samples will be shown on subspaces as long as they fall close enough
    to them. The threshold distance "close enough" is defined by gt_range
    :param dims: a list of lists, each containing a subset of dimensions to visualize. For example [[0], [1,
    2]] will produce two plots: one 1D with variable 0 and one 2D with variables 1 and 2. Variables may be identified
    either by name or by index. They may be repeated across several plots.
    :param diagrams: the type of diagram for each set of dimensions (list of strings). Can be "plot", "image" or
    "surface" (the latter is highly experimental)
    :param focus: where the subspaces pass through initially. Can be:
        - 'center' (default): the central focus of the tensor is chosen
        - A list of integers
        - 'maximum' or 'minimum': the TT's maximum or minimum will be estimated
        - 'sample': (requires `coords`): the sample that is the closest to the data set's barycenter will be chosen

    Examples:
    navigation(t, dims=[[0], [1], [2]], diagrams=["plot", "plot", "plot"])
    navigation(t, names=['L', 'h', 'M'], dims=[[2, 0], [1]], diagrams=["image", "plot"])

    """

    plt.style.use('seaborn')

    # Process and check arguments
    N = t.d
    if ticks_list is None:
        ticks_list = [np.arange(t.n[n], dtype=np.int) for n in range(N)]
    if names is None:
        names = ['$x_{}$'.format(n) for n in range(N)]
    if coords is None and ys is not None:
        raise ValueError('Please provide a P x N matrix of coordinates')
    if coords is not None and ys is None:
        raise ValueError('Please provide a vector of P values')
    if coords is None:
        Xs = None
    else:
        Xs = tr.core.coordinates_to_indices(coords, ticks_list=ticks_list)
    estimated_minimum, estimated_minimum_point = tr.core.minimize(t, nswp=5, rmax=5)
    estimated_maximum, estimated_maximum_point = tr.core.maximize(t, nswp=5, rmax=5)
    if ys is not None:
        estimated_minimum = np.min(ys)
        estimated_minimum_point = Xs[np.argmin(ys), :]
        estimated_maximum = np.max(ys)
        estimated_maximum_point = Xs[np.argmax(ys), :]
    if focus is 'center':
        focus = (np.array(t.n) / 2).astype(int)
    if focus is 'maximum':
        focus = copy.deepcopy(estimated_maximum_point)
    elif focus is 'minimum':
        focus = copy.deepcopy(estimated_minimum_point)
    elif focus is 'sample':  # Focus on the sample that is closest to the data set's barycenter
        assert coords is not None
        Xs_01 = Xs.astype(float) / (t.n[np.newaxis, :] - 1)
        barycenter = np.mean(Xs_01, axis=0)
        focus = Xs[np.argmin(np.sum((Xs_01 - barycenter) ** 2, axis=1)), :].copy()
    if dims is None:
        dims = [[i] for i in range(N)]
    for i in range(len(dims)):
        if isinstance(dims[i], six.string_types) or not isinstance(dims[i], collections.Iterable):
            dims[i] = [dims[i]]
        for j in range(len(dims[i])):
            if isinstance(dims[i][j], six.string_types):
                dims[i][j] = names.index(dims[i][j])
    if diagrams is None: # Default plot types for each dimension
        diagrams = []
        for index in dims:
            if len(index) == 1:
                diagrams.append("plot")
            elif len(index) == 2:
                diagrams.append("image")
            else:
                raise NotImplementedError
    assert [mode in ("plot", "image", "surface") for mode in diagrams], "Unsupported diagram mode"
    assert len(dims) == len(diagrams), "One diagram mode per subspace is expected"
    for index, mode in zip(dims, diagrams):
        if mode in ("plot", ):
            assert len(index) == 1, "1D plots require 1D subspaces"
        elif mode in ("image", "surface"):
            assert len(index) == 2, "Images and surface plots require 2D subspaces"
    if gt_range >= 0:  # Precompute these, as they could be large
        assert len(Xs) > 0
        assert len(ys) > 0
        positions_partial = {}
        for i in range(len(dims)):
            columns = np.delete(np.arange(N), dims[i])
            positions_partial[i] = np.asarray(
                Xs)[:, columns]
    state = {'press': False, 'initialized': False}
    labelsize = 11

    def update(event, labelsize=labelsize):
        if event is not None:
            if event.inaxes is None:  # The user clicked outside any axis
                return
            if event.inaxes == axmax:  # The user clicked a button axis
                for n in range(N):
                    focus[n] = estimated_maximum_point[n]
            elif event.inaxes == axmin:
                for n in range(N):
                    focus[n] = estimated_minimum_point[n]
            elif event.inaxes == axrand:
                for n in range(N):
                    focus[n] = np.random.randint(t.n[n])
            elif event.inaxes == axclosest:
                closest = Xs[np.argmin(np.sum((Xs - focus[np.newaxis, :])**2, axis=1)), :]
                for n in range(N):
                    focus[n] = closest[n]
            else:  # The user clicked a plot axis
                # Find which subplot was clicked
                clicked_axis = np.where(np.asarray(ax) == event.inaxes)[0][0]
                if event.dblclick:  # A double click brings us to the subspace with maximal variance
                    tmoments = tr.core.moments(t, modes=dims[clicked_axis], order=2, centered=True, normalized=False,
                                       keepdims=True, eps=1e-2, rmax=5, verbose=False)
                    if event.button == 1:  # Left double click: find maximum
                        val, point = tr.core.maximize(tmoments, rmax=5)
                    elif event.button == 3:  # Left double click: find minimum
                        val, point = tr.core.minimize(tmoments, rmax=5)
                    for n in range(N):
                        focus[n] = point[n]
                # Closest axis sample to the clicked focus
                elif diagrams[clicked_axis] == "plot":
                    new_x = (np.abs(ticks_list[dims[clicked_axis][
                             0]] - event.xdata)).argmin()
                    focus[dims[clicked_axis][0]] = new_x
                elif diagrams[clicked_axis] == "image":
                    new_x = (np.abs(ticks_list[dims[clicked_axis][
                             0]] - event.xdata)).argmin()
                    new_y = (np.abs(ticks_list[dims[clicked_axis][
                             1]] - event.ydata)).argmin()
                    focus[dims[clicked_axis][0]] = new_x
                    focus[dims[clicked_axis][1]] = new_y
                elif diagrams[clicked_axis] == "surface":
                    import mpl_toolkits
                    # pdb.set_trace()
                    xd, yd = event.xdata, event.ydata
                    p = (xd, yd)
                    edges = ax[clicked_axis].tunit_edges()
                    # lines = [proj3d.line2d(p0,p1) for (p0,p1) in edges]
                    from mpl_toolkits.mplot3d import proj3d
                    ldists = [(proj3d.line2d_seg_dist(p0, p1, p), i) for
                              i, (p0, p1) in enumerate(edges)]
                    ldists.sort()
                    # nearest edge
                    edgei = ldists[0][1]

                    p0, p1 = edges[edgei]

                    # scale the z value to match
                    x0, y0, z0 = p0
                    x1, y1, z1 = p1
                    d0 = np.hypot(x0 - xd, y0 - yd)
                    d1 = np.hypot(x1 - xd, y1 - yd)
                    dt = d0 + d1
                    z = d1 / dt * z0 + d0 / dt * z1

                    x, y, _ = proj3d.inv_transform(
                        xd, yd, z, ax[clicked_axis].M)
                    print(event.xdata, event.ydata)
                    ax[clicked_axis].format_coord(event.xdata, event.ydata)
                    new_x = (
                        np.abs(ticks_list[dims[clicked_axis][0]] - x)).argmin()
                    new_y = (
                        np.abs(ticks_list[dims[clicked_axis][1]] - y)).argmin()
                    print("*", event.button, ax[clicked_axis].format_coord(event.xdata, event.ydata))
                    focus[dims[clicked_axis][0]] = new_x
                    focus[dims[clicked_axis][1]] = new_y
                # print(focus)
        for i in range(len(dims)):
            index = dims[i]
            subspace = list(focus)
            for dim in index:
                subspace[dim] = slice(None)
            y = t[subspace].full()
            if diagrams[i] == "plot":
                if not state['initialized']:
                    # Fiber plots
                    lines[i], = ax[i].plot(
                        ticks_list[index[0]], y, linewidth=2)
                    ax[i].set_xlabel(
                        names[index[0]], fontsize=labelsize)
                    ax[i].set_ylabel(output_name, fontsize=labelsize)
                    ax[i].set_xlim([ticks_list[index[0]][
                                   0], ticks_list[index[0]][-1]])
                    a = estimated_minimum
                    b = estimated_maximum
                    ax[i].set_ylim([(a + b) / 2 - (b - a) / 2 * 1.1, (a + b) / 2 + (b - a) / 2 * 1.1])
                    # Vertical lines marking the focus
                    vlines[i] = ax[i].axvline(x=ticks_list[index[0]][
                                              focus[index[0]]], ymin=0, ymax=1, linewidth=5, color='red', alpha=0.5)
                else:
                    lines[i].set_ydata(y)
                    ax[i].draw_artist(ax[i].patch)
                    if gt_range >= 0:
                        gt_markers[i].remove()
                    plot_fillings[i].remove()
                    vlines[i].set_xdata(ticks_list[index[0]][focus[index[0]]])
                if gt_range >= 0:
                    # Detect and show ground-truth points that are not far from this fiber
                    # gt_range = np.ceil(s.shape[index[0]]*gt_factor)
                    point_partial = np.delete(focus, index)
                    dists = np.sqrt(
                        np.sum(np.square(positions_partial[i] - point_partial), axis=1))
                    plot_x = np.asarray(ticks_list[index[0]])[np.asarray(Xs).astype(int)[dists <= gt_range, index[0]]]
                    plot_y = np.asarray(ys)[dists <= gt_range]
                    rgba_colors = np.repeat(np.array([[0, 0, 0.6, 0]]), len(plot_x), axis=0)
                    dist_score = np.exp(-np.square(dists[dists <= gt_range]) / (
                        2 * np.square(gt_range / 6) + np.finfo(np.float32).eps))
                    rgba_colors[:, 3] = dist_score
                    gt_markers[i] = ax[i].scatter(
                        plot_x, plot_y, s=50*dist_score, c=rgba_colors, linewidths=0)
                a = estimated_minimum
                b = estimated_maximum
                plot_fillings[i] = ax[i].fill_between(
                    lines[i].get_xdata(), (a + b) / 2 - (b - a) / 2 * 1.1, y, alpha=0.1, interpolate=True, color='blue')
            elif diagrams[i] == "image":
                if not state['initialized']:
                    images[i] = ax[i].imshow(t[subspace].full().T, cmap=matplotlib.cm.get_cmap('pink'), origin="lower", vmin=estimated_minimum, vmax=estimated_maximum, aspect='auto', extent=[ticks_list[index[0]][0], ticks_list[index[0]][-1], ticks_list[index[1]][0], ticks_list[index[1]][-1]])
                    ax[i].set_xlim([ticks_list[index[0]][
                                   0], ticks_list[index[0]][-1]])
                    ax[i].set_ylim([ticks_list[index[1]][
                                   0], ticks_list[index[1]][-1]])
                    ax[i].set_xlabel(names[index[0]], fontsize=labelsize)
                    ax[i].set_ylabel(names[index[1]], fontsize=labelsize)
                    points[i] = ax[i].plot([ticks_list[index[0]][focus[index[0]]]], ticks_list[index[1]][focus[index[1]]], 'o', color='red')
                else:
                    # Image 2D plot, using an AxisImage
                    images[i].set_data(t[subspace].full().T)
                    # Point marker for the image
                    points[i][0].set_data([ticks_list[index[0]][focus[index[0]]]],
                                          ticks_list[index[1]][focus[index[1]]])
                    if gt_range >= 0:
                        gt_markers[i].remove()
                if gt_range >= 0:
                    # Detect and show ground-truth points that are not far
                    # from this image
                    point_partial = np.delete(focus, index)
                    dists = np.sqrt(
                        np.sum(np.square(positions_partial[i] - point_partial), axis=1))
                    plot_x = np.asarray(ticks_list[index[0]])[np.asarray(
                        Xs).astype(int)[dists <= gt_range, index[0]]]

                    plot_y = np.asarray(ticks_list[index[1]])[np.asarray(
                        Xs).astype(int)[dists <= gt_range, index[1]]]
                    rgba_colors = np.repeat(np.array([[0, 0, 0.6, 0]]), len(plot_x), axis=0)
                    rgba_colors[:, 3] = np.exp(-np.square(dists[dists <= gt_range]) / (
                        2 * np.square(gt_range / 5) + np.finfo(np.float32).eps))
                    gt_markers[i] = ax[i].scatter(
                        plot_x, plot_y, s=50, c=rgba_colors, linewidths=0)
            elif diagrams[i] == "surface":
                if not state['initialized']:
                    x, y = np.meshgrid(ticks_list[index[0]], ticks_list[
                                       index[1]])
                    surfaces[i] = ax[i].plot_surface(x, y, t[subspace].full().T,
                                                     cmap=matplotlib.cm.get_cmap(
                                                         'YlOrBr'),
                                                     vmin=estimated_minimum, vmax=estimated_maximum, cstride=10, rstride=10)
                    ax[i].set_xlabel(
                        names[index[0]], fontsize=labelsize, labelpad=15)
                    ax[i].set_ylabel(
                        names[index[1]], fontsize=labelsize, labelpad=15)
                    ax[i].set_zlabel(output_name)
                    ax[i].xaxis.set_rotate_label(False)
                    ax[i].yaxis.set_rotate_label(False)
                    # ax[i].zaxis.set_rotate_label(False)
                    ax[i].set_zlim(
                        [estimated_minimum, estimated_maximum])
                    points[i] = ax[i].plot([ticks_list[index[0]][focus[index[0]]]], [ticks_list[index[1]][
                                           focus[index[1]]]], [t[focus]], marker='o', color='red',
                                           markeredgecolor='black')
                else:
                    x, y = np.meshgrid(ticks_list[index[0]], ticks_list[index[1]])
                    surfaces[i].remove()
                    surfaces[i] = ax[i].plot_surface(x, y, t[subspace].full().T, cmap=matplotlib.cm.get_cmap('YlOrBr'),
                                                     vmin=estimated_minimum, vmax=estimated_maximum,
                                                     cstride=10, rstride=10)
                    points[i][0].set_data([ticks_list[index[0]][focus[index[0]]]],
                                          [ticks_list[index[1]][focus[index[1]]]])
                    points[i][0].set_3d_properties([t[focus]])
                    if gt_range >= 0:
                        gt_markers[i].remove()
                if gt_range >= 0:
                    # Detect and show ground-truth points that are not far
                    # from this image
                    point_partial = np.delete(focus, index)
                    dists = np.sqrt(
                        np.sum(np.square(positions_partial[i] - point_partial), axis=1))
                    plot_x = np.asarray(ticks_list[index[0]])[
                        np.asarray(Xs).astype(int)[dists <= gt_range, index[0]]]
                    plot_y = np.asarray(ticks_list[index[1]])[
                        np.asarray(Xs).astype(int)[dists <= gt_range, index[1]]]
                    plot_z = np.asarray(ys)[
                        dists <= gt_range]
                    rgba_colors = np.repeat(np.array([[0, 0, 0.6, 0]]), len(plot_x), axis=0)
                    rgba_colors[:, 3] = np.exp(
                        -np.square(dists[dists <= gt_range]) / (2 * np.square(gt_range / 5) + np.finfo(np.float32).eps))
                    gt_markers[i] = ax[i].scatter(
                        plot_x, plot_y, plot_z, s=50, c=rgba_colors, linewidths=0, depthshade=False)
        state['initialized'] = True
        point_values = tr.core.indices_to_coordinates(np.asarray([focus]), ticks_list)[0]
        point_info = "(" + ", ".join(["{:.3f}".format(point_values[i]) for i in range(N)]) + ") -> {:.4f}".format(t[focus])
        plt.suptitle("{}".format(point_info))
        # fig.canvas.update()
        fig.canvas.draw_idle()

    def on_press(event):
        state['press'] = True
        update(event)

    def on_motion(event):
        if state['press']:
            update(event)

    def on_release(event):
        state['press'] = False
        update(event)

    # Prepare figure: one subplot per subspace
    fig = plt.figure()
    ax = []
    import matplotlib.gridspec as gridspec
    if len(dims) <= 4:
        gs = gridspec.GridSpec(1, len(dims))
    elif len(dims) % 2 == 0:
        gs = gridspec.GridSpec(2, len(dims) // 2)
    else:
        toprow = len(dims) // 2 + 1
        bottomrow = len(dims) // 2
        gs = gridspec.GridSpec(2, toprow*2)
        gs = [gs[0, i*2:i*2+2] for i in range(toprow)] + [gs[1, i*2+1:i*2+3] for i in range(bottomrow)]

    # gs.update()
    for i in range(len(dims)):
        if diagrams[i] in ("plot", "image"):
            # ax.append(plt.subplot(1,len(dims),i+1,projection='3d'))
            ax.append(plt.subplot(gs[i]))
        elif diagrams[i] in ("surface"):
            ax.append(plt.subplot(gs[i], projection='3d'))
            # ax[-1].disable_mouse_rotation()
            plt.locator_params(nbins=6)
        else:
            raise ValueError("Supported diagrams: 'plot', 'image', and 'surface' (experimental)")
            # ax.append(plt.subplot(gs[i]))
    # fig.tight_layout()
    lines = {}
    vlines = {}
    images = {}
    points = {}
    surfaces = {}
    plot_fillings = {}
    gt_markers = {}

    # Connect to supported events
    # fig.canvas.mpl_disconnect('button_press_event')
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    # Run interactive loop
    update(None)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.2)

    # Navigation buttons
    axrand = plt.axes([0.39, 0.05, 0.1, 0.05])
    axmax = plt.axes([0.5, 0.05, 0.1, 0.05])
    axmin = plt.axes([0.61, 0.05, 0.1, 0.05])
    brand = Button(axrand, 'Random')
    bmin = Button(axmin, 'Minimum')
    bmax = Button(axmax, 'Maximum')
    if ys is not None:
        axclosest = plt.axes([0.72, 0.05, 0.25, 0.05])
        bclosest = Button(axclosest, 'Closest groundtruth point')
        bclosest.on_clicked(lambda event: update(None))
    else:
        axclosest = None
    bmax.on_clicked(lambda event: update(None))
    bmin.on_clicked(lambda event: update(None))
    brand.on_clicked(lambda event: update(None))
    # plt.subplots_adjust(top=0.9)
    plt.show()


def set_chord_diagram(st, names=None, color='darkblue', n_verts=32, threshold=0.0005, normalization=True, title=None, savefig=False):
    """
    Draw a chord diagram where:
        - Node area is proportional to 1st-order interactions;
        - Edge width is proportional to 2nd-order interactions;

    :param st: a 2^N tensor
    :param names: list of variable names

    """

    plt.style.use('seaborn')

    N = st.d
    if names is None:
        names = ['x_{}'.format(n) for n in range(N)]

    data = np.empty([N, N])
    for i in range(N):
        for j in range(N):
            data[i, j] = tr.core.set_choose(st, modes=[i, j])

    if normalization is True:
        normalization = data.max()
    coords = np.asarray([[np.cos(float(i)/N*2*np.pi), np.sin(float(i)/N*2*np.pi)] for i in range(N)])
    coords_rot = np.asarray([[np.cos(float(i+0.2)/N*2*np.pi)-0.05, np.sin(float(i+0.2)/N*2*np.pi)] for i in range(N)]) # For text

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.axis('off')
    # cm = plt.get_cmap(colormap)

    def cubic_bezier(P, n_verts=n_verts):
        """
        Creates a cubic Bezier curve in 2D between P[0] and P[3], with control points P[1] and P[2]

        :param P: a 2D array of size 4 x 2
        :param n_verts: how many points to generate
        :returns: a 2D array of size n_verts x 2

        """

        ts = np.linspace(0, 1, n_verts)[:, np.newaxis]
        return (1 - ts)**3*P[0:1, :] + 3*(1-ts)**2*ts*P[1:2, :] + 3*(1-ts)*ts**2*P[2:3, :] + ts**3*P[3:4, :]

    for data_i in range(N):
        for data_j in range(data_i+1, N):
            if np.abs(data[data_i, data_j] / normalization) >= threshold:
                x1 = coords[data_i]
                x2 = coords[data_j]
                verts = cubic_bezier(np.array([x1, x1*3/4, x2*3/4, x2]))
                for i in range(len(verts)-1):
                    ax.plot(verts[i:i+2, 0], verts[i:i+2, 1], lw=20*data[data_i, data_j]/normalization, zorder=1,
                            c=color, alpha=1)

    ax.scatter(coords[:, 0], coords[:, 1],
               20*matplotlib.rcParams['lines.markersize']*np.diag(data)/normalization, c=color, zorder=2)
    for coord, name in zip(coords_rot, names):
        txt = ax.text(coord[0], coord[1], name)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    if title is not None:
        plt.title(title)
    if savefig:
        plt.savefig(savefig)
    else:
        plt.show()
