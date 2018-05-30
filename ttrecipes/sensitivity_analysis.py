# -*- coding: utf-8 -*-
"""
Higher-level functions to gather many sensitivity analysis metrics
from N-dimensional black-box functions approximated as TT tensors.

Todo:
    * Add derivative-based and histogram-based metrics

"""

# -----------------------------------------------------------------------------
# Authors:      Rafael Ballester-Ripoll <rballester@ifi.uzh.ch>
#               Enrique G. Paredes <egparedes@ifi.uzh.ch>
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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import functools
import itertools
import pprint
import random
import time

import numpy as np
import scipy as sp
import scipy.stats
from tabulate import tabulate
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

import tt
import ttrecipes as tr


def var_metrics(fun=None, axes=None, t=None, default_bins=100, effective_threshold=0.95,
                dist_fraction=0.00005, fun_mode="array",
                eps=1e-5, verbose=False, cross_kwargs=None, random_seed=None,
                max_order=2, show=True):
    """Variance-based sensitivity analysis.

    User-friendly wrapper that conducts a general variance-based sensitivity
    analysis on an N-dimensional black-box function according to the given
    marginal distributions.

    Args:
        fun (callable): Callable object representing the N-dimensional function.
        axes (iterable): A list of axis definitions. Each axis is a `dict` with
            the name of the axis variable, its domain and its distribution function.
            If the domain and the distribution function are both undefined,
            a uniform(0, 1) distribution is assumed for the axis.

            axis: dict(name, domain, dist, display_name=None)
                name (str): Variable name. Defaults to x_{i} (i = axis index).

                domain (tuple or np.ndarray): Domain definition.
                    Defaults to tuple(None, None) when dist is defined.

                    tuple -> bounds (lower or None, upper or None). None values are
                        truncated to (dist.ppf(dist_fraction), dist.ppf(1.0 - dist_fraction))
                    np.ndarray -> samples (actual positions in the axis)

                dist (scipy.stats.rv_continuous or np.ndarray): Distribution definition.
                    Defaults to a uniform distribution in the defined domain.

                    scipy.stats.rv_continuous-> frozen scipy.stats distribution
                    np.ndarray -> values of the marginal PDF at the sampled domain positions.
                        In this case, domain must contain a np.ndarray of the same size.

                display_name (str, optional): Variable name using LaTeX fomat. Defaults to None
        t: a tensor to compute sensitivity indices of. Either `t` or `fun` must be given (but not both). Unlike `fun`,
        `t` does not necessarily require `axes`

        default_bins (int, optional): Number of bins for axes without explicit
            definition. Defaults to 100.
        effective_threshold (float, optional): Threshold (1 - tolerance) for computation of
            effective dimensions. Defaults to 0.95
        dist_fraction (float, optional): Fraction of the probability distribution
            discarded at each side of the sampling space when using 'spatial'
            sampling. Defaults to 0.00005
        fun_mode (str, optional): ['array' or 'parameters']. If 'array' (default),
            the function takes a matrix as input, with one row per evaluation,
            and returns a vector. If 'parameters', the function takes one instance
            per call (each variable is an argument), and returns a scalar.
        eps (float, optional): Tolerated relative approximation error in TT
            computations. Defaults to 1e-5.
        verbose (bool, optional): Verbose messages. Defaults to False.
        cross_kwargs (:obj:`dict`, optional): Parameters for the cross
            approximations. Defaults to None.
        random_seed: value to initialize random generators to get reproducible results.
            Defaults to None.
        max_order (int, optional): Maximum order of the collected Sobol indices.
            Defaults to 2.
        show (bool, optional): Print results. Defaults to True.

    Returns:
        tuple(dict, dict): A 'metrics' dictionary containing a structured and
            easily accessible collection of sensitivity metrics, and additional
            metainfomation with the actual Sobol Tensor Trains (for internal use
            of TT recipes routines).

            {
                'variables': ``list`` with the names of the model variables
                'sobol_indices': ``dict`` with Sobol indices up to ``max_order``
                'closed_sobol_indices': ``dict`` with Closed Sobol indices up to ``max_order``
                'total_sobol_indices': ``dict`` with Total Sobol indices up to ``max_order``
                'superset_sobol_indices': ``dict`` with Superset Sobol indices up to ``max_order``
                'dimension_distribution': ``numpy.array`` with shape [N]
                'mean_dimension': ``float`` in range [1, N]
                'effective_superposition': ``int`` in range [1, N]
                'effective_truncation': ``int`` in range [1, N]
                'effective_successive': ``int`` in range [1, N]
                'shapley_values': ``numpy.array`` with shape [N]
                'banzhaf_coleman_values': ``numpy.array`` with shape [N]
                '_tr_info': ``dict`` for internal use
            }

    Examples:
        Call the function with a user-defined function::

            from ttrecipes import sensitivity_analysis

            axes = [dict(name='Q', domain=(500, 3000), dist=sp.stats.gumbel_r(1013.0, 558.0)),
                    dict(name='K_s', domain=(15.0, None), dist=sp.stats.norm(loc=30, scale=8)),
                    dict(name='Z_m', dist=sp.stats.triang(loc=54.0, scale=2, c=0.5)),
                    dict(name='H_d', dist=sp.stats.uniform(loc=7, scale=2)),
                    dict(name='B', dist=sp.stats.triang(loc=295.0, scale=10, c=0.5)),
                    dict(name='delta', dist=sp.stats.lognorm(scale=np.exp(2.19), s=0.517))]

            metrics = sensitivity_analysis.var_metrics(
                my_function, axes, eps=1e-3, verbose=True)
            tabulate_metrics(metrics)

    References:
        * R. Ballester-Ripoll, E. G. Paredes, R. Pajarola. (2017)
            "Sobol Tensor Trains for Global Sensitivity Analysis"
            arXiv: https://arxiv.org/abs/1712.00233
        * R. Ballester-Ripoll, E. G. Paredes, R. Pajarola (2017)
            "Tensor Approximation of Advanced Metrics for Sensitivity Analysis"
            arXiv: https://arxiv.org/abs/1712.01633
        * Caflisch, R.E. and Morokoff, W.J. and Owen, A.B. (1997) "Valuation of Mortgage
            Backed Securities Using Brownian Bridges to Reduce Effective Dimension"
        * Pierre L'Ecuyer and Christiane Lemieux. (2000) "Variance Reduction via Lattice Rules"
        * Owen, Art B. (2003) "The dimension distribution and quadrature test functions"
        * Owen, Art B. (2014) "Sobol' Indices and Shapley Value"

    """


    if fun is None:
        assert t is not None
        axes = [dict(name='x_{}'.format(n), domain=np.arange(t.n[n])) for n in range(t.d)]
    if t is None:
        assert fun is not None
        assert axes is not None
    N = len(axes)
    names, ticks_list, marginals = tr.models.parse_axes(
        axes, default_bins=default_bins, dist_fraction=dist_fraction)

    axes = list(zip(names, ticks_list, marginals))

    if cross_kwargs is None:
        cross_kwargs = dict()

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    if verbose:
        print("\n-> Building surrogate model")
    pdf = tt.vector.from_list([marg[np.newaxis, :, np.newaxis] for marg in marginals])

    def fun_premultiplied(Xs):
        return fun(Xs) * tr.core.sparse_reco(pdf, tr.core.coordinates_to_indices(Xs, ticks_list=ticks_list))

    if t is None:
        model_time = time.time()
        tt_pdf, n_samples = tr.core.cross(ticks_list, fun_premultiplied, mode=fun_mode,
                                          return_n_samples=True, eps=eps, verbose=verbose,
                                          **cross_kwargs)
        model_time = time.time() - model_time
    else:
        tt_pdf = t*pdf
        n_samples = None
        model_time = None

    if verbose:
        print("\n-> Computing sensitivity metrics")
    sa_time = time.time()
    st = tr.core.sobol_tt(tt_pdf, pdf=pdf, premultiplied=True, eps=eps,
                          verbose=verbose, **cross_kwargs)

    tst = tr.core.to_upper(st)
    cst = tr.core.to_lower(st)
    sst = tr.core.to_superset(st)
    dim, var = tr.core.effective_dimension(st, effective_threshold, 'truncation')
    indices, _ = tr.core.largest_k_tuple(cst, dim)
    shapley_values = tr.core.semivalues(cst, ps='shapley')
    banzhaf_coleman_values = tr.core.semivalues(cst, ps='banzhaf-coleman')

    metrics = dict()
    metrics['variables'] = names
    metrics['dimension_distribution'] = tr.core.dimension_distribution(st)
    metrics['mean_dimension'] = metrics['dimension_distribution'].dot(np.arange(1, N + 1))
    metrics['effective_superposition'] = tr.core.effective_dimension(
        st, effective_threshold, 'superposition')
    metrics['effective_truncation'] = (dim, var, indices)
    metrics['effective_successive'] = tr.core.effective_dimension(
        st, effective_threshold, 'successive')
    metrics['effective_threshold'] = effective_threshold
    sa_time = time.time() - sa_time

    if verbose:
        print("\n-> Exporting results")

    metrics['_tr_info'] = dict(
        tags=('stt'),
        axes=axes,
        default_bins=default_bins,
        eps=eps,
        verbose=verbose,
        cross_kwargs=cross_kwargs,
        ticks_list=ticks_list,
        n_samples=n_samples,
        model_time=model_time,
        sa_time=sa_time,
        tt_pdf=tt_pdf,
        st=st,
        tst=tst,
        cst=cst,
        sst=sst
    )

    metrics['shapley_values'] = dict()
    metrics['banzhaf_coleman_values'] = dict()
    for i, name in enumerate(names):
        metrics['shapley_values'][name] = shapley_values[i]
        metrics['banzhaf_coleman_values'][name] = banzhaf_coleman_values[i]
    metrics = collect_sobol(metrics, max_order)

    if show:
        tabulate_metrics(metrics, max_order)

    return metrics


def collect_sobol(metrics, max_order):
    """Collect Sobol sensitivity indices up to a maximum order from a Sobol Tensor Train.

    Args:
        metrics (dict): Sobol Tensor Train metainformation
            (typically obtained from: metrics = var_metrics(...))
        max_order (int): Maximum order of the collected Sobol indices.

    Returns:
        dict: A dictionary with a structured and easily accessible collection
            of Sobol indices.

    """
    if (not isinstance(metrics, dict) or '_tr_info' not in metrics or
            'stt' not in metrics['_tr_info']['tags']):
        raise ValueError("Invalid 'metrics' object")
    if max_order < 1:
        raise ValueError("'max_order' must be larger than zero")

    names = [axis[0] for axis in metrics['_tr_info']['axes']]
    N = len(names)
    st = metrics['_tr_info']['st']
    tst = metrics['_tr_info']['tst']
    cst = metrics['_tr_info']['cst']
    sst = metrics['_tr_info']['sst']

    collected = dict(sobol_indices=dict(), total_sobol_indices=dict(),
                     closed_sobol_indices=dict(), superset_sobol_indices=dict())
    for order in range(1, max_order + 1):
        collected['sobol_indices'][order] = dict()
        collected['total_sobol_indices'][order] = dict()
        collected['closed_sobol_indices'][order] = dict()
        collected['superset_sobol_indices'][order] = dict()

        for idx in itertools.combinations(list(range(N)), order):
            key = tuple(names[i] for i in idx)
            collected['sobol_indices'][order][key] = tr.core.set_choose(st, idx)
            collected['total_sobol_indices'][order][key] = tr.core.set_choose(tst, idx)
            collected['closed_sobol_indices'][order][key] = tr.core.set_choose(cst, idx)
            collected['superset_sobol_indices'][order][key] = tr.core.set_choose(sst, idx)

    metrics.update(collected)

    return metrics


def tabulate_metrics(metrics, max_order=None,
                     selection=('indices', 'interactions', 'dimensions', 'dim_dist'),
                     tablefmt="presto", floatfmt=".6f",
                     numalign="decimal", stralign="left",
                     output_mode='console', show_titles=True):
    """Create tables with the contents of the collected sensitivity metrics.

    The tables are generated using 'tabulate'. It is possible to customize the
    output by passing optional arguments with specific tabulate format parameters.

    Args:
        metrics (dict): Collected sensitivity metrics
            (typically obtained from: metrics = var_metrics(...))
        max_order (int, optional): Maximum order of the collected Sobol indices.
            If None (default), use the current value in 'metrics'.
        selection (iterable, optional): Tables to be generated.
            The options are: 'indices', 'interactions', 'dimensions', 'dim_dist'.
            Defaults to ('indices', 'interactions', 'dimensions', 'dim_dist').
        tablefmt (str, optional): tabulate format parameter.
            Defaults to 'presto'.
        floatfmt (str, optional): tabulate format parameter.
            Defaults to '.6f'.
        numalign (str, optional): tabulate format parameter.
            Defaults to 'decimal'.
        stralign (str, optional): tabulate format parameter.
            Defaults to 'left'.
        output_mode (str, optional): Output mode: 'console', 'string' or 'dict'.
            Defaults to 'console'.
        show_titles (bool, optional): Add single line titles to the tables.
            Defaults to True.

    """
    for i in selection:
        if i not in ('indices', 'interactions', 'dimensions', 'dim_dist'):
            raise ValueError("selection items must be 'indices', 'interactions', "
                             "'dimensions' or 'dim_dist'")

    if max_order is None:
        max_order = max(metrics['sobol_indices'].keys())
    elif max(metrics['sobol_indices'].keys()) < max_order:
        collect_sobol(metrics, max_order)

    names = [axis[0] for axis in metrics['_tr_info']['axes']]
    outputs = dict()
    tab_fn = functools.partial(tabulate, tablefmt=tablefmt, floatfmt=floatfmt,
                               numalign=numalign, stralign=stralign)

    if 'indices' in selection:
        out = []
        table = [(n, metrics['sobol_indices'][1][(n,)],
                  metrics['total_sobol_indices'][1][(n,)],
                  metrics['shapley_values'][n],
                  metrics['banzhaf_coleman_values'][n])
                 for n in names]

        if show_titles:
            out.append("\n\t\t ** Sensitivity Indices **".upper())
        out.append(tab_fn(table, headers=["Variable", "Sobol", "Total",
                                          "Shapley", "Banzhaf-Coleman"]))
        out.append('\n')
        outputs['indices'] = '\n'.join(out)

    if 'interactions' in selection:
        for order in range(2, max_order + 1):
            out = []
            table = []
            for key in metrics['sobol_indices'][order].keys():
                table.append((*key, metrics['sobol_indices'][order][key],
                              metrics['total_sobol_indices'][order][key],
                              metrics['closed_sobol_indices'][order][key],
                              metrics['superset_sobol_indices'][order][key]))

            if show_titles:
                out.append("\t\t ** Sensitivity Indices (order {}) **".format(order).upper())
            out.append(tab_fn(table, headers=[*["Variable {}".format(i+1) for i in range(order)],
                                                "Sobol", "Total", "Closed", "Superset"]))
            out.append('\n')
            outputs['interactions_{}'.format(order)] = '\n'.join(out)

    if 'dimensions' in selection:
        out = []
        table = [('Mean dimension', metrics['mean_dimension']),
                 ('Effective (superposition)', *metrics['effective_superposition']),
                 ('Effective (successive)', *metrics['effective_successive']),
                 ('Effective (truncation)', metrics['effective_truncation'][0],
                  metrics['effective_truncation'][1], ", ".join(
                      [names[i] for i in metrics['effective_truncation'][2]]))]

        if show_titles:
            out.append(("\t\t ** Dimension metrics **".upper()))
        out.append(tab_fn(table, headers=["Dimension Metric", "Value",
                                          "Rel. Variance", "Variables"]))
        out.append('\n')
        outputs['dimensions'] = '\n'.join(out)

    if 'dim_dist' in selection:
        out = []
        cumul_dist = np.cumsum(metrics['dimension_distribution'])
        table = [[i + 1, value, cumul_dist[i]]
                 for i, value in enumerate(metrics['dimension_distribution'])]

        if show_titles:
            out.append("\t ** Dimension distribution **".upper())
        out.append(tab_fn(table, headers=["Order", "Rel. Variance",
                                          "Cumul. Variance"]))
        out.append("\n")
        outputs['dim_dist'] = '\n'.join(out)

    if output_mode == 'string' or output_mode == 'console':
        outputs = ''.join(outputs.values())
        if output_mode == 'console':
            print(outputs)
            outputs = None

    return outputs


def query_sobol(st, include=(), exclude=(), min_order=1, max_order=None,
                mode='highest', index_type='standard',
                eps=1e-6, verbose=False, **kwargs):
    """Interface to query sensitivity metrics: find a tuple of variables satisfying certain criteria.

    Args:
        st (object): A Sobol TT or a 'metrics' dictionary,
            typically obtained from: metrics = var_metrics(...))
        include (list): List of variables that must appear in the result.
            When `st` is a ttpy vector, it has to be a list of integers with the
            indices of the variables. When `st` is a 'metrics' dictionary, it may
            also be a list of strings with the names of the variables.
        exclude (list): List of variables that must NOT appear in the result.
            When `st` is a ttpy vector, it has to be a list of integers with the
            indices of the variables. When `st` is a 'metrics' dictionary, it may
            also be a list of strings with the names of the variables.
        min_order (int, optional): consider only tuples of this order or above.
            Defaults to 1
        max_order (int, optional): consider only tuples up to this order.
            If None (default), no bound
        mode (str, optional): must be 'highest' or 'lowest'.
            Defaults to 'highest'.
        index_type (str, optional): which Sobol indices or related indices to consider.
            Must be 'standard', 'closed', 'total' or 'superset'. Defaults to 'standard'.
        eps (float, optional): Tolerated relative error. Defaults to 1e-6.
        verbose (bool, optional: Activate verbose mode. Defaults to False.

    Returns:
        (tuple, value): the best variables, and their index

    """
    if isinstance(st, tt.core.vector.vector):
        st = st
        names = None
    elif isinstance(st, dict) and '_tr_info' in st and 'stt' in st['_tr_info']['tags']:
        names = [axis[0] for axis in st['_tr_info']['axes']]
        names_idx = dict([(name, i) for i, name in enumerate(names)])
        include = [i if isinstance(i, int) else names_idx[i] for i in include]
        exclude = [i if isinstance(i, int) else names_idx[i] for i in exclude]
        st = st['_tr_info']['st']
    else:
        raise ValueError("'STT' must be a ttpy vector or a 'metrics' object")

    N = st.d
    if max_order is None:
        max_order = N
    if index_type == 'closed':
        st = tr.core.to_lower(st)
    elif index_type == 'total':
        st = tr.core.to_upper(st)
    elif index_type == 'superset':
        st = tr.core.to_superset(st)
    elif index_type != 'standard':
        raise ValueError("index_type must be 'standard', 'closed', 'total' or 'superset'")

    # First mask: tuples that must be included
    mask1 = [np.array([1, 1])[np.newaxis, :, np.newaxis] for n in range(N)]
    for n in include:
        mask1[n] = np.array([0, 1])[np.newaxis, :, np.newaxis]
    mask1 = tt.vector.from_list(mask1)

    # Second mask: tuples that must be excluded
    mask2 = [np.array([1, 1])[np.newaxis, :, np.newaxis] for n in range(N)]
    for n in exclude:
        mask2[n] = np.array([1, 0])[np.newaxis, :, np.newaxis]
    mask2 = tt.vector.from_list(mask2)

    mask12 = mask1*mask2
    mask12 = mask12 - tr.core.hamming_weight(N)*eps  # Tiny gradient to follow

    # Last mask: order bounds
    hws = tr.core.hamming_weight_state(N)
    cores = tt.vector.to_list(hws)
    cores[-1][:, :min_order, :] = 0
    cores[-1][:, max_order+1:, :] = 0
    cores[-1] = np.sum(cores[-1], axis=1, keepdims=True)
    mask3 = tr.core.squeeze(tt.vector.from_list(cores))
    mask3 = tt.vector.round(mask3, eps=0)

    if mode == 'highest':
        st = tt.multifuncrs2([st, mask3], lambda x: x[:, 0] * x[:, 1],
                             eps=eps, verb=verbose, **kwargs) * mask12
        val, point = tr.core.maximize(st)
    elif mode == 'lowest':  # Shift down by one so that the masks' zeroing-out works as intended
        st = tt.multifuncrs2([st - tr.core.constant_tt(st.n), mask3], lambda x: x[:, 0] * x[:, 1],
                             eps=eps, verb=verbose, **kwargs) * mask12
        val, point = tr.core.minimize(st)
        val += 1  # Shift the value back up
    else:
        raise ValueError("Mode must be either 'highest' or 'lowest'")

    result = list(np.where(point)[0]), val
    if names is not None:
        result = ([names[i] for i in result[0]], val)
    return result


def plot_indices(metrics, indices=('sobol', 'shapley', 'total'),
                 title='Sensitivity Indices', labels=('Variables', 'Variance'),
                 fontsize=10.0, show=True):
    """Plot selected sensitivity indices (requires Matplotlib).

    Args:
        metrics (dict): Collected sensitivity metrics
            (typically obtained from: metrics = var_metrics(...))
        indices (iterable, optional): Ordered selection of sensitivity indices.
            The options are: 'sobol', 'shapley', 'total'.
            Defaults to ('sobol', 'shapley', 'total').
        title (str, optional): Plot title. Defaults to 'Sensitivity Indices'.
        labels ((str, str), optional): Labels for X and Y axes.
            Defaults to ('Variables', 'Variance').
        fontsize (float, optional): Font size for text labels. Defaults to 10.0
        show (bool, optional): Force inmediate plot of the results.
            Defaults to True.

    Returns:
        ax, fig: Tuple of 'axis' and 'figure' Matplotlib objects.

    """
    if matplotlib is None:
        print("Matplotlib not found: plotting functions are disabled")
        return None, None

    names = metrics['variables']
    n_vars = len(names)
    colors = dict(sobol='#9FC1D3',
                  shapley='#276090',
                  total='#A5CD82')
    values = dict(sobol=[metrics['sobol_indices'][1][name,] for name in names],
                  shapley=[metrics['shapley_values'][name] for name in names],
                  total=[metrics['total_sobol_indices'][1][name,] for name in names])
    bar_spacer = 0.2
    bar_width = (1 - bar_spacer) / len(indices)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylim(0, 1.09)
    ax.set_xlabel(labels[0], fontsize=fontsize)
    ax.set_ylabel(labels[1])
    ax.set_xticks(0.5 + np.arange(n_vars))
    ax.set_xticklabels(names)

    for i, index in enumerate(indices):
        if index not in ('sobol', 'shapley', 'total'):
            raise ValueError("mark must be 'sobol', 'shapley' or 'total'")
        bars = ax.bar(bar_spacer + i * bar_width + np.arange(n_vars), values[index],
                      bar_width, color=colors[index])
        bars.set_label(index)
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper right", ncol=1)

    if show:
        plt.show()

    return fig, ax


def plot_dim_distribution(metrics, marks=('cumulative', 'mean', 'superposition'),
                          title='Dimension distribution', labels=('', 'Variance'),
                          max_order=None, annotate=True, float_fmt='{:.2f}',
                          fontsize=10.0, show=True):
    """Plot selected sensitivity indices (requires Matplotlib).

    Args:
        metrics (dict): Collected sensitivity metrics
            (typically obtained from: metrics = var_metrics(...))
        marks (iterable, optional): Selection of dimension metrics.
            The options are: 'cumulative', 'mean', 'superposition'.
            Defaults to ('cumulative', 'mean', 'superposition').
        title (str, optional): Plot title. Defaults to 'Dimension distribution'.
        labels ((str, str), optional): Labels for X and Y axes.
            Defaults to ('', 'Variance').
        max_order (int, optional): Maximum order shown in the plot.
            Defaults to None
        annotate (bool, optional): Add labels with the exact values.
            Defaults to True.
        float_fmt (str, optional): format specification string for float values.
            Defaults to '{:.2f}'.
        fontsize (float, optional): Font size for text labels. Defaults to 10.0
        show (bool, optional): Force inmediate plot of the results.
            Defaults to True.

    Returns:
        ax, fig: Tuple of 'axis' and 'figure' Matplotlib objects.

    """

    if matplotlib is None:
        print("Matplotlib not found: plotting functions are disabled")
        return None, None

    dims_dist = metrics['dimension_distribution'][:max_order]
    cumul_dist = np.cumsum(dims_dist)
    d_length = len(dims_dist)
    orders = np.arange(1, d_length + 1)

    colors = dict(dist='#5D4470', cumul='#1B61A5',
                  mean='#469B55', superposition='#6C6C6C')
    width = 0.8
    x_offset = -fontsize * d_length / 400.0
    y_offset = 0.0025 * fontsize

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylim(0, 1.09)
    ax.set_xlim(0.5, d_length + 0.5)
    ax.set_xlabel(labels[0], fontsize=fontsize)
    ax.set_ylabel(labels[1])

    ax.bar(orders, dims_dist, width, color=colors['dist'])
    if annotate:
        for i, value in enumerate(dims_dist):
            if value > 2.5 * y_offset:
                y = value - 2 * y_offset
                color = 'w'
            else:
                y = value + y_offset
                color = 'k'
            ax.text(x_offset + i + 1, y, float_fmt.format(value),
                    color=color, fontsize=fontsize)

    if 'cumulative' in marks:
        ax.plot(orders, cumul_dist, colors['cumul'])
        ax.scatter(orders, cumul_dist, c=colors['cumul'], marker='s', zorder=3)
        if annotate:
            for i, value in enumerate(cumul_dist):
                if i > 0:
                    ax.text(x_offset + i + 1, y_offset + cumul_dist[i],
                            float_fmt.format(value), color=colors['cumul'],
                            fontsize=fontsize)

    if 'mean' in marks:
        mean_d = metrics['mean_dimension']
        ax.axvline(mean_d, c=colors['mean'], linestyle='-', zorder=1)
        if annotate:
            ax.text(mean_d, 0.5,
                    (' $D_S=' + float_fmt + '$').format(mean_d).format(mean_d),
                    color=colors['mean'], fontsize=fontsize)

    if 'superposition' in marks:
        super_d, super_d_var = metrics['effective_superposition']
        threshold = metrics['effective_threshold']
        ax.axvline(super_d, c=colors['superposition'], linestyle='--', zorder=0)
        ax.axhline(threshold, c=colors['superposition'], linestyle=':', zorder=0)
        if annotate:
            ax.text(super_d, 0.25, ' $d_S={}$'.format(super_d),
                    color=colors['superposition'], fontsize=fontsize)
            ax.text(d_length + 0.6 * width, threshold,
                    ('1-$\epsilon=' + float_fmt + '$').format(threshold),
                    color=colors['superposition'], fontsize=0.8 * fontsize)

    if show:
        plt.show()

    return fig, ax
