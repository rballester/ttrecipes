# -*- coding: utf-8 -*-
"""Sensitivity analysis functions.

This module contains user-friendly functions computing different sensitivity
analysis metrics on N-dimensional black-box functions approximated with
Tensor Trains.

Todo:
    * Add derivative-based and histogram-based metrics

"""

# -----------------------------------------------------------------------------
# Authors:      Rafael Ballester-Ripoll <rballester@ifi.uzh.ch>
#               Enrique G. Paredes <egparedes@ifi.uzh.ch>
#
# Copyright:    TT Recipes project (c) 2016-2017
#               VMMLab - University of Zurich
# -----------------------------------------------------------------------------

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import collections
import functools
import itertools
import pprint
import time

import numpy as np
import scipy as sp
import scipy.stats
from tabulate import tabulate

import tt
import ttrecipes as ttr


def compute_var_metrics(fun, axes, default_bins=100, effective_threshold=0.95,
                        sampling_mode='spatial', dist_fraction=0.00005,
                        eps=1e-5, verbose=False, cross_kwargs=None,
                        max_order=2, show=False):
    """Variance-based sensitivity analysis.

    User-friendly wrapper that conducts a general variance-based sensitivity
    analysis on an N-dimensional black-box function according to the given
    marginal distributions.

    Args:
        fun (callable): Callable object representing the N-dimensional function.
        axes (iterable): A list of axis definitions (bounds and marginal PDF).

            axis: (name, bins, hard_bounds, marginal_dist)
            name: ``str``
            bins: number or None
                if bins is None:
                    bins = default_bins
                else:
                    bins = int(number) if number > 1 else int(number * default_bins)
            hard_bounds: None or (lower or None, upper or None)
            marginal_dist: (scipy.rv_continuous, factor=1) or (dist_name, param1, param2, factor=1)
                rv_continuous: *frozen* 'scipy.stats.rv_continuous' like class
                dist_name, param1, param2:
                    - 'unif': ``param1`` and ``param2`` are ignored
                    - 'isotriang': ``param1`` and ``param2`` are ignored
                    - 'triang': ``param1`` = peak (``lower`` <= peak <= ``upper``)
                    - 'norm': ``param1`` = mu and ``param2`` = sigma
                    - 'lognorm': ``param1`` = mu and ``param2`` = sigma
                    - 'gumbel': ``param1`` = mu and ``param2`` = beta
                    - 'weibull': ``param1`` = alpha (shape) and ``param2`` = beta or lambda (scale)
                factor: final_pdf => factor * pdf() (factor=1 by default)

        default_bins (int, optional): Number of bins for axes without explicit
            definition. Defaults to 100.
        effective_threshold (float, optional): Threshold for computation of
            effective dimensions. Defaults to 0.95.
        sampling_mode (str, optional): ['dist' or 'spatial'] Create a regular
            distribution of the samples in the distribution or the spatial
            domain. Defaults to 'spatial'.
        dist_fraction (float, optional): Fraction of the probability distribution
            discarded at each side of the sampling space when using 'spatial'
            sampling. Defaults to 0.00005
        eps (float, optional): Tolerated relative approximation error in TT
            computations. Defaults to 1e-5.
        verbose (bool, optional): Verbose messages. Defaults to False.
        cross_kwargs (:obj:`dict`, optional): Parameters for the cross
            approximations. Defaults to None.
        max_order (int, optional): Maximum order of the collected Sobol indices.
            Defaults to 2.
        show (bool, optional): Print results. Defaults to False.

    Returns:
        tuple(dict, dict): A tuple of dictionaries 'metrics', 'stt_info'.
            The first one contains a structured and easily accessible collection
            of sensitivity metrics. The second one contains the actual Sobol
            Tensor Trains encoding all the posible metrics and additional
            metainformation for internal use of TT recipes routines.

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
            }

    Examples:
        Call the function using long declarative format::

            from ttrecipes import sensitivity_analysis

            axes = (('M', None, (30, 60), ('unif', None, None)),
                    ('S', 0.5, (0.005, 0.020), ('unif', None, None)),
                    ('rho_p', None, None, ('lognorm', -0.592, 0.219)),
                    ('m_l', 150, (0.0, None), ('norm', 1.18, 0.377)),
                    ('k', None, (1000, 5000), ('unif', None, None)))

            metrics, stt_info = sensitivity_analysis.compute_var_metrics(
                my_function, axes, eps=1e-3, verbose=True)
            print_metrics(metrics)

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

    def get_rv(dist_name, param1, param2, bounds):
        if dist_name == 'unif':
            assert bounds[0] <= bounds[1]
            dist = sp.stats.uniform(loc=bounds[0], scale=bounds[1] - bounds[0])
        elif dist_name == 'norm':
            dist = sp.stats.norm(loc=param1, scale=param2)
        elif dist_name == 'lognorm':
            dist = sp.stats.lognorm(scale=np.exp(param1), s=param2)
        elif dist_name == 'triang':
            assert bounds[0] <= param1 <= bounds[1]
            scale = bounds[1] - bounds[0]
            c = (param1 - bounds[0]) / scale
            dist = sp.stats.triang(scale=scale, c=c)
        elif dist_name == 'isotriang':
            dist = sp.stats.triang(loc=bounds[0], scale=bounds[1] - bounds[0],
                                   c=0.5)
        elif dist_name == 'gumbel':
            dist = sp.stats.gumbel_r(param1, param2)
        elif dist_name == 'weibull':
            assert param1 > 0 and param2 > 0
            dist = sp.stats.weibull_min(c=param1, scale=param2)
        else:
            raise ValueError(
                'Invalid distribution identification {}'.format(dist_name))
        return dist

    def sample_axis(n_ticks, rv, hard_bounds, factor=1.0, mode='pdf'):
        assert mode in ('dist', 'spatial')

        if mode == 'spatial':
            if hard_bounds[0] is None:
                hard_bounds[0] = rv.ppf(dist_fraction)
            if hard_bounds[1] is None:
                hard_bounds[1] = rv.ppf(1.0 - dist_fraction)
            half_bin = (hard_bounds[1] - hard_bounds[0]) / (2 * n_ticks)
            hard_bounds[0] += half_bin
            hard_bounds[1] -= half_bin

            ticks = np.linspace(hard_bounds[0], hard_bounds[1], n_ticks)
            marginal = rv.pdf(ticks)
            marginal /= np.sum(marginal)
        else:
            ppf_bounds = np.asarray([0.0, 1.0])
            if hard_bounds[0] is not None:
                ppf_bounds[0] = rv.cdf(hard_bounds[0])
            if hard_bounds[1] is not None:
                ppf_bounds[1] = rv.cdf(hard_bounds[1])
            half_bin = (ppf_bounds[1] - ppf_bounds[0]) / (2 * n_ticks)
            ppf_bounds[0] += half_bin
            ppf_bounds[1] -= half_bin

            ticks = rv.ppf(np.linspace(ppf_bounds[0], ppf_bounds[1], n_ticks))
            marginal = np.ones(ticks.shape) / n_ticks

        ticks *= factor
        assert np.isfinite(ticks).all() and np.isfinite(marginal).all()
        return ticks, marginal

    def sample_counter(fun, mode):
        assert mode in ("array", "parameters")
        if mode == "array":
            def wrapped(*args, **kwargs):
                wrapped.n_samples += args[0].shape[0]
                return fun(*args, **kwargs)
        else:
            def wrapped(*args, **kwargs):
                wrapped.n_samples += 1
                return fun(*args, **kwargs)

        wrapped.n_samples = 0
        return wrapped

    N = len(axes)
    names = [''] * N
    n_ticks = [None] * N
    rvs = [None] * N
    factors = [None] * N
    ticks_list = [None] * N
    marginals = [None] * N

    for i, (name, bins, hard_bounds, dist) in enumerate(axes):
        assert hard_bounds is None or len(hard_bounds) <= 2
        assert 1 <= len(dist) <= 4

        names[i] = name
        if bins is None:
            n_ticks[i] = default_bins
        else:
            n_ticks[i] = int(bins) if bins > 1 else int(bins * default_bins)

        if not isinstance(dist, collections.Iterable):
            dist = tuple(dist)
        if len(dist) < 3:
            rvs[i] = dist[0]
            factors[i] = 1.0 if len(dist) == 1 else dist[1]
        else:
            rvs[i] = get_rv(dist[0], dist[1], dist[2], hard_bounds)
            factors[i] = 1.0 if len(dist) < 4 else dist[3]

        if hard_bounds is None:
            hard_bounds = (None, None)

        ticks_list[i], marginals[i] = sample_axis(
            n_ticks[i], rvs[i], np.array(hard_bounds), factors[i], mode=sampling_mode)

    # pprint.pprint(ticks_list)
    # pprint.pprint(marginals)

    if callable(fun):
        f = fun
        mode = "array"
    else:
        f = fun[0]
        mode = fun[1]
    f = sample_counter(f, mode)

    if cross_kwargs is None:
        cross_kwargs = dict()

    if verbose:
        print("\n-> Building surrogate model")
    model_time = time.time()
    pdf = tt.vector.from_list([marg[np.newaxis, :, np.newaxis] for marg in marginals])

    def f_premultiplied(Xs):
        return f(Xs) * ttr.core.sparse_reco(pdf, ttr.core.coordinates_to_indices(Xs, ticks_list=ticks_list))

    tt_pdf = ttr.core.cross(ticks_list, f_premultiplied, mode=mode, eps=eps,
                            verbose=verbose, **cross_kwargs)
    model_time = time.time() - model_time

    if verbose:
        print("\n-> Computing sensitivity metrics")
    sa_time = time.time()
    st = ttr.core.sobol_tt(tt_pdf, pdf=pdf, premultiplied=True, eps=eps,
                           verbose=verbose, **cross_kwargs)

    tst = ttr.core.to_upper(st)
    cst = ttr.core.to_lower(st)
    sst = ttr.core.to_superset(st)
    dim, var = ttr.core.effective_dimension(st, effective_threshold, 'truncation')
    indices, _ = ttr.core.largest_k_tuple(cst, dim)
    shapley_values = ttr.core.semivalues(cst, ps='shapley')
    banzhaf_coleman_values = ttr.core.semivalues(cst, ps='banzhaf-coleman')

    metrics = dict()
    metrics['variables'] = names
    metrics['dimension_distribution'] = ttr.core.sensitivity.dimension_distribution(st)
    metrics['mean_dimension'] = metrics['dimension_distribution'].dot(np.arange(1, N + 1))
    metrics['effective_superposition'] = ttr.core.effective_dimension(
        st, effective_threshold, 'superposition')
    metrics['effective_truncation'] = (dim, var, indices)
    metrics['effective_successive'] = ttr.core.effective_dimension(
        st, effective_threshold, 'successive')
    sa_time = time.time() - sa_time

    if verbose:
        print("\n-> Exporting results")

    stt_info = dict(
        _tr_info=('stt'),
        axes=axes,
        default_bins=default_bins,
        effective_threshold=effective_threshold,
        eps=eps,
        verbose=verbose,
        cross_kwargs=cross_kwargs,
        ticks_list=ticks_list,
        n_samples=f.n_samples,
        model_time=model_time,
        sa_time=sa_time,
        tt_pdf=tt_pdf,
        st=st,
        tst=tst,
        cst=cst,
        sst=sst
    )
    metrics.update(collect_var_indices(stt_info, max_order))

    metrics['shapley_values'] = dict()
    metrics['banzhaf_coleman_values'] = dict()
    for i, name in enumerate(names):
        metrics['shapley_values'][name] = shapley_values[i]
        metrics['banzhaf_coleman_values'][name] = banzhaf_coleman_values[i]
    # pprint.pprint(metrics)
    # pprint.pprint(stt_info)

    if show:
        print_metrics(metrics)

    return metrics, stt_info


def collect_var_indices(stt_info, max_order):
    """Collect Sobol sensitivity indices up to a maximum order from a Sobol Tensor Train.

    Args:
        stt_info (dict): Sobol Tensor Train metainformation
            (typically obtained from: _, stt_info = compute_var_metrics(...))
        max_order (int): Maximum order of the collected Sobol indices.

    Returns:
        dict: A dictionary with a structured and easily accessible collection
            of sensitivity metrics.

    """
    assert('stt' in stt_info['_tr_info'])
    assert(max_order > 0)

    axes = stt_info['axes']
    N = len(axes)
    names = [axis[0] for axis in axes]
    st = stt_info['st']
    tst = stt_info['tst']
    cst = stt_info['cst']
    sst = stt_info['sst']

    metrics = dict(sobol_indices=dict(), total_sobol_indices=dict(),
                   closed_sobol_indices=dict(), superset_sobol_indices=dict())
    for order in range(1, max_order + 1):
        metrics['sobol_indices'][order] = dict()
        metrics['total_sobol_indices'][order] = dict()
        metrics['closed_sobol_indices'][order] = dict()
        metrics['superset_sobol_indices'][order] = dict()

        for idx in itertools.combinations(list(range(N)), order):
            key = tuple(names[i] for i in idx)
            metrics['sobol_indices'][order][key] = ttr.core.set_choose(st, idx)
            metrics['total_sobol_indices'][order][key] = ttr.core.set_choose(tst, idx)
            metrics['closed_sobol_indices'][order][key] = ttr.core.set_choose(cst, idx)
            metrics['superset_sobol_indices'][order][key] = ttr.core.set_choose(sst, idx)

    return metrics


def print_metrics(metrics, tablefmt="presto", floatfmt=".6f",
                  numalign="decimal", stralign="left"):
    """Print tables with the contents of the collected sensitivity metrics.

    The tables are generated using 'tabulate'. It is possible to customize the
    output by passing optional arguments with specific tabulate format
    parameters.

    Args:
        metrics (dict): Collected sensitivity metrics
            (typically obtained from: metrics, _ = compute_var_metrics(...))
        tablefmt (str, optional): tabulate format parameter.
            Defaults to 'presto'.
        floatfmt (str, optional): tabulate format parameter.
            Defaults to '.6f'.
        numalign (str, optional): tabulate format parameter.
            Defaults to 'decimal'.
        stralign (str, optional): tabulate format parameter.
            Defaults to 'left'.

    """
    tab_fn = functools.partial(tabulate, tablefmt=tablefmt, floatfmt=floatfmt,
                               numalign=numalign, stralign=stralign)

    max_order = max(metrics['sobol_indices'].keys())
    names = metrics['variables']
    table = [(n, metrics['sobol_indices'][1][(n,)],
              metrics['total_sobol_indices'][1][(n,)],
              metrics['shapley_values'][n],
              metrics['banzhaf_coleman_values'][n])
             for n in names]
    print("\n\n\t\t -- Sensitivity Indices --".upper())
    print(tab_fn(table, headers=["Variable", "Sobol", "Total", "Shapley",
                                 "Banzhaf-Coleman"]))

    for order in range(2, max_order + 1):
        table = []
        for key in metrics['sobol_indices'][order].keys():
            table.append((*key, metrics['sobol_indices'][order][key],
                          metrics['total_sobol_indices'][order][key],
                          metrics['closed_sobol_indices'][order][key],
                          metrics['superset_sobol_indices'][order][key]))

        print("\n\n\t\t -- Sensitivity Indices (order {}) --".format(order).upper())
        print(tab_fn(table, headers=[*["Variable {}".format(i+1) for i in range(order)],
                                     "Sobol", "Total", "Closed", "Superset"]))

    table = [('Mean dimension', metrics['mean_dimension']),
             ('Effective (superposition)', *metrics['effective_superposition']),
             ('Effective (successive)', *metrics['effective_successive']),
             ('Effective (truncation)', metrics['effective_truncation'][0],
              metrics['effective_truncation'][1], ", ".join(
                  [names[i] for i in metrics['effective_truncation'][2]]))]

    print(("\n\n\t\t -- Dimension metrics --".upper()))
    print(tab_fn(table, headers=["Dimension Metric", "Value", "Rel. Variance",
                                 "Variables"]))

    cumul_dist = np.cumsum(metrics['dimension_distribution'])
    table = [[i + 1, value, cumul_dist[i]]
             for i, value in enumerate(metrics['dimension_distribution'])]
    print("\n\n\t -- Dimension distribution --".upper())
    print(tab_fn(table, headers=["Order", "Rel. Variance", "Cumul. Variance"]))
    print("\n")
