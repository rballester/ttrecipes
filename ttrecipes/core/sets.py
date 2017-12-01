"""
Operations for 2^N tensors representing set cardinalities (or hyperedges in a hypergraph). All input and output tensors are ttpy's tensor trains
"""

# -----------------------------------------------------------------------------
# Authors:      Rafael Ballester-Ripoll <rballester@ifi.uzh.ch>
#
# Copyright:    TensorChart project (c) 2016-2017
#               VMMLab - University of Zurich
# -----------------------------------------------------------------------------

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import tt
import tt.optimize.tt_min as tt_min

import ttrecipes as tr


def complement(t):
    return tt.vector.from_list([np.concatenate([core[:, 1:2, :], core[:, 0:1, :]], axis=1) for core in tt.vector.to_list(t)])  # Swap the two slices of each core


def to_superset(t):
    return tt.vector.from_list([np.concatenate([core[:, 0:1, :] + core[:, 1:2, :], core[:, 1:2, :]], axis=1) for core in tt.vector.to_list(t)])  # The first slice becomes the sum of both


def from_superset(t):
    return tt.vector.from_list([np.concatenate([core[:, 0:1, :] - core[:, 1:2, :], core[:, 1:2, :]], axis=1) for core in tt.vector.to_list(t)])


def to_lower(t):
    return tt.vector.from_list([np.concatenate([core[:, 0:1, :], core[:, 0:1, :] + core[:, 1:2, :]], axis=1) for core in tt.vector.to_list(t)])  # The second slice becomes the sum of both


def from_lower(t):
    return tt.vector.from_list([np.concatenate([core[:, 0:1, :], core[:, 1:2, :] - core[:, 0:1, :]], axis=1) for core in tt.vector.to_list(t)])


def to_upper(t):
    return tt.vector.from_list([np.ones([1, sh, 1]) for sh in t.n]) - tt.vector.from_list([np.concatenate([core[:, 0:1, :] + core[:, 1:2, :], core[:, 0:1, :]], axis=1) for core in tt.vector.to_list(t)])  # 1 - the lower's complement


def from_upper(t):
    return tt.vector.from_list([np.ones([1, sh, 1]) for sh in t.n]) - tt.vector.from_list([np.concatenate([core[:, 1:2, :], core[:, 0:1, :] - core[:, 1:2, :]], axis=1) for core in tt.vector.to_list(t)])  # 1 - the complement's from_lower


def largest_k_tuple(t, k, verbose=False, **kwargs):
    """
    Find the largest element of a given order

    :param st: a 2^N TT
    :param k: a positive integer
    :return: (a vector, its value)

    """

    assert k >= 1
    assert np.all(t.n == 2)

    N = t.d
    weighted = t * tr.core.hamming_eq_mask(N, k, loss=1e-6)
    val, point = tr.core.maximize(weighted, verbose=verbose, **kwargs)
    return np.where(point)[0], val


def set_dump(t, min_order=1, max_order=1):
    """
    Return a TT (interpreted as a power set) as a string, ordered lexicographically

    :param t: a 2^N TT
    :param min_order: orders below this will be skipped (default is 1)
    :param max_order: orders above this will be skipped (default is 1)

    """

    assert np.all(t.n == 2)
    assert 0 <= min_order <= t.d
    assert min_order <= max_order <= t.d
    inds = {}
    strings = []

    def recursive(inds, max_order=2, maximum=-1):
        if len(inds.keys()) >= min_order:
            strings.append(str(list(inds.keys())) + ': {}'.format(set_choose(t, inds.keys())))
        if len(inds.keys()) >= max_order:
            return
        for i in range(maximum+1, t.d):
            inds[i] = 1
            recursive(inds, max_order=max_order, maximum=i)
            inds.pop(i)

    recursive(inds, max_order=max_order)
    return '\n'.join(strings)


def set_choose(t, modes):
    """
    Interpret a TT as a power set and return the value associated to a certain subset

    :param t: a 2^N TT
    :param modes:
    :return:

    """

    assert np.all(t.n == 2)

    if not hasattr(modes, '__len__'):
        modes = [modes]
    index = [0, ]*t.d
    for mode in modes:
        index[mode] = 1
    return t[index]


def cardinality_deviation(t):
    """
    Given a TT set, return another one that maps each tuple to its "deviation" or "disproportionality" from its expected cardinality

    References:
    - Lex et al., "UpSet: Visualization of Intersecting Sets"
    - Alsallakh et al., "Radial sets: Interactive visual analysis of large overlapping set"

    :param t: a 2^N TT
    :return:

    """

    N = t.d
    total = tr.core.sum(t)
    singletons = tr.core.sparse_reco(t, np.eye(N))
    cores = [np.concatenate([(1 - singletons[n]/total)[np.newaxis, np.newaxis, np.newaxis], (singletons[n]/total)[np.newaxis, np.newaxis, np.newaxis]], axis=1) for n in range(N)]
    return t*(1/total) - tt.vector.from_list(cores)


def mean_dimension_tensor(t, eps=1e-6, verbose=False, **kwargs):
    """
    Given a TT set t, return another that maps each tuple to the mean dimension of t restricted to the tuple

    :param t: a 2^N TT
    :return:

    """

    N = t.d

    ct = tr.core.to_superset(t)
    ct = tt.vector.from_list([core[:, [0, 0, 1], :] for core in tt.vector.to_list(ct)])

    t = tt.vector.from_list([core[:, [0, 1, 1], :] for core in tt.vector.to_list(t)])

    w = tr.core.hamming_weight(N)
    w = tt.vector.from_list([core[:, [0, 1, 1], :] for core in tt.vector.to_list(w)])

    def fun(Xs):
        result = np.zeros(len(Xs))
        idx = np.where(Xs[:, 2] != 0)[0]
        result[idx] = Xs[idx, 0]*Xs[idx, 1]/Xs[idx, 2]
        return result

    t = tt.multifuncrs2([t, w, ct], fun, eps=eps, verb=verbose, **kwargs)
    t = tt.vector.from_list([np.concatenate([np.sum(core[:, 0:2, :], axis=1, keepdims=True), core[:, 2:3, :]], axis=1) for core in tt.vector.to_list(t)])
    return t.round(eps=0)


def power_set(N, min_order=0, max_order=None, include=(), exclude=()):  # TODO as generator
    """
    Generate (in lexicographical order) all subsets of [0, ..., N-1]

    :param N:
    :param min_order: subsets of size smaller than this are skipped
    :param max_order: subsets of size larger than this are skipped
    :param include: these elements will be included. Default is ()
    :param exclude: these elements will be excluded. Default is ()
    :return: a list with all requested subsets

    """

    if max_order is None:
        max_order = N

    result = []
    candidates = np.arange(N)
    assert np.all(np.isin(include, candidates))
    assert np.all(np.isin(exclude, candidates))
    if np.intersect1d(include, exclude).size > 0:
        return []
    candidates = np.delete(candidates, include+exclude)

    def recursive(inds, maximum):
        depth = len(inds.keys())
        if depth >= min_order:
            result.append(np.sort(list(inds.keys())))
        if depth == max_order:
            return
        for i in range(maximum + 1, len(candidates)):
            inds[candidates[i]] = 1
            recursive(inds, i)
            inds.pop(candidates[i])

    inds = {i: 1 for i in include}
    recursive(inds, -1)
    return result


def order_query(t, min_order=0, max_order=1, include=(), exclude=(), k=None, threshold=None):
    """
    Compute elements of a TT-set, optionally capping their order and/or magnitude

    :param t: a 2^N TT
    :param min_order: orders below this will be excluded. Default is 0
    :param max_order: orders above this will be excluded. Default is 1
    :param include: these elements will be included. Default is ()
    :param exclude: these elements will be excluded. Default is ()
    :param k: only the `k` largest elements will be returned (default is None)
    :param threshold: only elements above this value will be returned (default is None)
    :return: a list of pairs (tuple, value) sorted alphabetically by tuple

    """

    N = t.d
    tuples = tr.core.power_set(N, min_order=min_order, max_order=max_order, include=include, exclude=exclude)
    tuples = [(tuple, tr.core.set_choose(t, tuple)) for tuple in tuples]
    if threshold is not None:
        tuples = [tuple for tuple in tuples if tuple[1] >= threshold]
    if k is not None and len(tuples) > k:
        all_values = [tuple[1] for tuple in tuples]
        idx = np.argsort(all_values)
        tuples = [tuples[i] for i in idx[-k:]]
    return tuples
