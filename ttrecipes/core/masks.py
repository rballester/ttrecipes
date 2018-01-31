"""
2^N tensors that act as selection masks, built as automata
"""

# -----------------------------------------------------------------------------
# Authors:      Enrique G. Paredes <egparedes@ifi.uzh.ch>
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

import numpy as np
import tt


def hamming_weight(N):
    """
    Build a 2^N TT that stores the number of '1' bits of each position
    """

    assert(N > 0)

    cores = []
    core = np.ones([1, 2, N])
    core[0, 0, 0] = 0
    cores.append(core)
    for i in range(1, N - 1):
        core = np.repeat(np.eye(N)[:, np.newaxis, :], 2, axis=1)
        core[i, 0, i] = 0
        cores.append(core)
    core = np.ones([N, 2, 1])
    core[N - 1, 0, 0] = 0
    cores.append(core)
    return tt.vector.from_list(cores)


def hamming_weight_state(N):
    """
    Build an (N+1)-dimensional TT:
    [b1, ..., bN, i] = 1 if hamming_weight(b1...bN) == i, and 0 otherwise
    """

    assert(N > 0)

    cores = []
    core = np.zeros([1, 2, N + 1])
    core[0, 0, 0] = 1
    core[0, 1, 1] = 1
    cores.append(core)

    slices = [np.eye(N + 1)] * 2
    slices[1] = np.roll(np.eye(N + 1), 1, axis=1)
    slices[1][:, 0] = 0
    core = np.stack(slices, axis=1)
    cores.extend([core] * (N - 1))

    core = np.zeros([N + 1, N + 1, 1])
    for i in range(N + 1):
        core[i, i, 0] = 1
    cores.append(core)

    return tt.vector.from_list(cores)


def hamming_eq_mask(N, weight, invert=False, loss=0):
    """
    Build a 2^N TT that stores 1 at each position where the Hamming weight is
    equal to :param weight:, and a false value elsewhere. If :param loss: is 0,
    the false value would be also 0, otherwise it would be a value in the
    range [-loss ... 0), proportional to the distance to :param weight:
    """

    assert(0 <= weight <= N > 0)
    assert(loss >= 0)

    if invert:
        count_i = 0
    else:
        count_i = 1

    # Depending on the requested values, count True or False to obtain
    # a TT with the lowest possible rank
    if weight > N / 2:
        weight = N - weight
        count_i = 1 - count_i
    pass_i = 1 - count_i

    if weight == 0:
        core = np.zeros([1, 2, 1])
        core[0, pass_i, 0] = 1
        cores = [core] * N
    else:
        cores = []
        core = np.zeros([1, 2, weight + 1])
        core[0, pass_i, 0] = 1
        core[0, count_i, 1] = 1
        cores.append(core)

        slices = [np.eye(weight + 1)] * 2
        slices[count_i] = np.roll(np.eye(weight + 1), 1, axis=1)
        slices[count_i][:, 0] = 0
        core = np.stack(slices, axis=1)
        cores.extend([core] * (N - 2))

        core = np.zeros([weight + 1, 2, 1])
        core[weight, pass_i, 0] = 1
        core[weight - 1, count_i, 0] = 1
        cores.append(core)

    if loss > 0:
        loss_step = -loss / (N/2 + abs(weight - N/2))
        core = cores[0]
        new_core = np.zeros((core.shape[0], core.shape[1], core.shape[2] + 2))
        new_core[:, :, :-2] = core
        new_core[:, pass_i, -2] = -1 if weight > 0 else 1
        new_core[:, count_i, -2] = -1 if weight > 1 else 1
        new_core[:, pass_i, -1] = weight
        new_core[:, count_i, -1] = abs(weight - 1)
        cores[0] = new_core

        core = cores[1]
        new_core = np.zeros((core.shape[0] + 2, core.shape[1], core.shape[2] + 2))
        new_core[:-2, :, :-2] = core
        new_core[-2, :, -2] = 1
        if core.shape[0] > 1:
            new_core[-4, count_i, -2] = 2
        new_core[-1, :, -1] = 1
        new_core[-2, count_i, -1] = 1
        cores[1:N - 1] = [new_core] * (N - 2)

        core = cores[-1]
        new_core = np.zeros((core.shape[0] + 2, core.shape[1], core.shape[2]))
        new_core[:-2, :, :] = core
        new_core[-2, pass_i, 0] = 0
        new_core[-2, count_i, 0] = loss_step
        new_core[-1, pass_i, 0] = loss_step
        new_core[-1, count_i, 0] = loss_step
        cores[-1] = new_core

    return tt.vector.from_list(cores)


def hamming_le_mask(N, weight, invert=False):
    """
    Build a 2^N TT that stores 1 at each position where the Hamming weight is
    less or equal to :param weight:, and 0 elsewhere
    """

    assert(0 <= weight <= N)

    if invert: 
        count_i = 0
    else:
        count_i = 1
    pass_i = 1 - count_i

    # Depending on the requested values, count True or False to obtain
    # a TT with the lowest possible rank
    if weight > N / 2:
        return hamming_ge_mask(N, N - weight, invert=(count_i == 1))

    if weight == 0:
        core = np.zeros([1, 2, weight + 1])
        core[0, pass_i, 0] = 1
        cores = [core] * N
    else:
        cores = []
        core = np.zeros([1, 2, weight + 1])
        core[0, pass_i, 0] = 1
        core[0, count_i, 1] = 1
        cores.append(core)

        slices = [np.eye(weight + 1)] * 2
        slices[count_i] = np.roll(np.eye(weight + 1), 1, axis=1)
        slices[count_i][:, 0] = 0
        core = np.stack(slices, axis=1)
        cores.extend([core] * (N - 2))

        core = np.ones([weight + 1, 2, 1])
        core[weight, count_i, 0] = 0
        cores.append(core)

    return tt.vector.from_list(cores)


def hamming_lt_mask(N, weight, invert=False):
    """
    Build a 2^N TT that stores 1 at each position where the Hamming weight is
    strictly less than to :param weight:, and 0 elsewhere
    """
    return hamming_le_mask(N, weight - 1, invert=invert)


def hamming_ge_mask(N, weight, invert=False):
    """
    Build a 2^N TT that stores 1 at each position where the Hamming weight is
    larger or equal to :param weight:, and 0 elsewhere
    """

    assert(0 <= weight <= N)

    if invert: 
        count_i = 0
    else:
        count_i = 1
    pass_i = 1 - count_i

    # Depending on the requested values, count True or False to obtain
    # a TT with the lowest possible rank
    if weight > N / 2:
        return hamming_le_mask(N, N - weight, invert=(count_i == 1))

    if weight == 0:
        core = np.ones([1, 2, 1])
        cores = [core] * N
    else:
        cores = []
        core = np.zeros([1, 2, weight + 1])
        core[0, pass_i, 0] = 1
        core[0, count_i, 1] = 1
        cores.append(core)

        slices = [np.eye(weight + 1)] * 2
        slices[count_i] = np.roll(np.eye(weight + 1), 1, axis=1)
        slices[count_i][:, 0] = 0
        slices[count_i][weight, weight] = 1
        core = np.stack(slices, axis=1)
        cores.extend([core] * (N - 2))

        core = np.zeros([weight + 1, 2, 1])
        core[weight, pass_i, 0] = 1
        core[weight - 1, count_i, 0] = 1
        core[weight, count_i, 0] = 1
        cores.append(core)

    return tt.vector.from_list(cores)


def hamming_gt_mask(N, weight, invert=False):
    """
    Build a 2^N TT that stores 1 at each position where the Hamming weight is
    strictly larger than to :param weight:, and 0 elsewhere
    """
    return hamming_ge_mask(N, weight + 1, invert=invert)


def lness_len(N):
    """
    LNESS: Largest Non-Empty Sub-String

    Compute the length of the string comprised between the first and last '1' bits
    in the index. E.g. [0, 1, 0, 1, 1, 0] -> 4
    """

    cores = []
    core = np.zeros((1, 2, 4))
    core[0, 0, 0] = 1.
    core[0, :, -1] = 1.
    core[0, 1, -2] = 1.
    cores.append(core)

    for i in range(1, N-1):
        core = np.zeros((4, 2, 4))
        core[:, 0, :] = np.eye(4)
        core[:, 1, 1] = [i, 1., 0., 0.]
        core[:, 1, 2] = [-i, -1, 0, i+1]
        core[-1, 1, -1] = 1
        cores.append(core)

    core = np.zeros((4, 2, 1))
    core[-2, 0, 0] = 1
    core[:, 1, 0] = [-(N-1.0),-1, 0, N]
    cores.append(core)

    return tt.vector.from_list(cores)


def lness_le_mask(N, length):
    """
    LNESS: Longest Non-Empty Sub-String

    Build a 2^N TT that stores 1 at each position where the length of the bitstring
    comprised between the first and last '1' in the index is less or equal
    to :param length:, and 0 elsewhere.
    """

    assert(0 <= length <= N)

    zero_i = 0
    one_i = 1

    # Ranks: current_state
    n_ranks = length + 1
    if length == 0:
        core = np.zeros([1, 2, 1])
        core[0, zero_i, 0] = 1
        cores = [core] * N
    else:
        cores = []
        core = np.zeros([1, 2, n_ranks])
        core[0, zero_i, 0] = 1
        core[0, one_i, 1] = 1
        # print('Core0\n', core[:,0,:], '\n\t -- 1 --\n', core[:,1,:], '\n\t ===== \n')
        cores.append(core)

        core = np.zeros([n_ranks, 2, n_ranks])
        core[0, zero_i, 0] = 1
        core[1:, zero_i, 1:] = np.roll(np.eye(n_ranks - 1), 1, axis=1)
        core[:, zero_i, 1] = 0
        core[-1, zero_i, -1] = 1
        core[1:, one_i, 1:] = np.roll(np.eye(n_ranks - 1), 1, axis=1)
        core[:, one_i, 1] = 0
        core[0, one_i, 1] = 1
        # print('CoreI\n', core[:,0,:], '\n\t -- 1 --\n', core[:,1,:], '\n\t ===== \n')
        cores.extend([core] * (N - 2))

        core = np.zeros([n_ranks, 2, 1])
        core[:, zero_i, 0] = 1
        core[:-1, one_i, 0] = 1
        # print('CoreNI\n', core[:,0,:], '\n\t -- 1 --\n', core[:,1,:], '\n\t ===== \n')
        cores.append(core)

    return tt.vector.from_list(cores)


def lness_state(N):
    """
    LNESS: Longest Non-Empty Sub-String

    Build an (N+1)-dimensional TT:
    [b1, ..., bN, i] = 1 if Longest Non-Empty Sub-String(b1...bN) == i, and 0 otherwise
    """

    assert N > 2
    zero_i = 0
    one_i = 1

    # Ranks: two copies of current_state with one hot encoding: [confirmed_length, possible_length]
    # confirmed_length uses N+1 ranks [0:N+1]
    # possible_length uses N-1 ranks [N+1:2*N] (reuses states 0 and N)
    n_ranks = 2 * N
    cores = []
    core = np.zeros([1, 2, n_ranks])
    core[0, zero_i, 0] = 1
    core[0, one_i, 1] = 1
    core[0, one_i, N+1] = 1
    # print('Core0\n', core[:,0,:], '\n\t -- 1 --\n', core[:,1,:], '\n\t ===== \n')
    cores.append(core)

    core = np.zeros([n_ranks, 2, n_ranks])
    # confirmed_length transitions:
    core[:N+1, zero_i, :N+1] = np.eye(N + 1)
    core[0, one_i, 1] = 1
    core[N+1:, one_i, 2:N+1] = np.eye(N - 1)
    # possible_length transitions:
    core[0, one_i, N+1] = 1
    core[N+1:-1, zero_i, N+2:] = np.eye(N - 2)
    core[N+1:-1, one_i, N+2:] = np.eye(N - 2)
    # print('CoreI\n', core[:,0,:], '\n\t -- 1 --\n', core[:,1,:], '\n\t ===== \n')
    cores.extend([core] * (N - 1))

    core = np.zeros([n_ranks, N + 1, 1])
    core[:N+1, :, 0] = np.eye(N + 1)
    # print('CoreN+1\n')
    # for i in range(N+1):
    #     print(core[:,i,:], '\n\t -- {} --\n'.format(i+1))
    cores.append(core)

    return tt.vector.from_list(cores)
