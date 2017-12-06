#!/usr/bin/env python

# -----------------------------------------------------------------------------
# Authors:      Enrique G. Paredes <egparedes@ifi.uzh.ch>
#               Rafael Ballester-Ripoll <rballester@ifi.uzh.ch>
#
# Copyright:    ttrecipes project (c) 2017
#               VMMLab - University of Zurich
# -----------------------------------------------------------------------------

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import pprint
import random

import numpy as np

import ttrecipes as tr


def main():
    default_bins = 100
    default_order = 2
    default_dims = 20
    default_acoeff = 0.0

    parser = argparse.ArgumentParser(
        description="Perform sensitivity analysis of a the Sobol G function.")
    parser.add_argument('--seed', type=int,
                        help="Random seed (default: not set)")
    parser.add_argument('--verbose', action='store_true',
                        help="Verbose mode (default: False)")
    parser.add_argument('-b', '--bins', default=default_bins, type=int,
                        help="Number of samples for each input axis "
                             "(default: {})".format(default_bins))
    parser.add_argument('--order', default=default_order, type=int,
                        help="Maximum order of the collected indices "
                             "(default: {})".format(default_order))
    parser.add_argument('-d', '--dims', default=default_dims, type=int,
                        help='Number of dimensions '
                             '(default: {})'.format(default_dims))
    parser.add_argument('-a', '--acoeff', default=default_acoeff, type=float,
                        help='Constant oefficient for Sobol G function '
                             '(default: {})'.format(default_acoeff))

    args = parser.parse_args()
    print("*TT recipes* example:", parser.description, "\n")
    if args.verbose:
        pprint.pprint(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    f, axes = tr.models.get_sobol_g(args.dims, a=args.acoeff, 
                                    name_tmpl='X_{:0>2}')
    if args.verbose:
        pprint.pprint(axes)

    print("+ Computing tensor approximations of variance-based sensitivity metrics...")
    metrics, stt_info = tr.sensitivity_analysis.compute_var_metrics(
        f, axes, default_bins=args.bins, verbose=args.verbose, eps=1e-10,
        cross_kwargs=dict(kickrank=2), max_order=args.order, show=True)


if __name__ == "__main__":
    main()
