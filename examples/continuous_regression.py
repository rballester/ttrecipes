# -----------------------------------------------------------------------------
# Example interpolation in the TT format for a tensor with continuous variables
# This method is equivalent to polynomial chaos expansions (PCE) with 3 differences:
#
# - We discretize the polynomial bases along each axis. This is optional, e.g. the spectral
# tensor train decomposition (Bigoni et al., 2014) keeps the continuous factors. Our way introduces
# numerical error (usually negligible), but on the other hand, makes things simpler: the
# surrogate is a standard TT, and also no special treatment has to be given to categorical variables.
#
# - We consider the full space of polynomials (up to a bounded degree). In classical PCE this
# would be a problem due to the curse of dimensionality. So in practice PCE users bound the total
# degree sum*, use the hyperbolic constrain by Blatman and Sudret (2010), or add a regularization
# term to encourage sparsity.
#
# - The space of polynomial coefficients is expressed as a low-rank TT format. This adds extra
# regularization, and also a hyperparameter: which TT rank (or ranks) to take.
#
# Authors:      Rafael Ballester-Ripoll <rballester@ifi.uzh.ch>
#
# Copyright:    ttrecipes project (c) 2017
#               VMMLab - University of Zurich
# -----------------------------------------------------------------------------

import numpy as np
np.random.seed(1)
import sklearn.model_selection
import ttrecipes as tr
import ttrecipes.mpl


I = 64  # Ticks per axis
function, axes = tr.models.get_piston()  # An example model
_, ticks_list, _ = tr.models.parse_axes(axes, default_bins=I)  # Discretize axes to get a list of ticks
N = len(ticks_list)  # Number of variables
P = 500  # How many samples to draw

Xs = tr.core.LHS([I]*N, P)  # Draw P tensor entries at random using Latin Hypercube Sampling (LHS)
coords = tr.core.indices_to_coordinates(Xs, ticks_list=ticks_list)  # Convert discrete indices -> tick coordinates
ys = function(coords)  # Groundtruth evaluations
Xs_train, Xs_test, ys_train, ys_test = sklearn.model_selection.train_test_split(Xs, ys, test_size=0.25)  # Split into training and test

t = tr.core.pce_interpolation(Xs_train, ys_train, shape=[I] * N, ranks=3, ranks2=3, maxswp=2, verbose=True)  # Train a TT predictor
pred = tr.core.sparse_reco(t, Xs_test)  # Predict the test instances
print('Test relative error: {}'.format(np.linalg.norm(pred - ys_test) / np.linalg.norm(ys_test)))
# Visualize the result
tr.mpl.navigation(t, ticks_list=ticks_list, coords=coords, ys=ys, gt_range=50)
