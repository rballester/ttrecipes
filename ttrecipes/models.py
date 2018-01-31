# -*- coding: utf-8 -*-
"""
Sample functions for surrogate modeling and sensitivity analysis.

In most cases we report the function's sensitivity metrics extracted as:

```
import ttrecipes as tr

function, axes = tr.models.get_modelname()
tr.sensitivity_analysis.var_metrics(function, axes)
```
"""

# -----------------------------------------------------------------------------
# Authors:      Enrique G. Paredes <egparedes@ifi.uzh.ch>
#               Rafael Ballester-Ripoll <rballester@ifi.uzh.ch>
#
# Copyright:    TT Recipes project (c) 2016-2017
#               VMMLab - University of Zurich
# -----------------------------------------------------------------------------

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import collections
import scipy as sp
import scipy.stats


def parse_axes(axes, default_bins, dist_fraction=1e-4/2):
    N = len(axes)
    names = [''] * N
    ticks_list = [None] * N
    marginals = [None] * N

    for i, axis in enumerate(axes):
        names[i] = axis.get('name', 'x_{}'.format(i))
        bounds = None
        samples = None

        domain = axis.get('domain', None)
        if domain is not None:
            if len(domain) == 2 and not isinstance(domain, np.ndarray):
                bounds = list(domain)
            else:
                samples = np.asarray(domain)

        dist = axis.get('dist', None)
        if dist is not None:
            if bounds is None:
                bounds = [None, None]
        elif samples is not None:
            dist = np.ones(samples.shape)
        elif bounds is not None:
            dist = sp.stats.uniform(loc=bounds[0], scale=bounds[1] - bounds[0])
        else:
            dist = sp.stats.uniform(loc=0, scale=1)
            bounds = [0., 1.]

        assert dist is not None and (bounds is not None or samples is not None)

        if isinstance(dist, scipy.stats.distributions.rv_frozen):
            if samples is None:
                if bounds[0] is None:
                    bounds[0] = dist.ppf(dist_fraction)
                if bounds[1] is None:
                    bounds[1] = dist.ppf(1.0 - dist_fraction)
                half_bin = (bounds[1] - bounds[0]) / (2 * default_bins)
                samples = np.linspace(bounds[0] + half_bin,
                                      bounds[1] - half_bin, default_bins)

            ticks_list[i] = np.asarray(samples)
            marginals[i] = dist.pdf(ticks_list[i])

        elif isinstance(dist, collections.Sized):
            dist = np.asarray(dist)
            if (not isinstance(samples, np.ndarray) or
                    len(samples) != len(dist) or len(dist) == 0):
                raise ValueError("Axes[{}]: 'dist' and 'domain' ndarrays must "
                                 "have equal and valid length".format(i))

            ticks_list[i] = samples
            marginals[i] = dist

        else:
            raise ValueError("Unrecognized axis distribution: must be either a vector "
                             "containing the discretized pdf, or a frozen distribution "
                             "from scipy.stats.distributions")

        marginals[i] /= np.sum(marginals[i])

        assert np.isfinite(ticks_list[i]).all() and np.isfinite(marginals[i]).all()

    return names, ticks_list, marginals


def get_linear_mix(N, betas=1.0, sigmas=1.0, name_tmpl='x_{}'):
    """Test function used for various numerical estimation methods.

    References:
        - Bertrand Iooss, Clementine Prieur. "Shapley effects for
            sensitivity analysis with dependent inputs: comparisons with
            Sobol’ indices, numerical estimation and applications".

         https://hal.inria.fr/hal-01556303/document

    """

    assert N > 0

    betas = np.asarray(betas) * np.ones(N)
    sigmas = np.asarray(sigmas) * np.ones(N)

    def function(Xs, betas=betas):
        assert Xs.shape[1] == N == len(betas)
        return Xs.dot(betas)

    axes = [dict(name=name_tmpl.format(n), dist=sp.stats.norm(scale=sigma))
            for n, sigma in enumerate(sigmas)]

    return function, axes


def get_ishigami(a=7, b=0.1, name_tmpl='x_{}'):
    """The Ishigami function of Ishigami & Homma (1990).

     This very well-known function exhibits strong nonlinearity and
     nonmonotonicity. It also has a peculiar dependence on x3, as described
     by Sobol' & Levitan (1999). The values of a and b used by
     Crestaux et al. (2007) and Marrel et al. (2009) are: a = 7 and b = 0.1.
     Sobol' & Levitan (1999) use a = 7 and b = 0.05.

    References:
        - http://www.sfu.ca/~ssurjano/ishigami.html
        - Sobol', I. M., & Levitan, Y. L. (1999). "On the use of variance
            reducing multipliers in Monte Carlo computations of a global
            sensitivity index". Computer Physics Communications, 117(1), 52-61.
        - Ishigami, T., & Homma, T. (1990, December). "An importance quantification
            technique in uncertainty analysis for computer models"
            In Uncertainty Modeling and Analysis, 1990. Proceedings.,
            First International Symposium on (pp. 398-403). IEEE.

            ** SENSITIVITY INDICES **
    Variable        Sobol     Total    Shapley    Banzhaf-Coleman
    ----------  ---------  --------  ---------  -----------------
    x_0          0.313925  0.557366   0.435645           0.435645
    x_1          0.442634  0.442634   0.442634           0.442634
    x_2          0.000000  0.243441   0.121720           0.121720

            ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.756559           0.756559
          2         0.243441           1.000000

            ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  -------------
    Mean dimension             1.243441
    Effective (superposition)  2.000000         1.000000
    Effective (successive)     3.000000         1.000000
    Effective (truncation)     3.000000         1.000000  x_0, x_1, x_2

    """

    def function(Xs, a=a, b=b):
        assert Xs.shape[1] == 3
        return np.sin(Xs[:, 0]) + a * np.sin(Xs[:, 1])**2 + b * (Xs[:, 2]**4) * np.sin(Xs[:, 0])

    axes = [dict(name=name_tmpl.format(n), domain=(-np.pi, np.pi)) for n in range(3)]

    return function, axes


def get_sobol_g(N, a=6.52, name_tmpl='x_{}'):
    """Test function used for various numerical estimation methods.

    The function is integrated over the hypercube [0, 1], ``for i in range(d)``
    For each index ``i``, a lower value of ai indicates a higher importance
    of the input variable ``x_i``.

    Kucherenko et al. (2011) use the values a0 = a1 = 0 and a3 = … = ad = 6.52.
    When used with ai = 6.52, for all i = 1, … d, they classify it as a Type B function,
    meaning that it has dominant low-order terms and a small effective dimension.

    In many cases, for instance in Marrel et al. (2008), there is the constraint
    that ai ≥ 0. They conclude that:

        for ai = 0, xi is very important
        for ai = 1, xi is relatively important
        for ai = 9, xi is non-important
        for ai = 99, xi is non-significant


    References:
        - http://www.sfu.ca/~ssurjano/gfunc.html
        - Kucherenko, S., Feil, B., Shah, N., & Mauntz, W. (2011). "The identification
            of model effective dimensions using global sensitivity analysis"
            Reliability Engineering & System Safety, 96(4), 440-449.
        - Marrel, A., Iooss, B., Laurent, B., & Roustant, O. (2009).
            "Calculations of sobol indices for the gaussian process metamodel"
            Reliability Engineering & System Safety, 94(3), 742-751.
        - Saltelli, A., & Sobol', I. Y. M. (1994). "Sensitivity analysis for
            nonlinear mathematical models: numerical experience"
            Matematicheskoe Modelirovanie, 7(11), 16-28.

    References (a=0 -> davis_rabinowitz_integral):
        - Kucherenko, S., Feil, B., Shah, N., & Mauntz, W. (2011). "The identification
            of model effective dimensions using global sensitivity analysis"
            Reliability Engineering & System Safety, 96(4), 440-449.
        -  Bratley P, Fox B. (1988). "Algorithm 659: Implementing Sobol's
            Quasirandom Sequence Generator". ACM Transactions on
            Mathematical Software. Vol. 14, Number 1, March 1988, pages 88-100
        - Philip J. Davis Philip Rabinowitz (1984). "Methods of Numerical
            Integration, 2nd Edition".  Academic Press

    """

    assert N > 0

    a = np.asarray(a) * np.ones(N)

    def function(Xs, a=a):
        assert(Xs.shape[1] == N)
        result = np.ones(Xs.shape[0])
        for i in range(Xs.shape[1]):
            result *= (np.abs(4. * Xs[:, i] - 2) + a[i]) / (1. + a[i])
        return result

    axes = [dict(name=name_tmpl.format(n)) for n in range(N)]

    return function, axes


def get_prod(N, k=1, name_tmpl='x_{}'):
    """Analytical test function used for sensitivity analysis.

    References:
        - Kucherenko, S., Feil, B., Shah, N., & Mauntz, W. (2011). "The identification
            of model effective dimensions using global sensitivity analysis"
            Reliability Engineering & System Safety, 96(4), 440-449.

    """

    assert N > 0

    def function(Xs, k=k):

        Xs = np.asarray(Xs)
        assert(Xs.shape[1] == N)
        return np.prod(np.abs(k * Xs - 2), axis=1)

    axes = [dict(name=name_tmpl.format(n)) for n in range(N)]

    return function, axes


def get_decay_poisson(N, span=5.0, hl_range=(3*(1.0/12.0), 3.0),
                      time_step=1.0 / 365, name_tmpl='x_{}'):
    """Simulated decay of species with different decay rates

        - Xs: decay_rates in years
        - span(float): simulated time span in years
        - output: fractions of the last specie

    References:
        - Andrea Saltelli, Paola Annoni. "How to avoid a perfunctory sensitivity analysis"
            http://ac.els-cdn.com/S1364815210001180/1-s2.0-S1364815210001180-main.pdf?_tid=96c74b44-365c-11e7-9cc2-00000aab0f27&acdnat=1494515875_1dcbccb878ead778df602acd24051628
        - http://129.173.120.78/~kreplak/wordpress/wp-content/uploads/2010/12/Phyc2150_2016_Lecture10.pdf

    """
    assert N > 0

    def half_life_to_decay_rate(half_lives, year_fraction):
        # decay rates as the fraction of the original number of atoms that decays per year_fraction
        return 1 - np.exp(- year_fraction * np.log(2) / np.asarray(half_lives))

    def function(Xs, span=span, time_step=time_step):
        Xs = np.asarray(Xs)
        if Xs.ndim == 1:
            Xs = np.expand_dims(Xs, axis=0)
        n_products = Xs.shape[1]
        quantities = np.zeros((Xs.shape[0], n_products + 1))
        quantities[:, 0] = 1.0

        # decay rates as average number of atom decays per time step
        # rates = 1 - np.exp(- day * math.log(2) / Xs)
        # rates = half_life_to_decay_rate(Xs, 1.0 / 365.0)
        rates = Xs
        n_steps = int(span / time_step)
        for n in range(n_steps):
            offsets = quantities[:, :n_products] * rates
            quantities[:, :n_products] -= offsets
            quantities[:, 1:] += offsets

        return quantities[:, -1]

    decay_range = tuple(half_life_to_decay_rate(hl_range, time_step)[::-1])
    axes = [dict(name=name_tmpl.format(n), domain=decay_range) for n in range(N)]

    return function, axes


def get_borehole():
    """Example of analytical model with non-uniform variables

    References:
        - http://www.sfu.ca/~ssurjano/borehole.html
        - https://tel.archives-ouvertes.fr/tel-01143694/document (p. 84)

             ** SENSITIVITY INDICES **
    Variable       Sobol     Total    Shapley    Banzhaf-Coleman
    ----------  --------  --------  ---------  -----------------
    r_w         0.659434  0.689659   0.674489           0.674460
    r           0.000002  0.000003   0.000002           0.000002
    T_u         0.000000  0.000000   0.000000           0.000000
    H_u         0.096265  0.107466   0.101838           0.101825
    T_l         0.000006  0.000008   0.000007           0.000007
    H_l         0.096265  0.107466   0.101838           0.101825
    L           0.091975  0.104098   0.097988           0.097963
    K_w         0.022259  0.025449   0.023837           0.023829

         ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.966205           0.966205
          2         0.033441           0.999646
          3         0.000353           0.999999
          4         0.000001           1.000000
          5         0.000000           1.000000

             ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  ----------------
    Mean dimension             1.034149
    Effective (superposition)  1.000000         0.966205
    Effective (successive)     1.000000         0.966205
    Effective (truncation)     4.000000         0.974540  r_w, H_u, H_l, L

    """

    def function(Xs):
        rw = Xs[:, 0]
        r = Xs[:, 1]
        Tu = Xs[:, 2]
        Hu = Xs[:, 3]
        Tl = Xs[:, 4]
        Hl = Xs[:, 5]
        L = Xs[:, 6]
        Kw = Xs[:, 7]

        frac1 = 2 * np.pi * Tu * (Hu - Hl)
        frac2a = 2 * L * Tu / (np.log(r / rw) * (rw**2) * Kw)
        frac2b = Tu / Tl
        frac2 = np.log(r / rw) * (1 + frac2a + frac2b)

        return frac1 / frac2

    axes = [dict(name='r_w', domain=(0.05, 0.15), dist=sp.stats.norm(loc=0.10, scale=0.0161812)),
            dict(name='r', domain=(100, 50000), dist=sp.stats.lognorm(scale=np.exp(7.71), s=1.0056)),
            dict(name='T_u', domain=(63070, 115600)),
            dict(name='H_u', domain=(990, 1110)),
            dict(name='T_l', domain=(63.1, 116)),
            dict(name='H_l', domain=(700, 820)),
            dict(name='L', domain=(1120, 1680)),
            dict(name='K_w', domain=(9855, 12045))]

    return function, axes


def get_piston():
    """Piston simulation routine.

    References:
        - http://www.sfu.ca/~ssurjano/piston.html

             ** SENSITIVITY INDICES **
    Variable       Sobol     Total    Shapley    Banzhaf-Coleman
    ----------  --------  --------  ---------  -----------------
    M           0.039118  0.050628   0.044690           0.044598
    S           0.557301  0.599225   0.576625           0.575805
    V_0         0.321154  0.352602   0.335269           0.334465
    k           0.020614  0.066660   0.042092           0.041319
    P_0         0.001237  0.001324   0.001274           0.001271
    T_a         0.000003  0.000008   0.000006           0.000006
    T_0         0.000026  0.000066   0.000045           0.000044

         ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.939453           0.939453
          2         0.050614           0.990066
          3         0.009902           0.999968
          4         0.000032           1.000000

             ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  ------------
    Mean dimension             1.070513
    Effective (superposition)  2.000000         0.990066
    Effective (successive)     2.000000         0.960821
    Effective (truncation)     4.000000         0.998602  M, S, V_0, k

    """

    def function(Xs):
        M = Xs[:, 0]
        S = Xs[:, 1]
        V0 = Xs[:, 2]
        k = Xs[:, 3]
        P0 = Xs[:, 4]
        Ta = Xs[:, 5]
        T0 = Xs[:, 6]

        Aterm1 = P0 * S
        Aterm2 = 19.62 * M
        Aterm3 = -k * V0 / S
        A = Aterm1 + Aterm2 + Aterm3

        Vfact1 = S / (2 * k)
        Vfact2 = np.sqrt(A**2 + 4 * k * (P0 * V0 / T0) * Ta)
        V = Vfact1 * (Vfact2 - A)

        fact1 = M
        fact2 = k + (S**2) * (P0 * V0 / T0) * (Ta / (V**2))

        C = 2 * np.pi * np.sqrt(fact1 / fact2)
        return C

    axes = [dict(name='M', domain=(30, 60)),
            dict(name='S', domain=(0.005, 0.020)),
            dict(name='V_0', domain=(0.002, 0.010)),
            dict(name='k', domain=(1000, 5000)),
            dict(name='P_0', domain=(90000, 110000)),
            dict(name='T_a', domain=(290, 296)),
            dict(name='T_0', domain=(340, 360))]

    return function, axes


def get_dike(output='cost'):
    """Analytical model with many non-uniform variables.

    References:
        - http://statweb.stanford.edu/~owen/pubtalks/siamUQ.pdf, page 30

             ** SENSITIVITY INDICES **
    Variable       Sobol     Total    Shapley    Banzhaf-Coleman
    ----------  --------  --------  ---------  -----------------
    Q           0.380565  0.512742   0.444096           0.442818
    K_s         0.167290  0.268184   0.215444           0.214298
    Z_v         0.179477  0.237146   0.206648           0.205817
    Z_m         0.004123  0.008207   0.005968           0.005870
    H_d         0.062146  0.122725   0.090833           0.090032
    C_b         0.032133  0.042160   0.036857           0.036713
    L           0.000000  0.000001   0.000001           0.000001
    B           0.000106  0.000207   0.000151           0.000149

         ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.825840           0.825840
          2         0.157686           0.983526
          3         0.015742           0.999268
          4         0.000725           0.999993
          5         0.000007           1.000000

             ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  ---------------------
    Mean dimension             1.191372
    Effective (superposition)  2.000000         0.983526
    Effective (successive)     4.000000         0.960343
    Effective (truncation)     5.000000         0.991585  Q, K_s, Z_v, H_d, C_b

    """

    assert output in ('H', 'overflow', 'cost')

    def function(Xs, output=output):
        Q = Xs[:, 0]
        Ks = Xs[:, 1]
        Zv = Xs[:, 2]
        Zm = Xs[:, 3]
        Hd = Xs[:, 4]
        Cb = Xs[:, 5]
        L = Xs[:, 6]
        B = Xs[:, 7]

        H = (Q / (B * Ks * np.sqrt((Zm - Zv) / L))) ** (3. / 5)
        S = Zv + H - Hd - Cb

        if output == 'H':
            return H
        elif output == 'overflow':
            return S
        elif output == 'cost':
            return (S > 0) + (S <= 0) * (0.2 + 0.8 * (1 - np.exp(-1000/S**4))) + 0.05*np.minimum(Hd, 8)
        else:
            raise ValueError('Invalid output specification')

    axes = [dict(name='Q', domain=(500, 3000), dist=sp.stats.gumbel_r(1013.0, 558.0)),
            dict(name='K_s', domain=(15.0, None), dist=sp.stats.norm(loc=30, scale=8)),
            dict(name='Z_v', dist=sp.stats.triang(loc=49.0, scale=2, c=0.5)),
            dict(name='Z_m', dist=sp.stats.triang(loc=54.0, scale=2, c=0.5)),
            dict(name='H_d', dist=sp.stats.uniform(loc=7, scale=2)),
            dict(name='C_b', dist=sp.stats.triang(loc=55.0, scale=1, c=0.5)),
            dict(name='L', dist=sp.stats.triang(loc=4990.0, scale=20, c=0.5)),
            dict(name='B', dist=sp.stats.triang(loc=295.0, scale=10, c=0.5))]

    return function, axes


def get_fire_spread(wind_factor=1.0):
    """Returns the rate of spread and the reaction intensity of fire

    Args:
        :param delta: fuel depth
        :param sigma: fuel particle area-to-volume ratio
        :param h: fuel particle low heat content
        :param rho: ovendry particle density
        :param ml: moisture content of the live fuel
        :param md: moisture content of the dead fuel
        :param S: fuel particle total mineral content
        :param U: wind speed at midflame height
        :param tanphi: slope
        :param P: dead fuel loading vs. total loading

    Notes:
        Provided by: Eunhye Song <eunhyesong2016@u.northwestern.edu>

    References:
        - E. Song, B.L. Nelson, and J. Staum, 2016, Shapley effects for global
            sensitivity analysis: Theory and computation, SIAM/ASA Journal of
            Uncertainty Quantification, 4, 1060–1083.

             ** SENSITIVITY INDICES **
    Variable       Sobol     Total    Shapley    Banzhaf-Coleman
    ----------  --------  --------  ---------  -----------------
    delta       0.147682  0.365639   0.247308           0.242636
    sigma       0.009722  0.038442   0.021836           0.020716
    h           0.001795  0.005743   0.003508           0.003379
    rho_p       0.005386  0.026633   0.014262           0.013390
    m_l         0.208320  0.383794   0.289394           0.286066
    m_d         0.139058  0.286946   0.206347           0.203024
    S_T         0.001562  0.004998   0.003053           0.002940
    U           0.036260  0.104798   0.066792           0.064926
    tan_phi     0.045218  0.125940   0.081356           0.079247
    P           0.042625  0.094803   0.066142           0.064858

         ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.637628           0.637628
          2         0.294176           0.931804
          3         0.061395           0.993199
          4         0.006443           0.999642
          5         0.000347           0.999990
          6         0.000010           1.000000

             ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  -------------------------------------
    Mean dimension             1.437736
    Effective (superposition)  3.000000         0.993199
    Effective (successive)     9.000000         0.976128
    Effective (truncation)     7.000000         0.962842  delta, sigma, m_l, m_d, U, tan_phi, P

    """

    def function(Xs):
        w0 = 1 / (1 + np.exp((15 - Xs[:, 0]) / 2)) / 4.8824
        # w0 = 0.95/(1+2.43*exp((15-X[1])*0.33))**(1/2.26) /4.8824
        delta = Xs[:, 0] / 30.48
        sigma = Xs[:, 1] * 30.48
        h = Xs[:, 2] * 0.4536 / 0.25
        rho = Xs[:, 3] / 0.48824
        ml = Xs[:, 4]
        md = Xs[:, 5]
        S = Xs[:, 6]
        U = Xs[:, 7] / 0.018288
        tanphi = Xs[:, 8]
        P = Xs[:, 9]

        rmax = sigma**1.5 / (495 + 0.0594 * sigma**1.5)
        bop = 3.348 * sigma**(-0.8189)
        # A =1/(4.774*sigma**(0.1) - 7.27)
        A = 133 * sigma**(-0.7913)
        thestar = (301.4 - 305.87 * (ml - md) + 2260 * md) / (2260 * ml)
        theta = np.minimum(1, np.maximum(thestar, 0))
        muM = np.exp(-7.3 * P * md - (7.3 * theta + 2.13) * (1 - P) * ml)

        # muM = exp(-2*(P*md + (1-P)*ml))

        # muM = min(max(0,1-2.59*(md/ml)+5.11*(md/ml)**2 - 3.52*(md/ml)**3),1)
        muS = np.minimum(0.174 * S**(-0.19), 1)
        C = 7.47 * np.exp(-0.133 * sigma**0.55)
        B = 0.02526 * sigma**0.54
        E = 0.715 * np.exp(-3.59 * 10**(-4) * sigma)
        # wn = w0/(1+S)
        wn = w0 * (1 - S)
        rhob = w0 / delta
        eps = np.exp(-138 / sigma)
        Qig = (401.41 + md * 2565.87) * 0.4536 / 1.060
        # Qig = 250 + 1116*md
        beta = rhob / rho
        r = rmax * (beta / bop)**A * np.exp(A * (1 - beta / bop))
        xi = ((192 + 0.2595 * sigma)**(-1) * np.exp((0.792 + 0.681 * sigma**0.5) * (beta + 0.1)))
        phiW = C * U**B * (beta / bop)**(-E)
        phiS = 5.275 * beta**(-0.3) * tanphi**2

        # Reaction intensity
        IR = r * wn * h * muM * muS
        # Rate of spread
        R = IR * xi * (1 + phiW + phiS) / (rhob * eps * Qig)

        # c(R*30.48/60)
        results = R * 30.48 / 60
        assert not np.any(np.isnan(results))

        return results

    axes = [dict(name='delta', dist=sp.stats.lognorm(scale=np.exp(2.19), s=0.517)),
            dict(name='sigma', domain=(3.0 / 0.6, None), dist=sp.stats.lognorm(scale=np.exp(3.31), s=0.294)),
            dict(name='h', dist=sp.stats.lognorm(scale=np.exp(8.48), s=0.063)),
            dict(name='rho_p', dist=sp.stats.lognorm(scale=np.exp(-0.592), s=0.219)),
            dict(name='m_l', domain=(0, None), dist=sp.stats.norm(loc=1.18, scale=0.377)),
            dict(name='m_d', dist=sp.stats.norm(loc=0.19, scale=0.047)),
            dict(name='S_T', domain=(0, None), dist=sp.stats.norm(loc=0.049, scale=0.011)),
            dict(name='U', dist=sp.stats.lognorm(scale=np.exp(1.0174)*wind_factor, s=0.5569)),
            dict(name='tan_phi', domain=(0, None), dist=sp.stats.norm(loc=0.38, scale=0.186)),
            dict(name='P', domain=(None, 1), dist=sp.stats.lognorm(scale=np.exp(-2.19), s=0.64))]

    return function, axes


def get_robot_arm():
    """Distance of a 4-segment robot arm's end to its origin over a plane (https://www.sfu.ca/~ssurjano/robot.html)

    Note: variables are interleaved so as to reduce the TT rank

    Args:
        phi1 - phi4: angles of the four arm segments
        L1 - L4: lengths of the four arm segments

    References:
        - An, J., & Owen, A. (2001). Quasi-regression. Journal of Complexity, 17(4), 588-607.

             ** SENSITIVITY INDICES **
    Variable       Sobol      Total    Shapley    Banzhaf-Coleman
    ----------  --------  ---------  ---------  -----------------
    phi1        0.000000   0.000000   0.000000           0.000000
    L1          0.050800   0.149860   0.084483           0.076672
    phi2        0.072274   0.401519   0.196556           0.176517
    L2          0.049909   0.148474   0.089201           0.084242
    phi3        0.077015   0.536635   0.256871           0.232026
    L3          0.050284   0.148362   0.089475           0.084586
    phi4        0.073504   0.405552   0.199067           0.178966
    L4          0.050026   0.150884   0.084347           0.076403

         ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.423812           0.423812
          2         0.298511           0.722323
          3         0.200763           0.923086
          4         0.066868           0.989954
          5         0.009598           0.999552
          6         0.000433           0.999985
          7         0.000015           1.000000

             ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  --------------------------------
    Mean dimension             1.941287
    Effective (superposition)  4.000000         0.989954
    Effective (successive)     6.000000         0.988964
    Effective (truncation)     7.000000         1.000000  L1, phi2, L2, phi3, L3, phi4, L4

    """

    def function(Xs):
        u = np.sum(Xs[:, [1, 3, 5, 7]] * np.cos(np.cumsum(Xs[:, [0, 2, 4, 6]], axis=1)), axis=1)
        v = np.sum(Xs[:, [1, 3, 5, 7]] * np.sin(np.cumsum(Xs[:, [0, 2, 4, 6]], axis=1)), axis=1)
        return np.sqrt(u**2 + v**2)

    axes = [dict(name='phi1', domain=(0, 2*np.pi)),
            dict(name='L1', domain=(0, 1)),
            dict(name='phi2', domain=(0, 2*np.pi)),
            dict(name='L2', domain=(0, 1)),
            dict(name='phi3', domain=(0, 2*np.pi)),
            dict(name='L3', domain=(0, 1)),
            dict(name='phi4', domain=(0, 2*np.pi)),
            dict(name='L4', domain=(0, 1))]

    return function, axes


def get_wing_weight():
    """Weight of a wing (https://www.sfu.ca/~ssurjano/wingweight.html)

    References:
        - Forrester, A., Sobester, A., & Keane, A. (2008). Engineering design via
            surrogate modelling: a practical guide. Wiley.
        - Moon, H. (2010). Design and Analysis of Computer Experiments for
            Screening Input Variables (Doctoral dissertation, Ohio State University).
        - Moon, H., Dean, A. M., & Santner, T. J. (2012). Two-stage sensitivity-based
            group screening in computer experiments. Technometrics, 54(4), 376-387.

            ** SENSITIVITY INDICES **
    Variable       Sobol     Total    Shapley    Banzhaf-Coleman
    ----------  --------  --------  ---------  -----------------
    Sw          0.124474  0.127901   0.126182           0.126179
    Wfw         0.000003  0.000003   0.000003           0.000003
    A           0.220242  0.226013   0.223120           0.223116
    Lambda      0.000490  0.000506   0.000498           0.000498
    q           0.000088  0.000091   0.000090           0.000090
    lambda      0.001810  0.001871   0.001840           0.001840
    tc          0.140980  0.145068   0.143017           0.143014
    Nz          0.411610  0.419647   0.415619           0.415615
    Wdg         0.084968  0.087601   0.086280           0.086278
    Wp          0.003339  0.003362   0.003351           0.003351

            ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.988004           0.988004
          2         0.011928           0.999932
          3         0.000068           1.000000

             ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  ------------------
    Mean dimension             1.012063
    Effective (superposition)  1.000000         0.988004
    Effective (successive)     1.000000         0.988004
    Effective (truncation)     5.000000         0.994166  Sw, A, tc, Nz, Wdg

    """

    def function(Xs):
        Xs[:, 3] *= 2*np.pi/360  # Original bounds for 'Lambda" are in degrees
        return 0.036 * Xs[:, 0]**0.758 * Xs[:, 1]**0.0035 * (Xs[:, 2] / np.cos(Xs[:, 3])**2)**0.6 * Xs[:, 4]**0.006 * Xs[:, 5]**0.04 * (100*Xs[:, 6] / np.cos(Xs[:, 3]))**-0.3 * (Xs[:, 7]*Xs[:, 8])**0.49 + Xs[:, 0]*Xs[:, 9]

    axes = [dict(name='Sw', domain=(150, 200)),
            dict(name='Wfw', domain=(220, 300)),
            dict(name='A', domain=(6, 10)),
            dict(name='Lambda', domain=(-10, 10)),
            dict(name='q', domain=(16, 45)),
            dict(name='lambda', domain=(0.5, 1)),
            dict(name='tc', domain=(0.08, 0.18)),
            dict(name='Nz', domain=(2.5, 6)),
            dict(name='Wdg', domain=(1700, 2500)),
            dict(name='Wp', domain=(0.025, 0.08))]

    return function, axes


def get_otl_circuit():
    """Midpoint voltage of a push-pull circuit (https://www.sfu.ca/~ssurjano/otlcircuit.html)
    
    References:
        - Ben-Ari, E. N., & Steinberg, D. M. (2007). Modeling data from computer
            experiments: an empirical comparison of kriging with MARS and projection
            pursuit regression. Quality Engineering, 19(4), 327-338.
        - Moon, H. (2010). Design and Analysis of Computer Experiments for
            Screening Input Variables (Doctoral dissertation, Ohio State University).
        - Moon, H., Dean, A. M., & Santner, T. J. (2012). Two-stage sensitivity-based
            group screening in computer experiments. Technometrics, 54(4), 376-387.


            ** SENSITIVITY INDICES **
    Variable       Sobol     Total    Shapley    Banzhaf-Coleman
    ----------  --------  --------  ---------  -----------------
    Rb1         0.495655  0.500136   0.497895           0.497895
    Rb2         0.407205  0.411687   0.409446           0.409446
    Rf          0.070842  0.074007   0.072425           0.072425
    Rc1         0.018639  0.021802   0.020220           0.020220
    Rc2         0.000000  0.000000   0.000000           0.000000
    beta        0.000013  0.000015   0.000014           0.000014

            ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.992354           0.992354
          2         0.007646           1.000000

             ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  ------------
    Mean dimension             1.007647
    Effective (superposition)  1.000000         0.992354
    Effective (successive)     1.000000         0.992354
    Effective (truncation)     3.000000         0.978183  Rb1, Rb2, Rf

    """

    def function(Xs):
        Vb1 = 12 * Xs[:, 1] / (Xs[:, 0] + Xs[:, 1])
        part1 = (Vb1 + 0.74) * Xs[:, 5] * (Xs[:, 4] + 9) / (Xs[:, 5]*(Xs[:, 4]+9) + Xs[:, 2])
        part2 = 11.35 * Xs[:, 2] / (Xs[:, 5]*(Xs[:, 4]+9)+Xs[:, 2])
        part3 = 0.74 * Xs[:, 2] * Xs[:, 5] * (Xs[:, 4] + 9) / ((Xs[:, 5] * (Xs[:, 4] + 9) + Xs[:, 2]) * Xs[:, 3])
        return part1 + part2 + part3

    axes = [dict(name='Rb1', domain=(50, 150)),
            dict(name='Rb2', domain=(25, 70)),
            dict(name='Rf', domain=(0.5, 3)),
            dict(name='Rc1', domain=(1.2, 2.5)),
            dict(name='Rc2', domain=(0.25, 1.2)),
            dict(name='beta', domain=(50, 300))]

    return function, axes


def get_welch_1992():
    """High-dimensional function designed for screening purposes (https://www.sfu.ca/~ssurjano/welchetal92.html)

    References:
        - Ben-Ari, E. N., & Steinberg, D. M. (2007). Modeling data from computer
            experiments: an empirical comparison of kriging with MARS and
            projection pursuit regression. Quality Engineering, 19(4), 327-338.
        - Welch, W. J., Buck, R. J., Sacks, J., Wynn, H. P., Mitchell, T. J., & Morris, M. D. (1992).
            Screening, predicting, and computer experiments. Technometrics, 34(1), 15-25.

             ** SENSITIVITY INDICES **
    Variable       Sobol     Total    Shapley    Banzhaf-Coleman
    ----------  --------  --------  ---------  -----------------
    x_0         0.000000  0.005734   0.002867           0.002867
    x_1         0.000113  0.000113   0.000113           0.000113
    x_2         0.000000  0.000000   0.000000           0.000000
    x_3         0.075599  0.453707   0.264653           0.264653
    x_4         0.052928  0.052928   0.052928           0.052928
    x_5         0.000041  0.000041   0.000041           0.000041
    x_6         0.000041  0.000041   0.000041           0.000041
    x_7         0.000000  0.000000   0.000000           0.000000
    x_8         0.000368  0.000368   0.000368           0.000368
    x_9         0.000005  0.000005   0.000005           0.000005
    x_10        0.000222  0.000222   0.000222           0.000222
    x_11        0.054767  0.060500   0.057634           0.057634
    x_12        0.000189  0.000189   0.000189           0.000189
    x_13        0.000073  0.000073   0.000073           0.000073
    x_14        0.000163  0.000163   0.000163           0.000163
    x_15        0.000000  0.000000   0.000000           0.000000
    x_16        0.000005  0.000005   0.000005           0.000005
    x_17        0.000041  0.000041   0.000041           0.000041
    x_18        0.356006  0.356006   0.356006           0.356006
    x_19        0.075599  0.453707   0.264653           0.264653

            ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.616159           0.616159
          2         0.383841           1.000000

             ** DIMENSION METRICS **
    Dimension Metric               Value    Rel. Variance  Variables
    -------------------------  ---------  ---------------  --------------------------
    Mean dimension              1.383841
    Effective (superposition)   2.000000         1.000000
    Effective (successive)     17.000000         1.000000
    Effective (truncation)      5.000000         0.993007  x_3, x_4, x_11, x_18, x_19

    """

    def function(Xs):
        return Xs[:, 11] / (1 + Xs[:, 0]) + 5*(Xs[:, 3] - Xs[:, 19])**2 + Xs[:, 4] + 40*Xs[:, 18]**3 - 5*Xs[:, 18] + 0.05*Xs[:, 1] + 0.08*Xs[:, 4] - 0.03*Xs[:, 5] + 0.03*Xs[:, 6] - 0.09*Xs[:, 8] - 0.01*Xs[:, 9] - 0.07*Xs[:, 10] + 0.25*Xs[:, 12]**2 - 0.04*Xs[:, 13] + 0.06*Xs[:, 14] - 0.01*Xs[:, 16] - 0.03*Xs[:, 17]

    axes = [dict(domain=(-0.5, 0.5)) for n in range(20)]

    return function, axes


def get_dette_pepelyshev():
    """Highly curved synthetic function (https://www.sfu.ca/~ssurjano/detpep10curv.html)

    References:
        - Dette, H., & Pepelyshev, A. (2010). Generalized Latin hypercube
            design for computer experiments. Technometrics, 52(4).

            ** SENSITIVITY INDICES **
    Variable       Sobol     Total    Shapley    Banzhaf-Coleman
    ----------  --------  --------  ---------  -----------------
    x_0         0.004541  0.040873   0.022707           0.022707
    x_1         0.291170  0.327502   0.309336           0.309336
    x_2         0.667958  0.667958   0.667958           0.667958

            ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.963668           0.963668
          2         0.036332           1.000000

            ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  -----------
    Mean dimension             1.036332
    Effective (superposition)  1.000000         0.963668
    Effective (successive)     1.000000         0.963668
    Effective (truncation)     2.000000         0.959127  x_1, x_2

    """

    def function(Xs):
        return 4*(Xs[:, 0] - 2 + 8*Xs[:, 1] - 8*Xs[:, 1]**2)**2 + (3 - 4*Xs[:, 1])**2 + 16*np.sqrt(Xs[:, 2] + 1)*(2*Xs[:, 2] - 1)**2

    axes = [dict(domain=(0, 1)) for n in range(3)]

    return function, axes


def get_environmental_model():
    """6D model (4 input variables, output is 2D) modeling the concentration of a pollutant after an accidental spill (https://www.sfu.ca/~ssurjano/environ.html)

    For the output dimensions, we use bounds (0, 3) (spatial axis) and (0.3, 60) (time axis)

    References:
        - Bliznyuk, N., Ruppert, D., Shoemaker, C., Regis, R., Wild, S., & Mugunthan, P. (2008).
            Bayesian calibration and uncertainty analysis for computationally
            expensive models using optimization and radial basis function approximation.
            Journal of Computational and Graphical Statistics, 17(2).


            ** SENSITIVITY INDICES **
    Variable       Sobol     Total    Shapley    Banzhaf-Coleman
    ----------  --------  --------  ---------  -----------------
    M           0.050135  0.077798   0.060672           0.059034
    D           0.012723  0.084168   0.038813           0.034006
    L           0.004704  0.355360   0.131566           0.107343
    tau         0.000001  0.002237   0.000558           0.000280
    s           0.124481  0.641080   0.331417           0.305744
    t           0.267264  0.705564   0.436973           0.412262

            ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.459308           0.459308
          2         0.247523           0.706831
          3         0.261584           0.968416
          4         0.030825           0.999241
          5         0.000757           0.999998
          6         0.000002           1.000000

             ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  -------------
    Mean dimension             1.866206
    Effective (superposition)  3.000000         0.968416
    Effective (successive)     5.000000         0.979452
    Effective (truncation)     5.000000         0.997763  M, D, L, s, t

    """

    def function(Xs):
        result = Xs[:, 0] / np.sqrt(4*np.pi*Xs[:, 1]*Xs[:, 5]) * np.exp(-(Xs[:, 4]**2) / (4*Xs[:, 1]*Xs[:, 5]))
        idx = (Xs[:, 5] > Xs[:, 3])
        result[idx] += Xs[idx, 0] / np.sqrt(4*np.pi*Xs[idx, 1]*(Xs[idx, 5]-Xs[idx, 3])) * np.exp(-((Xs[idx, 4]-Xs[idx, 2])**2) / (4*Xs[idx, 1]*(Xs[idx, 5]-Xs[idx, 3])))
        return np.sqrt(4*np.pi) * result

    axes = [dict(name='M', domain=(7, 13)),
            dict(name='D', domain=(0.02, 0.12)),
            dict(name='L', domain=(0.01, 3)),
            dict(name='tau', domain=(30.01, 30.295)),
            dict(name='s', domain=(0, 3)),
            dict(name='t', domain=(0.3, 60))]

    return function, axes


def get_hilbert(N, size):
    """
    The Hilbert tensor, defined as 1 / (x_1 + ... x_N). It is known to have high rank, but also to be extremely well approximated by low-rank tensors
    """

    def function(Xs):
        return 1./np.sum(Xs, axis=1)

    axes = [dict(domain=(1, size)) for n in range(N)]

    return function, axes


def lognormal_with_given_moments(mean, variance):
    return sp.stats.lognorm(scale=mean**2 / np.sqrt(variance + mean**2), s=np.sqrt(np.log(variance / mean**2 + 1)))


def get_simple_beam_deflection():
    """
    A rank-1 function modeling the maximum deflection at the middle of a simply supported beam, as a function of the beam width b, its height h, its span L, Young's modulus E, and the uniform load p

    Reference:
        - K. Konakli, B. Sudret. "Low-Rank Tensor Approximations for Reliability Analysis" (2011), https://hal.archives-ouvertes.fr/hal-01169564/document

            ** SENSITIVITY INDICES **
    Variable       Sobol     Total    Shapley    Banzhaf-Coleman
    ----------  --------  --------  ---------  -----------------
    b           0.027403  0.029824   0.028602           0.028597
    h           0.249043  0.265700   0.257325           0.257302
    L           0.009869  0.010758   0.010310           0.010307
    E           0.246565  0.263114   0.254794           0.254770
    p           0.438237  0.459799   0.448969           0.448945

            ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.971117           0.971117
          2         0.028571           0.999688
          3         0.000311           0.999999
          4         0.000001           1.000000

            ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  -----------
    Mean dimension             1.029196
    Effective (superposition)  1.000000         0.971117
    Effective (successive)     1.000000         0.971117
    Effective (truncation)     3.000000         0.959444  h, E, p

    """

    def function(Xs):
        return Xs[:, 4] * Xs[:, 2]**3 / (4 * Xs[:, 3] * Xs[:, 0] * Xs[:, 1]**3)

    axes = [dict(name='b', dist=lognormal_with_given_moments(0.15, 0.0075**2)),
            dict(name='h', dist=lognormal_with_given_moments(0.3, 0.015**2)),
            dict(name='L', dist=lognormal_with_given_moments(5, 0.05**2)),
            dict(name='E', dist=lognormal_with_given_moments(30000, 4500**2)),
            dict(name='p', dist=lognormal_with_given_moments(10000, 2000**2))]

    return function, axes


def get_damped_oscillator(p=3):
    """
    Dubourg's oscillator function for 2 masses; it has 8 independent lognormal variables as inputs plus the peak
    factor p. The output is the peak force in the secondary spring

    References:

    - "Metamodel-based importance sampling for structural reliability analysis", by Dubourg et al. (2013)
    - "Uncertainty Propagation of P-Boxes Using Sparse Polynomial Chaos Expansions", by Schoebi and Sudret (2016)

            ** SENSITIVITY INDICES **
    Variable       Sobol     Total    Shapley    Banzhaf-Coleman
    ----------  --------  --------  ---------  -----------------
    m_p         0.041272  0.090945   0.058892           0.055303
    m_s         0.106763  0.162462   0.127068           0.123314
    k_p         0.215040  0.334491   0.263963           0.258581
    k_s         0.146806  0.261877   0.193555           0.188181
    z_p         0.056309  0.083121   0.067595           0.066553
    z_s         0.047952  0.064552   0.055676           0.055392
    S_0         0.006767  0.008665   0.007640           0.007603
    F_s         0.225617  0.225590   0.225611           0.225614

            ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.846526           0.846526
          2         0.091934           0.938460
          3         0.046414           0.984874
          4         0.013584           0.998458
          5         0.001521           0.999979
          6         0.000021           1.000000

            ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  ---------------------------------
    Mean dimension             1.231703
    Effective (superposition)  3.000000         0.984874
    Effective (successive)     4.000000         0.988875
    Effective (truncation)     7.000000         0.991335  m_p, m_s, k_p, k_s, z_p, z_s, F_s

    """

    def function(Xs):

        # Input variables
        m_p = Xs[:, 0]  # Mass 1
        m_s = Xs[:, 1]  # Mass 2
        k_p = Xs[:, 2]  # Stiffness (spring 1)
        k_s = Xs[:, 3]  # Stiffness (spring 2)
        zeta_p = Xs[:, 4]  # Damping ratio (damper 1)
        zeta_s = Xs[:, 5]  # Damping ratio (damper 2)
        S_0 = Xs[:, 6]  # White noise excitation
        F_s = Xs[:, 7]  # Force capacity (spring 2)

        # Intermediate variables
        omega_p = np.sqrt(k_p / m_p)  # Natural frequency (spring 1)
        omega_s = np.sqrt(k_s / m_s)  # Natural frequency (spring 2)
        gamma = m_s / m_p  # Relative mass
        omega_a = (omega_p+omega_s) / 2  # Average natural frequency
        zeta_a = (zeta_p+zeta_s) / 2  # Average damping ratio
        theta = (omega_p-omega_s) / omega_a  # Tuning parameter
        msrd = np.pi*S_0 / (4*zeta_s*omega_s**3) * zeta_a * zeta_s / (zeta_p * zeta_s * (4*zeta_a**2+theta**2) + gamma * zeta_a**2) * (zeta_p * omega_p**3 + zeta_s * omega_s**3) * omega_p / (4 * zeta_a * omega_a**4)  # Mean-square relative displacement

        return F_s - p*k_s*np.sqrt(msrd)  # Peak force in the secondary spring

    axes = [dict(name='m_p', dist=lognormal_with_given_moments(1.5, 0.15**2)),
            dict(name='m_s', dist=lognormal_with_given_moments(0.01, 0.001**2)),
            dict(name='k_p', dist=lognormal_with_given_moments(1, 0.2**2)),
            dict(name='k_s', dist=lognormal_with_given_moments(0.01, 0.002**2)),
            dict(name='z_p', dist=lognormal_with_given_moments(0.05, 0.02**2)),
            dict(name='z_s', dist=lognormal_with_given_moments(0.02, 0.01**2)),
            dict(name='S_0', dist=lognormal_with_given_moments(100, 10**2)),
            dict(name='F_s', dist=lognormal_with_given_moments(15, 1.5**2))]

    return function, axes


def get_cantilever_beam(output='displacement'):
    """
    A beam receives both a vertical and horizontal load on its extreme. We measure its mass/stress/displacement
    Reference: Dakota Sensitivity Analysis and Uncertainty Quantification, with Examples (Sandia National Labs, https://dakota.sandia.gov/sites/default/files/docs/training/201508/DakotaTraining_SensitivityAnalysis.pdf)

            ** SENSITIVITY INDICES **
    Variable       Sobol      Total    Shapley    Banzhaf-Coleman
    ----------  --------  ---------  ---------  -----------------
    L           0.407853   0.468404   0.437815           0.437658
    w           0.099618   0.117881   0.108608           0.108537
    t           0.330980   0.383948   0.357171           0.357024
    rho         0.000000   0.000000   0.000000           0.000000
    E           0.052904   0.066693   0.059642           0.059563
    X           0.002111   0.003077   0.002579           0.002572
    Y           0.029570   0.039058   0.034185           0.034121

            ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.923036           0.923036
          2         0.074892           0.997928
          3         0.002048           0.999976
          4         0.000024           1.000000

            ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  -----------
    Mean dimension             1.079060
    Effective (superposition)  2.000000         0.997928
    Effective (successive)     3.000000         0.981763
    Effective (truncation)     4.000000         0.957885  L, w, t, E

    """

    assert output in ('mass', 'stress', 'displacement')

    def function(Xs):

        L = Xs[:, 0]  # Beam length
        w = Xs[:, 1]  # Width
        t = Xs[:, 2]  # Thickness
        rho = Xs[:, 3]  # Density
        E = Xs[:, 4]  # Young's modulus
        X = Xs[:, 5]  # Horizontal load
        Y = Xs[:, 6]  # Vertical load

        if output == 'mass':
            return rho*w*t*L / (12**3)
        elif output == 'stress':
            return 600/(w*t**2)*Y + 600/(w**2*t)*X
        else:
            return 4*L**3/(E*w*t) * np.sqrt((Y / t**2)**2 + (X / w**2)**2)

    axes = [dict(name='L', domain=(4, 6)),
            dict(name='w', domain=(0.8, 1.2)),
            dict(name='t', domain=(0.8, 1.2)),
            dict(name='rho', domain=(400, 600)),
            dict(name='E', domain=(23e6, 35e6)),
            dict(name='X', domain=(40, 60)),
            dict(name='Y', domain=(80, 120))]

    return function, axes


def get_ebola_spread(country='Liberia'):
    """
    8-dimensional model that predicts the Ebola virus reproduction number, i.e. the average number of secondary
    infections that are expected in an outbreak of the disease. The inputs' ranges depend on the country (Liberia or
    Sierra Leone).

    References:

    - P. Diaz, P. Constantine, K. Kalmbach, E. Jones, and S. Pankavich: "A Modified SEIR Model for the Spread of Ebola in Western Africa and Metrics for Resource Allocation" (2016)
    - P.G. Constantine, E. Dow, and Q. Wang: "Active subspaces in theory and practice: applications to kriging
    surfaces" (2014)

            ** SENSITIVITY INDICES **
    Variable       Sobol     Total    Shapley    Banzhaf-Coleman
    ----------  --------  --------  ---------  -----------------
    beta_1      0.185387  0.229694   0.207100           0.206880
    beta_2      0.006249  0.008929   0.007544           0.007522
    beta_3      0.252781  0.279617   0.266080           0.266021
    rho_1       0.003039  0.004596   0.003785           0.003768
    gamma_1     0.164199  0.188713   0.176349           0.176296
    gamma_1     0.164199  0.188713   0.176349           0.176296
    omega       0.002118  0.003261   0.002663           0.002650
    phi         0.232172  0.311165   0.271076           0.270779

            ** DIMENSION DISTRIBUTION **
      Order    Rel. Variance    Cumul. Variance
    -------  ---------------  -----------------
          1         0.892648           0.892648
          2         0.103536           0.996184
          3         0.003793           0.999977
          4         0.000023           1.000000

            ** DIMENSION METRICS **
    Dimension Metric              Value    Rel. Variance  Variables
    -------------------------  --------  ---------------  -------------------------------------
    Mean dimension             1.111192
    Effective (superposition)  2.000000         0.996184
    Effective (successive)     5.000000         0.952108
    Effective (truncation)     5.000000         0.984219  beta_1, beta_3, gamma_1, gamma_1, phi

    """

    assert country in ('Liberia', 'Sierra Leone')

    def function(Xs):

        # Parameter interpretation: https://arxiv.org/pdf/1603.04955.pdf, table 1

        beta_1 = Xs[:, 0]
        beta_2 = Xs[:, 1]
        beta_3 = Xs[:, 2]
        rho_1 = Xs[:, 3]
        gamma_1 = Xs[:, 4]
        gamma_2 = Xs[:, 5]
        omega = Xs[:, 6]
        phi = Xs[:, 7]

        R_0 = (beta_1 + (beta_2 * rho_1 * gamma_1)/omega + beta_3/gamma_2*phi) / (gamma_1 + phi)  # Reproduction number

        return R_0

    # Axes: from https://arxiv.org/pdf/1603.04955.pdf, table 7

    if country == 'Liberia':
        axes = [dict(name='beta_1', domain=(0.1, 0.4)),
                dict(name='beta_2', domain=(0.1, 0.4)),
                dict(name='beta_3', domain=(0.05, 0.2)),
                dict(name='rho_1', domain=(0.41, 1)),
                dict(name='gamma_1', domain=(0.0276, 0.1702)),
                dict(name='gamma_1', domain=(0.081, 0.21)),
                dict(name='omega', domain=(0.25, 0.5)),
                dict(name='phi', domain=(0.0833, 0.7))]
    else:
        axes = [dict(name='beta_1', domain=(0.1, 0.4)),
                dict(name='beta_2', domain=(0.1, 0.4)),
                dict(name='beta_3', domain=(0.05, 0.2)),
                dict(name='rho_1', domain=(0.41, 1)),
                dict(name='gamma_1', domain=(0.0275, 0.1569)),
                dict(name='gamma_1', domain=(0.1236, 0.384)),
                dict(name='omega', domain=(0.25, 0.5)),
                dict(name='phi', domain=(0.0833, 0.7))]

    return function, axes
