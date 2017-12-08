# -*- coding: utf-8 -*-
"""Sample functions for surrogate modeling and sensitivity analysis.

This module contains different examples of multidimensional functions that may
work as sensible examples for real-world models.

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
import scipy as sp
import scipy.stats


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


def get_prod(N, constant=1, name_tmpl='x_{}'):
    """Analytical test function used for sensitivity analysis.

    References:
        - Kucherenko, S., Feil, B., Shah, N., & Mauntz, W. (2011). "The identification
            of model effective dimensions using global sensitivity analysis"
            Reliability Engineering & System Safety, 96(4), 440-449.

    """

    assert N > 0

    def function(Xs, constant=constant):

        Xs = np.asarray(Xs)
        assert(Xs.shape[1] == N)
        return np.prod(np.abs(constant * Xs - 2), axis=1)

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


def get_simple_beam_deflection():
    """
    A rank-1 function modeling the maximum deflection at the middle of a simply supported beam, as a function of the beam width b, its height h, its span L, Young's modulus E, and the uniform load p

    Reference:
        - K. Konakli, B. Sudret. "Low-Rank Tensor Approximations for Reliability Analysis" (2011), https://hal.archives-ouvertes.fr/hal-01169564/document
    """

    def function(Xs):
        return Xs[:, 4] * Xs[:, 2]**3 / (4 * Xs[:, 3] * Xs[:, 0] * Xs[:, 1]**3)

    def lognormal_with_given_moments(mean, variance):
        return sp.stats.lognorm(scale=mean**2 / np.sqrt(variance + mean**2), s=np.sqrt(np.log(variance / mean**2 + 1)))

    axes = [dict(name='b', dist=lognormal_with_given_moments(0.15, 0.0075**2)),
            dict(name='h', dist=lognormal_with_given_moments(0.3, 0.015**2)),
            dict(name='L', dist=lognormal_with_given_moments(5, 0.05**2)),
            dict(name='E', dist=lognormal_with_given_moments(30000, 4500**2)),
            dict(name='p', dist=lognormal_with_given_moments(10000, 2000**2))]

    return function, axes
