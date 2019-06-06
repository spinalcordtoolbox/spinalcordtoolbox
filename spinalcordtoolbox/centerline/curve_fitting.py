#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with 2d and 3d curve fitting

from __future__ import absolute_import

import logging

logger = logging.getLogger(__name__)


def polyfit_1d(x, y, xref, deg=5):
    """
    1d Polynomial fitting using np.polyfit
    :param x:
    :param y:
    :param deg:
    :param xref: np.array: vector of abscissa on which to project the fitted curve. Example: np.linspace(0, 50, 51)
    :return: p(xref): Fitted polynomial for each xref point
    :return: p_diff(xref): Derivatives for each xref point
    """
    from numpy import poly1d, polyfit
    p = poly1d(polyfit(x, y, deg=deg))
    return p(xref), p.deriv(1)(xref)


def bspline(x, y, xref, smooth, deg_bspline=3):
    """
    Bspline interpolation. Length of x needs to be superior to deg.
    :param x:
    :param y:
    :param xref:
    :param smooth: float: Smoothing factor. 0: no smoothing, 5: moderate smoothing, 50: large smoothing
    :param deg_bspline: int: Degree of spline
    :return:
    """
    # TODO: add flag to enforce boundaries, using weight flag in bspline function
    from scipy import interpolate
    if len(x) <= deg_bspline:
        deg_bspline -= 2
    # Compute smoothing factor
    # s = (len(x) - sqrt(2 * len(x))) * smooth
    s = smooth  # TODO: adjust with pz
    logger.debug('Smoothing factor: smooth={}'.format(s))
    # Then, run bspline interpolation
    tck = interpolate.splrep(x, y, s=s, k=deg_bspline)
    y_fit = interpolate.splev(xref, tck, der=0)
    y_fit_der = interpolate.splev(xref, tck, der=1)
    return y_fit, y_fit_der


def linear(x, y, xref):
    """
    Linear interpolation.
    :param x:
    :param y:
    :param xref:
    :return:
    """
    # TODO: implement smoothing
    from numpy import interp
    return interp(xref, x, y, left=None, right=None, period=None)
