#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with 2d and 3d curve fitting

from __future__ import absolute_import

import sct_utils as sct


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
    from numpy import poly1d, polyfit, polyder
    p = poly1d(polyfit(x, y, deg=deg))
    return p(xref), p.deriv(1)(xref)


def bspline(x, y, xref, deg=3):
    """
    Bspline interpolation. Length of x needs to be superior to deg.
    :param x:
    :param y:
    :param xref:
    :param deg:
    :return:
    """
    from scipy import interpolate
    # Make sure the condition len(x_mean) > k is satisfied. Otherwise, change k to avoid crashing.
    if not len(x) > deg:
        deg = len(x) - 1
        sct.log.warning('Input array size is smaller than degree. Forcing deg = ' + str(deg))
    # Then, run bspline interpolation
    tck = interpolate.splrep(x, y, s=5, k=deg)  # TODO: find s based on pix dim
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
    from numpy import interp
    return interp(xref, x, y, left=None, right=None, period=None)
