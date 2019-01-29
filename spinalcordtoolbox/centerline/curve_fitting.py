#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with 2d and 3d curve fitting

from __future__ import absolute_import


def polyfit_1d(x, y, xref, deg=3):
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
    from numpy import where
    from scipy import interpolate
    # First, take the center of mass to avoid issue (https://stackoverflow.com/questions/2009379/interpolate-question)
    x_mean, y_mean = [], []
    # Loop across min/max indices
    for ix in range(x.min(), x.max()):
        # Get indices corresponding to ix
        ind_x = where(x == ix)
        if ind_x:
            # Average all y values at ind_x
            x_mean.append(ix)
            y_mean.append(y[ind_x].mean())
    # Then, run bspline interpolation
    tck = interpolate.splrep(x_mean, y_mean, s=5, k=deg)  # TODO: find s based on pix dim
    y_fit = interpolate.splev(xref, tck, der=0)
    y_fit_der = interpolate.splev(xref, tck, der=1)
    return y_fit, y_fit_der


def sinc_interp(y, x, xref):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")

    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
    """
    import numpy as np

    if len(x) != len(y):
        raise Exception, 'x and s must be the same length'
    # Find the period
    t = y[1] - y[0]
    sinc_m = np.tile(xref, (len(y), 1)) - np.tile(y[:, np.newaxis], (1, len(xref)))
    # TODO: return derivatives
    return np.dot(x, np.sinc(sinc_m / t)), 0