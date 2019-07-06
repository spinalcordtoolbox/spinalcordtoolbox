#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with 2d and 3d curve fitting
# TODO: implement robust fitting, i.e., detection and removal of outliers. See:
#  https://github.com/neuropoly/spinalcordtoolbox/blob/24ec6668d623be00194b21038f275134c82122de/scripts/msct_smooth.py#L568

from __future__ import absolute_import

import logging
import numpy as np

logger = logging.getLogger(__name__)


def polyfit_1d(x, y, xref, deg=5):
    """
    1d Polynomial fitting using np.polyfit
    :param x:
    :param y:
    :param deg:
    :param xref: np.array: vector of abscissa on which to project the fitted curve. Example: np.linspace(0, 50, 51)
    :return: p(xref): Fitted polynomial for each xref point
    :return: p.deriv(xref): Derivatives for each xref point
    """
    from numpy import poly1d, polyfit
    p = poly1d(polyfit(x, y, deg=deg))
    return p(xref), p.deriv(1)(xref)


def bspline(x, y, xref, smooth, deg_bspline=3, pz=1):
    """
    Bspline interpolation.

    The smoothing factor (s) is calculated based on an empirical formula (made by JCA, based on
    preliminary results) and is a function of pz, density of points and an input smoothing parameter (smooth). The
    formula is adjusted such that the parameter (smooth) produces similar smoothing results than a Hanning window with
    length smooth, as implemented in linear().

    :param x:
    :param y:
    :param xref:
    :param smooth: float: Smoothing factor. 0: no smoothing, 5: moderate smoothing, 50: large smoothing
    :param deg_bspline: int: Degree of spline
    :param pz: float: dimension of pixel along superior-inferior direction (z, assuming RPI orientation)
    :return:
    """
    # TODO: add flag to enforce boundaries, using weight flag in bspline function
    from scipy import interpolate
    if len(x) <= deg_bspline:
        deg_bspline -= 2
    density = (float(len(x)) / len(xref)) ** 2
    s = density * smooth * pz / float(3)
    logger.debug('Smoothing factor: smooth={}'.format(s))
    # Then, run bspline interpolation
    tck = interpolate.splrep(x, y, s=s, k=deg_bspline)
    y_fit = interpolate.splev(xref, tck, der=0)
    y_fit_der = interpolate.splev(xref, tck, der=1)
    return y_fit, y_fit_der


def linear(x, y, xref, smooth=0, pz=1):
    """
    Linear interpolation followed by smoothing.

    :param x:
    :param y:
    :param xref:
    :param smooth: float: Smoothing factor corresponding to the length of the filter window (in mm). 0: no smoothing.
    :param pz: float: dimension of pixel along superior-inferior direction (z, assuming RPI orientation)
    :return:
    """
    from numpy import interp
    y_fit = interp(xref, x, y, left=None, right=None, period=None)
    window_len = round_up_to_odd(smooth / float(pz))
    logger.debug('Smoothing window: {}'.format(window_len))
    y_fit = smooth1d(y_fit, window_len)
    y_fit_der = np.gradient(y_fit)
    return y_fit, y_fit_der


def round_up_to_odd(f):
    """Round input float to next odd integer."""
    return int(np.ceil(f) // 2 * 2 + 1)


def smooth1d(x, window_len, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies (central symmetry) of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    Modified from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    """

    if x.ndim != 1:
        raise ValueError("Smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        window_len = x.size
        logger.warning("Input vector is smaller than window size. Forcing window_len = x.size")

    if window_len < 3:
        logger.warning("Window length needs to be >= 3. Returning input signal.")
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # Pad signal using central symmetry
    s = np.r_[2 * x[0] - x[window_len - 1:0:-1], x, 2 * x[-1] - x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(np.ceil(window_len/2-1)):-int(np.floor(window_len/2))]
