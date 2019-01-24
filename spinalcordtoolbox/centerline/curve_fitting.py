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
    # Debug
    # import matplotlib.pyplot as plt
    # _ = plt.plot(x, y, '.', z, p(z), '-')
    # plt.show()
    return p(xref), polyder(p)

