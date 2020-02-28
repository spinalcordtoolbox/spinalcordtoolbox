#!/usr/bin/env python
# -*- coding: utf-8
# Functions that perform mathematical operations on an image.

from __future__ import absolute_import

import logging
import numpy as np

from skimage.morphology import erosion, ball


logger = logging.getLogger(__name__)


def dilate(data, radius):
    """
    Dilate data using ball structuring element
    :param data: 2d or 3d array
    :param radius: radius of structuring element OR comma-separated int.
    :return: data dilated
    """
    from skimage.morphology import dilation, ball
    if len(radius) == 1:
        # define structured element as a ball
        selem = ball(radius[0])
    else:
        # define structured element as a box with input dimensions
        selem = np.ones((radius[0], radius[1], radius[2]), dtype=np.dtype)
    return dilation(data, selem=selem, out=None)


def erode(data, radius):
    """
    Erode data using ball structuring element
    :param data: 2d or 3d array
    :param radius: radius of structuring element
    :return: data eroded
    """
    if len(radius) == 1:
        # define structured element as a ball
        selem = ball(radius[0])
    else:
        # define structured element as a box with input dimensions
        selem = np.ones((radius[0], radius[1], radius[2]), dtype=np.dtype)
    return erosion(data, selem=selem, out=None)