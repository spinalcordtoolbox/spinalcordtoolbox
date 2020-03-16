#!/usr/bin/env python
# -*- coding: utf-8
# Functions that perform mathematical operations on an image.

from __future__ import absolute_import

import logging
import numpy as np

from skimage.morphology import erosion, dilation, disk, ball, square, cube


logger = logging.getLogger(__name__)


def dilate(data, size, shape, dim=None):
    """
    Dilate data using ball structuring element
    :param data: numpy array: 2d or 3d array
    :param size: int: If shape={'square', 'cube'}: Corresponds to the length of an edge (size=1 has no effect).
    If shape={'disk', 'ball'}: Corresponds to the radius, not including the center element (size=0 has no effect).
    :param shape: {'square', 'cube', 'disk', 'ball'}
    :param dim: {0, 1, 2}: Dimension of the array which 2D structural element will be orthogonal to. For example, if
    you wish to apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be: shape=disk, dim=2.
    :return: numpy array: data dilated
    """
    # TODO: make a build_selem(radius, shape) function called here
    # TODO: enable custom selem
    # Create structuring element of desired shape and radius
    # Note: the trick here is to use the variable shape as the skimage.morphology function itself
    selem = globals()[shape](size)
    # If 2d kernel, replicate it along the specified dimension
    if len(selem.shape) == 2:
        selem3d = np.zeros([selem.shape[0]]*3)  # Note: selem.shape[0] and selem.shape[1] are supposed to be the same
        imid = np.floor(selem.shape[0] / 2).astype(int)
        if dim == 0:
            selem3d[:, imid, imid] = selem
        elif dim == 1:
            selem3d[imid, :, imid] = selem
        elif dim == 2:
            selem3d[:, :, imid] = selem
        else:
            raise ValueError("dim can only take values: {0, 1, 2}")
        selem = selem3d
    # else:
    #     # define structured element as a box with input dimensions
    #     selem = np.ones((radius[0], radius[1], radius[2]), dtype=np.dtype)
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