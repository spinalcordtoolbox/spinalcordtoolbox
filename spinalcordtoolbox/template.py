#!/usr/bin/env python
# -*- coding: utf-8
# Functions that utilize the template (e.g., PAM50)

from __future__ import absolute_import

import numpy as np

from sct_utils import log

def get_slices_from_vertebral_levels(im_vertlevel, level):
    """
    Find the slices of the corresponding vertebral level.
    Important: This function assumes that the 3rd dimension is Z.
    :param im_vertlevel: image object of vertebral labeling (e.g., label/template/PAM50_levels.nii.gz)
    :param level: int: vertebral level
    :return: list of int: slices
    """
    data_vertlevel = im_vertlevel.data
    slices = []
    # loop across z
    for iz in range(im_vertlevel.dim[2]):
        # find indices of non-null values
        indx, indy = np.where(data_vertlevel[:, :, iz])
        # average non-null values and round to closest
        try:
            average_value = int(round(np.mean(data_vertlevel[indx, indy, iz])))
            # if that matches the desired level, append it to slice list
            if average_value == level:
                slices.append(iz)
        except ValueError as e:
            # slice is empty (no indx found). Do nothing.
            log.debug('Empty slice: z=%s (%s)', iz, e)
    return slices


def get_vertebral_level_from_slice(im_vertlevel, idx_slice):
    """
    Find the vertebral level of the corresponding slice.
    Important: This function assumes that the 3rd dimension is Z.
    :param im_vertlevel: image object of vertebral labeling (e.g., label/template/PAM50_levels.nii.gz)
    :param idx_slice: int: slice (z)
    :return: int: vertebral level. If no level is found (only zeros on this slice), return None.
    """
    data_vertlevel = im_vertlevel.data
    # average non-null values and round to closest
    try:
        # find indices of non-null values
        indx, indy = np.where(data_vertlevel[:, :, idx_slice])
        vert_level = int(round(np.mean(data_vertlevel[indx, indy, idx_slice])))
    except ValueError as e:
        # slice is empty (no indx found). Do nothing.
        log.debug('Empty slice: z=%s (%s)', iz, e)
        vert_level = None
    return vert_level

