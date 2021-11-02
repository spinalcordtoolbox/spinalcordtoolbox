#!/usr/bin/env python
# -*- coding: utf-8
# Functions to get distance from PMJ for processing segmentation data
# Author: Sandrine Bédard
import logging

import numpy as np
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import get_centerline

logger = logging.getLogger(__name__)

NEAR_ZERO_THRESHOLD = 1e-6


def get_slices_for_pmj_distance(segmentation, pmj, distance, extent, param_centerline=None, verbose=1):
    """
    Interpolate centerline with pontomedullary junction (PMJ) label and compute distance from PMJ along the centerline.
    Generate a mask from segmentation of the slices used to process segmentation data corresponding to a distance from PMJ.
    :param segmentation: input segmentation. Could be either an Image or a file name.
    :param pmj: label of PMJ.
    :param distance: float: Distance from PMJ in mm.
    :param extent: extent of the coverage mask in mm.
    :param param_centerline: see centerline.core.ParamCenterline()
    :param verbose:
    :return im_ctl:
    :return mask:
    :return slices:

    """
    im_seg = Image(segmentation).change_orientation('RPI')
    native_orientation = im_seg.orientation
    im_seg.change_orientation('RPI')
    im_pmj = Image(pmj).change_orientation('RPI')
    if not im_seg.data.shape == im_pmj.data.shape:
        raise RuntimeError("segmentation and pmj should be in the same space coordinate.")
    # Add PMJ label to the segmentation and then extrapolate to obtain a Centerline object defined between the PMJ
    # and the lower end of the centerline.
    im_seg_with_pmj = im_seg.copy()
    im_seg_with_pmj.data = im_seg_with_pmj.data + im_pmj.data

    # Get max and min index of the segmentation with pmj
    _, _, Z = (im_seg_with_pmj.data > NEAR_ZERO_THRESHOLD).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    from spinalcordtoolbox.straightening import _get_centerline
    # Linear interpolation (vs. bspline) ensures strong robustness towards defective segmentations at the top slices.
    param_centerline.algo_fitting = 'linear'
    # On top of the linear interpolation we add some smoothing to remove discontinuities.
    param_centerline.smooth = 50
    param_centerline.minmax = True
    # Compute spinalcordtoolbox.types.Centerline class
    ctl_seg_with_pmj = _get_centerline(im_seg_with_pmj, param_centerline, verbose=verbose)
    # Also get the image centerline (because it is a required output)
    # TODO: merge _get_centerline into get_centerline
    im_ctl_seg_with_pmj, arr_ctl, _, _ = get_centerline(im_seg_with_pmj, param_centerline, verbose=verbose)
    # Compute the incremental distance from the PMJ along each point in the centerline
    length_from_pmj = ctl_seg_with_pmj.incremental_length_inverse[::-1]
    # From this incremental distance, find the indices corresponding to the requested distance +/- extent/2 from the PMJ
    # Get the z index corresponding to the segmentation since the centerline only includes slices of the segmentation.
    z_ref = np.array(range(min_z_index.astype(int), max_z_index.max().astype(int) + 1))
    zmin = z_ref[np.argmin(np.array([np.abs(i - distance - extent/2) for i in length_from_pmj]))]
    zmax = z_ref[np.argmin(np.array([np.abs(i - distance + extent/2) for i in length_from_pmj]))]

    # Check if distance is out of bounds
    if distance > length_from_pmj[0]:
        raise ValueError("Input distance of " + str(distance) + " mm is out of bounds for maximum distance of " + str(length_from_pmj[0]) + " mm")

    if distance < length_from_pmj[-1]:
        raise ValueError("Input distance of " + str(distance) + " mm is out of bounds for minimum distance of " + str(length_from_pmj[-1]) + " mm")

    # Check if the range of selected slices are covered by the segmentation
    if not all(np.any(im_seg.data[:, :, z]) for z in range(zmin, zmax)):
        raise ValueError(f"The requested distances from the PMJ are not fully covered by the segmentation.\n"
                         f"The range of slices are: [{zmin}, {zmax}]")

    # Create mask from segmentation centered on distance from PMJ and with extent length on z axis.
    mask = im_seg.copy()
    mask.data[:, :, 0:zmin] = 0
    mask.data[:, :, zmax:] = 0
    mask.change_orientation(native_orientation)

    # Get corresponding slices
    slices = "{}:{}".format(zmin, zmax-1)  # -1 since the last slice is included to compute CSA after.

    return im_ctl_seg_with_pmj.change_orientation(native_orientation), mask, slices, arr_ctl
