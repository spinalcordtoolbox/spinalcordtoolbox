#!/usr/bin/env python
# -*- coding: utf-8
# Functions to get distance from PMJ for processing segmentation data
# Author: Sandrine Bédard
import logging
import sys

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import get_centerline
from spinalcordtoolbox.centerline import curve_fitting
from spinalcordtoolbox.resampling import resample_nib

logger = logging.getLogger(__name__)

NEAR_ZERO_THRESHOLD = 1e-6


def get_slices_for_pmj_distance(segmentation, pmj, distance, extent, param_centerline=None, verbose=1):
    """
    Compute distance from PMJ projection on centerline for all the centerline.
    Generate mask from segmentation of the slices used to process segmentation data corresponding to a distance from PMJ projection.
    :param segmentation: input segmentation. Could be either an Image or a file name.
    :param pmj: label of PMJ.
    :param distance: float: Distance from Ponto-Medullary Junction (PMJ) in mm.
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
        raise RuntimeError(f"segmentation and pmj should be in the same space coordinate.")
    # Add PMJ label to the segmentation and then extrapolate to obtain a Centerline object defines between the PMJ
    # and the lower end of the centerline.
    im_seg_with_pmj = im_seg.copy()
    im_seg_with_pmj.data = im_seg_with_pmj.data + im_pmj.data
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
    im_ctl_seg_with_pmj, _, _, _ = get_centerline(im_seg_with_pmj, param_centerline, verbose=verbose)
    # Compute the incremental distance from the PMJ along each point in the centerline
    length_from_pmj = [ctl_seg_with_pmj.incremental_length[-1] - i for i in ctl_seg_with_pmj.incremental_length]
    # From this incremental distance, find the indices corresponding to the requested distance +/- extent/2 from the PMJ
    zmin = np.argmin(np.array([np.abs(i - distance - extent/2) for i in length_from_pmj]))
    zmax = np.argmin(np.array([np.abs(i - distance + extent/2) for i in length_from_pmj]))
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
    # TODO: why the "-1"?
    slices = "{}:{}".format(zmin, zmax-1)
    return im_ctl_seg_with_pmj.change_orientation(native_orientation), mask, slices


def get_intersection_plane_line(plane_point, plane_norm_vect, line_point, line_vect):  # TO REMOVE
    """
    Get intersection between a plane and a line.
    :param plane_point: coordinates of a point on the plane.
    :param plane_norm_vect: normal vector of the plane.
    :param line_point: point of the line.
    :param line_vect: vector of the direction of the line.
    :return: coordinates of intersection between line and plane.
    """
    det = plane_norm_vect.dot(line_vect)
    w = line_point - plane_point
    si = -plane_norm_vect.dot(w) / det
    intersection = w + si * line_vect + plane_point

    return intersection


def get_distance_from_pmj(centerline_points, z_index, px, py, pz):
    """
    Compute distance from projected PMJ on centerline and cord centerline.
    :param centerline_points: 3xn array: Centerline in continuous coordinate (float) for each slice in RPI orientation.
    :param z_index: z index of projected PMJ on the centerline.
    :param px: x pixel size.
    :param py: y pixel size.
    :param pz: z pixel size.
    :return: nd-array: distance from PMJ and corresponding indexes.
    """
    length = 0
    arr_length = [0]
    for i in range(z_index, 0, -1):
        distance = np.sqrt(((centerline_points[0, i] - centerline_points[0, i - 1]) * px) ** 2 +
                           ((centerline_points[1, i] - centerline_points[1, i - 1]) * py) ** 2 +
                           ((centerline_points[2, i] - centerline_points[2, i - 1]) * pz) ** 2)
        length += distance
        arr_length.append(length)
    arr_length = arr_length[::-1]
    arr_length = np.stack((arr_length, centerline_points[2][:z_index + 1]), axis=0)
    return arr_length


def extrapolate_centerline(centerline, im_seg, pz):  # TO REMOVE
    """
    Compute distance from projected PMJ on centerline and cord centerline.
    :param centerline: 3xn array: Centerline in continuous coordinate (float) for each slice in RPI orientation.
    :param im_seg: Image(): input segmentation.
    :param pz: float: z pixel size.
    """
    smooth = 30
    diff = np.sqrt((centerline[0, -1] - np.mean(centerline[0, -6:-1])) ** 2 +
                   (centerline[1, -1] - np.mean(centerline[1, -6:-1])) ** 2)
    if diff > 2:
        centerline = centerline[:, :-5]  # Remove last slices (maybe always do that?)
    x_mean = centerline[0, :]
    y_mean = centerline[1, :]
    z_mean = centerline[2, :]
    z_ref = np.array(range(im_seg.dim[2]))
    x_centerline_fit, x_centerline_deriv = curve_fitting.bspline(z_mean, x_mean, z_ref, smooth, pz=pz, deg_bspline=3)
    y_centerline_fit, y_centerline_deriv = curve_fitting.bspline(z_mean, y_mean, z_ref, smooth, pz=pz, deg_bspline=3)
    return np.array([x_centerline_fit, y_centerline_fit, z_ref])


def get_nearest_index(arr, value):
    """
    Return the index of the closest value to the distance in arr.
    :param arr: nd-array: distance from PMJ and corresponding indexes.
    :param value: float: value to find the closest index to.
    :returns index: int:
    """
    difference_array = np.absolute(arr[0]-value)
    index = arr[1][difference_array.argmin()]
    return int(index)


def get_min_distance(pmj, centerline, px, py, pz):
    distance = np.sqrt(((centerline[0, :] - pmj[0]) * px) ** 2 +
                       ((centerline[1, :] - pmj[1]) * py) ** 2 +
                       ((centerline[2, :] - pmj[2]) * pz) ** 2)
    return int(centerline[2, distance.argmin()])


def validate_length(z_min, z_max, arr, value):  # TODO: Find a better way for this
    length = arr[0, z_min] - arr[0, z_max]
    if length > value:
        length_1 = arr[0, z_min + 1] - arr[0, z_max]
        length_2 = arr[0, z_min] - arr[0, z_max - 1]
        if np.abs(length_2 - value) < np.abs(length_1 - value):
            closest_length = length_2
            z_min_2 = z_min
            z_max_2 = z_max - 1
        else:
            closest_length = length_1
            z_min_2 = z_min + 1
            z_max_2 = z_max
        if (np.abs(length - value)) < (np.abs(closest_length - value)):
            return z_min, z_max
        else:
            return z_min_2, z_max_2

    if length < value:
        length_1 = arr[0, z_min - 1] - arr[0, z_max]
        length_2 = arr[0, z_min] - arr[0, z_max + 1]

        if np.abs(length_2 - value) < np.abs(length_1 - value):
            closest_length = length_2
            z_min_2 = z_min
            z_max_2 = z_max + 1
        else:
            closest_length = length_1
            z_min_2 = z_min - 1
            z_max_2 = z_max
        if (np.abs(length - value)) < (np.abs(closest_length - value)):
            return z_min, z_max
        else:
            return z_min_2, z_max_2
