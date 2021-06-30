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
    im_pmj = Image(pmj).change_orientation('RPI')
    native_orientation = im_seg.orientation
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    data_pmj = im_pmj.data

    if not im_seg.data.shape == im_pmj.data.shape:
        raise RuntimeError(f"segmentation and pmj should be in the same space coordinate.")

    # Extract min and max index in Z direction
    data_seg = im_seg.data
    X, Y, Z = (data_seg > NEAR_ZERO_THRESHOLD).nonzero()
    _, max_z_index = min(Z), max(Z)

    # Remove top slices  | TODO: check if center of mass of top slices is close to other slices, if not, remove
    im_seg.data[:, :, max_z_index - 4:max_z_index + 1] = 0

    # Compute the spinal cord centerline based on the spinal cord segmentation
    param_centerline.minmax = False  # Set to false to extrapolate centerline
    im_ctl, arr_ctl, arr_ctl_der, fit_results = get_centerline(im_seg, param=param_centerline, verbose=verbose)
    im_ctl.change_orientation(native_orientation)

    # Get coordinate of PMJ label
    pmj_coord = np.argwhere(data_pmj != 0)[0]
    # Get Z index of PMJ project on extrapolated centerline
    pmj_index = get_min_distance(pmj_coord, arr_ctl, px, py, pz)
    # Compute distance from PMJ along centerline
    arr_length = get_distance_from_pmj(arr_ctl, pmj_index, px, py, pz)

    # Get Z index of corresponding distance from PMJ with the specified extent
    zmin = get_nearest_index(arr_length, distance + extent/2)
    zmax = get_nearest_index(arr_length, distance - extent/2)

    zmin = np.argmin(np.array([np.abs(i - distance - extent/2) for i in arr_length[0]]))  # Check if seg starts at 1, need to get the right index
    zmax = np.argmin(np.array([np.abs(i - distance + extent/2) for i in arr_length[0]]))

    # Check if distance is out of bound
    if distance + extent/2 > arr_length[0][0]:
        raise ValueError("Input distance of " + str(distance) + " mm is out of bound for maximum distance of " + str(arr_length[0][0]) + " mm")

    if distance - extent/2 < arr_length[0][max_z_index]:  # Do we want instead max_z_index (so that we know that the segmentation is available?)
        raise ValueError("Input distance of " + str(distance) + " mm is out of bound for minimum distance of " + str(arr_length[0][-1]) + " mm")

    # Check if the range of selected slices are covered by the segmentation
    if not all(np.any(im_seg.data[:, :, z]) for z in range(zmin, zmax)):
        raise ValueError(f"The requested distances from the PMJ are not fully covered by the segmentation.\n"
                         f"The range of slices are: [{zmin}, {zmax}]")
    
    print('min', arr_length[0][zmax], 'max', arr_length[0][zmin])
    print(arr_length[0][zmin]-arr_length[0][zmax])
    # Create mask from segmentation centered on distance from PMJ and with extent length on z axis.
    mask = im_seg.copy()
    mask.data[:, :, 0:zmin] = 0
    mask.data[:, :, zmax:] = 0
    mask.change_orientation(native_orientation)

    # Get corresponding slices
    slices = "{}:{}".format(zmin, zmax - 1)  # TODO check if we include last slice
    return im_ctl, mask, slices


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
