#!/usr/bin/env python
# -*- coding: utf-8
# Functions to get distance from PMJ for processing segmentation data
# Author: Sandrine Bédard
import logging
import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import get_centerline
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
    native_orientation = Image(segmentation).orientation
    im_seg = Image(segmentation).change_orientation('RPI')
    im_pmj = Image(pmj).change_orientation('RPI')
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    pr = min([px, py])
    im_segr = resample_nib(im_seg, new_size=[pr, pr, pz], new_size_type='mm', interpolation='linear')
    im_pmjr = resample_nib(im_pmj, new_size=[pr, pr, pz], new_size_type='mm', interpolation='linear')
    data_pmj = im_pmjr.data
    # Update dimensions from resampled image
    nx, ny, nz, nt, px, py, pz, pt = im_segr.dim

    # Extract min and max index in Z direction
    data_seg = im_segr.data
    X, Y, Z = (data_seg > NEAR_ZERO_THRESHOLD).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # Remove top slices  | TODO: check if center of mass of top slices is close to other slices, if not, remove
    im_segr.data[:, :, max_z_index - 4:max_z_index + 1] = 0

    # Compute the spinal cord centerline based on the spinal cord segmentation
    param_centerline.minmax = False  # Set to false to extrapolate centerline
    im_ctl, arr_ctl, arr_ctl_der, fit_results = get_centerline(im_segr, param=param_centerline, verbose=verbose)
    im_ctl.change_orientation(native_orientation)

    # Get coordinate of PMJ label
    pmj_coord = np.argwhere(data_pmj != 0)[0]
    # Get Z index of PMJ project on extrapolated centerline
    pmj_index = get_min_distance(pmj_coord, arr_ctl, px, py, pz)
    # Compute distance from PMJ along centerline
    arr_length = get_distance_from_pmj(arr_ctl, pmj_index, px, py, pz)

    # Check if distance is out of bound
    if distance > arr_length[0][0]:
        raise ValueError("Input distance of " + str(distance) + " mm is out of bound for maximum distance of " + str(arr_length[0][0]) + " mm")

    if distance < arr_length[0][-1]:  # Do we want instead max_z_index (so that we know that the segmentation is available?)
        raise ValueError("Input distance of " + str(distance) + " mm is out of bound for minimum distance of " + str(arr_length[0][-1]) + " mm")

    # Get Z index of corresponding distance from PMJ with the specified extent
    z_index_extent_min = get_nearest_index(arr_length, distance + extent/2)
    z_index_extent_max = get_nearest_index(arr_length, distance - extent/2)
    # Check if extent corresponds to the lenght, if not add or remove a slice
    # z_index_extent_min, z_index_extent_max = validate_length(z_index_extent_min, z_index_extent_max, arr_length, extent)  # Find a quicker way to solve this

    # Check if min Z index is available in the segmentation, if not, use the min_z_index of segmentation
    if z_index_extent_min < min_z_index:
        z_index_extent_min = min_z_index
        new_extent = arr_length[0][z_index_extent_min] - arr_length[0][z_index_extent_max]
        logger.warning("Extent of {} mm is out of bounds for given segmentation at a distance of {} mm from PMJ. Will use an extent of {} mm".format(extent, distance, new_extent))

    # Create mask from segmentation centered on distance from PMJ and with extent length on z axis.
    mask = im_seg.copy()
    mask.data[:, :, 0:z_index_extent_min] = 0
    mask.data[:, :, z_index_extent_max:] = 0
    mask.change_orientation(native_orientation)

    # Get corresponding slices
    slices = "{}:{}".format(z_index_extent_min, z_index_extent_max - 1)
    return im_ctl, mask, slices


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
