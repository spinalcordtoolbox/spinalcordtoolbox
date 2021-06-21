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
    :param distance: float: Distance from PMJ.
    :param param_centerline: see centerline.core.ParamCenterline()
    :param verbose:
    :return mask:
    :return slices:

    """
    # TODO: loop through distance (when multiple distances are given)
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

    # Compute the spinal cord centerline based on the spinal cord segmentation
    _, arr_ctl, arr_ctl_der, fit_results = get_centerline(im_segr, param=param_centerline, verbose=verbose)

    # Get tagent vector at the top of the centerline | Maybe take a fiew slices to get the vector ?
    iz = max_z_index
    tangent_vect = np.array([arr_ctl_der[0][iz - min_z_index] * px, arr_ctl_der[1][iz - min_z_index] * py, pz])
    # Normalize vector by its L2 norm
    tangent_vect = tangent_vect / np.linalg.norm(tangent_vect)
    plane_pt = arr_ctl[:, -1] + [0, 0, min_z_index]
    line_pt = np.argwhere(data_pmj != 0)[0]  # list(im_pmjr.getNonZeroCoordinates()[0])[:-1]

    # Get intersection of plane passing through PMJ and continuation of centerline
    init_point = get_intersection_plane_line(plane_pt, tangent_vect, line_pt, tangent_vect)
    # Get the distance from PMJ for all slices of the centerline
    arr_length = get_distance_from_pmj(arr_ctl, init_point, px, py, pz)

    # Check if distance is out of bound
    if distance > arr_length[0][0]:
        raise ValueError("Input distance of " + str(distance) + " mm is out of bound for maximum distance of " + str(arr_length[0][0]) + " mm")
        # logging.error("Input distance of " + str(distance) + "mm is out of bound for maximum distance of " + str(arr_length[0][0]) + "mm")

    # Get Z index of corresponding distance from PMJ with the specified extent
    # TODO: add warining if specified
    z_index_extent_min = get_nearest_index(arr_length, distance + extent/2)
    z_index_extent_max = get_nearest_index(arr_length, distance - extent/2)
    # Check if extent corresponds to the lenght, if not add or remove a slice
    z_index_extent_min, z_index_extent_max = validate_length(z_index_extent_min, z_index_extent_max, arr_length, extent)  # Find a quicker way to solve this
    # Check if min Z index is available in the segmentation, if not, use the min_z_index of segmentation
    if z_index_extent_min < min_z_index:
        z_index_extent_min = min_z_index

    # Create mask from segmentation centered on distance from PMJ and with extent length on z axis.
    mask = im_seg.copy()
    mask.data[:, :, 0:z_index_extent_min] = 0
    mask.data[:, :, z_index_extent_max:] = 0

    # Get corresponding slices
    slices = "{}:{}".format(z_index_extent_min, z_index_extent_max - 1)
    slices = [slices, distance]

    return mask, slices


def get_intersection_plane_line(plane_point, plane_norm_vect, line_point, line_vect):
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


def get_distance_from_pmj(centerline_points, init_point, px, py, pz):
    """
    Compute distance from projected PMJ on centerline and cord centerline.
    :param centerline_points: 3xn array: Centerline in continuous coordinate (float) for each slice in RPI orientation.
    :param init_point: coordinates of projected PMJ on the centerline.
    :param px: x pixel size.
    :param py: y pixel size.
    :param pz: z pixel size.
    :return: nd-array: distance from PMJ and corresponding indexes.
    """
    number_of_points = centerline_points.shape[1]
    # Get distance from initial point and first coordinate of the centerline
    init_length = np.sqrt(((init_point[0] - centerline_points[0, -1]) * px) ** 2 +
                          ((init_point[1] - centerline_points[1, -1]) * py) ** 2 +
                          ((init_point[2] - centerline_points[2, -1]) * pz) ** 2)
    # Initialize length
    length = init_length
    arr_length = [init_length]
    for i in range(number_of_points - 1, 0, -1):
        distance = np.sqrt(((centerline_points[0, i] - centerline_points[0, i - 1]) * px) ** 2 +
                           ((centerline_points[1, i] - centerline_points[1, i - 1]) * py) ** 2 +
                           ((centerline_points[2, i] - centerline_points[2, i - 1]) * pz) ** 2)
        length += distance
        arr_length.append(length)
    arr_length = arr_length[::-1]
    arr_length = np.stack((arr_length, centerline_points[2]), axis=0)
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
