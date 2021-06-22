#!/usr/bin/env python
# -*- coding: utf-8
# Functions to get distance from PMJ for processing segmentation data
# Author: Sandrine Bédard
import logging
import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import get_centerline, round_and_clip
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
    min_z_index = min(Z)
    # Compute the spinal cord centerline based on the spinal cord segmentation
    im_ctl, arr_ctl, arr_ctl_der, fit_results = get_centerline(im_segr, param=param_centerline, verbose=verbose)

    # Extrapolate centerline
    arr_ctl_extra = extrapolate_centerline(arr_ctl, im_segr, pz)

    # Create an image with the centerline | ONLY TO VALIDATE --> to remove
    im_centerline = im_segr.copy()
    im_centerline.data = np.zeros(im_centerline.data.shape)
    # Assign value=1 to centerline. Make sure to clip to avoid array overflow.
    im_centerline.data[round_and_clip(arr_ctl_extra[0], clip=[0, im_centerline.data.shape[0]]),
                       round_and_clip(arr_ctl_extra[1], clip=[0, im_centerline.data.shape[1]]),
                       np.array(range(im_segr.dim[2]))] = 1
    im_centerline.save('centerline_extrapolated.nii.gz')
    pmj_coord = np.argwhere(data_pmj != 0)[0]
    # Get Z index of PMJ project on extrapolated centerline
    pmj_index = get_min_distance(pmj_coord, arr_ctl_extra, px, py, pz)
    # Compute distance from PMJ along centerline
    arr_length = get_distance_from_pmj(arr_ctl_extra, pmj_index, px, py, pz)

    # Check if distance is out of bound
    if distance > arr_length[0][0]:
        raise ValueError("Input distance of " + str(distance) + " mm is out of bound for maximum distance of " + str(arr_length[0][0]) + " mm")

    if distance < arr_length[0][-1]:
        raise ValueError("Input distance of " + str(distance) + " mm is out of bound for minimum distance of " + str(arr_length[0][-1]) + " mm")

    # Get Z index of corresponding distance from PMJ with the specified extent
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
    mask.change_orientation(native_orientation)

    # Get corresponding slices
    slices = "{}:{}".format(z_index_extent_min, z_index_extent_max - 1)
    return mask, slices


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
    arr_length = []
    for i in range(z_index, 0, -1):
        distance = np.sqrt(((centerline_points[0, i] - centerline_points[0, i - 1]) * px) ** 2 +
                           ((centerline_points[1, i] - centerline_points[1, i - 1]) * py) ** 2 +
                           ((centerline_points[2, i] - centerline_points[2, i - 1]) * pz) ** 2)
        length += distance
        arr_length.append(length)
    arr_length = arr_length[::-1]
    arr_length = np.stack((arr_length, centerline_points[2][:z_index]), axis=0)
    return arr_length


def extrapolate_centerline(centerline, im_seg, pz):
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
