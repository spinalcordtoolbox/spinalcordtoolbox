#!/usr/bin/env python
# -*- coding: utf-8
# Functions to get distance from PMJ for processing segmentation data

import logging
import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import get_centerline
from spinalcordtoolbox.resampling import resample_nib

NEAR_ZERO_THRESHOLD = 1e-6


def compute_csa_from_pmj(segmentation, pmj, distance, extent, param_centerline=None, verbose=1):  # TODO: change function name
    """
    Compute distance from PMJ projection on centerline for all teh centerline.
    Generate mask from segmentation of the slices used to process segmentation data corresponding to a distance from PMJ projection. 
    :param segmentation: input segmentation. Could be either an Image or a file name.
    :param pmj: label of PMJ.
    :param distance: 
    :param param_centerline: see centerline.core.ParamCenterline()
    :param verbose:
    :return :
    """
    # TODO: loop through distance (when multiple distances are given)
    im_seg = Image(segmentation).change_orientation('RPI')
    im_pmj = Image(pmj).change_orientation('RPI')
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    pr = min([px, py])
    im_segr = resample_nib(im_seg, new_size=[pr, pr, pz], new_size_type='mm', interpolation='linear')
    im_pmjr = resample_nib(im_pmj, new_size=[pr, pr, pz], new_size_type='mm', interpolation='linear')
    data_pmj = im_pmjr.data
    # Update dimensions from resampled image.
    nx, ny, nz, nt, px, py, pz, pt = im_segr.dim

    # Extract min and max index in Z direction
    data_seg = im_segr.data
    X, Y, Z = (data_seg > NEAR_ZERO_THRESHOLD).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # compute the spinal cord centerline based on the spinal cord segmentation
    _, arr_ctl, arr_ctl_der, fit_results = get_centerline(im_segr, param=param_centerline, verbose=verbose)

    # Get tagent vector at the top of the centerline | Maybe take a fiew slices to get the vector
    iz = max_z_index
    tangent_vect = np.array([arr_ctl_der[0][iz - min_z_index] * px, arr_ctl_der[1][iz - min_z_index] * py, pz])
    # Normalize vector by its L2 norm
    tangent_vect = tangent_vect / np.linalg.norm(tangent_vect)
    plane_pt = arr_ctl[:, -1] + [0, 0, min_z_index]
    line_pt = np.argwhere(data_pmj != 0)[0]  # list(im_pmjr.getNonZeroCoordinates()[0])[:-1]
    
    init_point = get_intersection_plane_line(plane_pt, tangent_vect, line_pt, tangent_vect)

    arr_length = get_distance_from_pmj(arr_ctl, init_point, px, py, pz)
    mask, slices = get_mask(arr_length, im_seg, distance, extent)
    slices = [slices, distance]
    return mask, slices


def get_intersection_plane_line(plane_point, plane_norm_vect, line_point, line_vect):
    """
    Compute intersection between a plane and a line.
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
    init_length = np.sqrt(((init_point[0] - centerline_points[0, -1]) * px)** 2 +
                 ((init_point[1] - centerline_points[1, -1]) * py)** 2 +
                 ((init_point[2] - centerline_points[2, -1]) * pz)** 2)
    length = init_length
    arr_length = [init_length]
    for i in range(number_of_points - 1, 0, -1):
        distance = np.sqrt(((centerline_points[0, i] - centerline_points[0, i - 1]) * px) ** 2  +
                    ((centerline_points[1, i] - centerline_points[1, i - 1]) * py) ** 2  +
                    ((centerline_points[2, i] - centerline_points[2, i - 1]) * pz)** 2)
        length += distance
        arr_length.append(length)
    arr_length = arr_length[::-1]
    arr_length = np.stack((arr_length, centerline_points[2]), axis=0)
    return arr_length


def get_mask(arr_lenght, im_seg, distance, extent):
    """
    Generate mask and corresponding slices from segmentation centered on distance from PMJ and with extent lenght on z axis.
    :param arr_lenght: nd-array: distance from PMJ and corresponding indexes.
    :param im_seg: Class: input segmentation.
    :return: mask, slices
    """       
    z_index_extent_min = get_nearest_index(arr_lenght, distance + extent/2)
    z_index_extent_max = get_nearest_index(arr_lenght, distance - extent/2)
    z_index_extent_min, z_index_extent_max = validate_lenght(z_index_extent_min, z_index_extent_max, arr_lenght, extent)

    mask = im_seg.copy()
    mask.data[:, :, 0:z_index_extent_min] = 0
    mask.data[:, :, z_index_extent_max :] = 0 # if last slice included, add + 1

    slices = "{}:{}".format(z_index_extent_min, z_index_extent_max -1) # is the last slice included or not??
    return mask, slices


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


def validate_lenght(z_min, z_max, arr, value): # TODO: Find a better way for this plz
    lenght = arr[0, z_min] - arr[0, z_max]
    if lenght > value:
        lenght_1 = arr[0, z_min + 1] - arr[0, z_max]
        lenght_2 = arr[0, z_min] - arr[0, z_max - 1]
        if np.abs(lenght_2 - value) < np.abs(lenght_1 - value):
            closest_lenght = lenght_2
            z_min_2 = z_min
            z_max_2 = z_max - 1
        else:
            closest_lenght = lenght_1
            z_min_2 = z_min + 1
            z_max_2 = z_max         
        if (np.abs(lenght - value)) < (np.abs(closest_lenght - value)):
            return  z_min, z_max
        else:
            return  z_min_2, z_max_2
    
    if lenght < value:
        lenght_1 = arr[0, z_min - 1] - arr[0, z_max]
        lenght_2 = arr[0, z_min] - arr[0, z_max + 1]

        if np.abs(lenght_2 - value) < np.abs(lenght_1 - value):
            closest_lenght = lenght_2
            z_min_2 = z_min
            z_max_2 = z_max + 1
        else:
            closest_lenght = lenght_1
            z_min_2 = z_min - 1
            z_max_2 = z_max         
        if (np.abs(lenght - value)) < (np.abs(closest_lenght - value)):
            return  z_min, z_max
        else:
            return  z_min_2, z_max_2

