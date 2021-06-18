#!/usr/bin/env python
# -*- coding: utf-8

import logging

from matplotlib.pyplot import axis
import nibabel
import numpy as np

from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.aggregate_slicewise import Metric # TO remove
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.utils import sct_progress_bar

NEAR_ZERO_THRESHOLD = 1e-6

def compute_csa_from_pmj(segmentation, pmj, distance, extent, metric, param_centerline=None, verbose=1):
    """
    Compute morphometric measures of the spinal cord in the transverse (axial) plane from the segmentation.
    The segmentation could be binary or weighted for partial volume [0,1].

    :param segmentation: input segmentation. Could be either an Image or a file name.
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
    tangent_vect = np.array([arr_ctl_der[0][iz - min_z_index] * px,
                                     arr_ctl_der[1][iz - min_z_index] * py,
                                     pz])
    # Normalize vector by its L2 norm
    tangent_vect = tangent_vect / np.linalg.norm(tangent_vect)
    plane_pt = arr_ctl[:, -1] + [0,0, min_z_index]
    line_pt = list(im_pmjr.getNonZeroCoordinates()[0])[:-1]  # np.argwhere(data_pmj != 0)[0]
    
    init_point = get_intersection_plane_line(plane_pt, 
                                             tangent_vect,
                                             line_pt,tangent_vect)

    arr_length = get_distance_from_pmj(arr_ctl,
                                        init_point,
                                        px,
                                        py,
                                        pz)
    mask, slices = get_mask(arr_length, im_seg, distance, extent, metric)
    slices = [slices, distance]
    return mask, slices


def get_intersection_plane_line(plane_pt, plane_norm_vect, line_pt, line_vect):
    """
    
    """
    det = plane_norm_vect.dot(line_vect)
    w = line_pt - plane_pt
    si = -plane_norm_vect.dot(w) / det
    intersection = w + si * line_vect + plane_pt

    return intersection


def get_distance_from_pmj(centerline_points, init_point, px, py, pz):
    """

    """
    number_of_points = centerline_points.shape[1]
    init_length = np.sqrt(((init_point[0] - centerline_points[0, -1]) * px ) ** 2 +
                  ((init_point[1] - centerline_points[1, -1]) * py ) ** 2 + 
                    ((init_point[2] - centerline_points[2, -1]) * pz ) ** 2)
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


def get_mask(arr_lenght, seg, distance, extent, metric):
    """

    """       
    # TODO check with a segmentation that misses first slice
    z_index_extent_min = get_nearest_index(arr_lenght, distance + extent/2)
    z_index_extent_max = get_nearest_index(arr_lenght, distance - extent/2)
    z_index_extent_min, z_index_extent_max = validate_lenght(z_index_extent_min, z_index_extent_max, arr_lenght, extent)

    mask = seg.copy()
    mask.data[:, :, 0:z_index_extent_min] = 0
    mask.data[:, :, z_index_extent_max :] = 0 # if last slice included, add + 1

    slices = "{}:{}".format(z_index_extent_min, z_index_extent_max -1) # is the last slice included or not??
    return mask, slices


def get_nearest_index(arr, value):
    """

    """
    difference_array = np.absolute(arr[0]-value)
    index = arr[1][difference_array.argmin()]
    return int(index)

def validate_lenght(z_min, z_max, arr, value):
    lenght = arr[0, z_min] - arr[0, z_max]
    print('lenght:', lenght)
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
            print(closest_lenght)
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
            print(closest_lenght)
            return  z_min_2, z_max_2

