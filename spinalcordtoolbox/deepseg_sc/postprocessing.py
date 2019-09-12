#!/usr/bin/env python
# -*- coding: utf-8
# Deals with postprocessing on generated segmentation: remove outliers, fill holes, etc.


import logging
import numpy as np
from scipy.ndimage.measurements import label

logger = logging.getLogger(__name__)


def _fill_z_holes(zz_lst, data, z_spaccing):
    data_interpol = np.copy(data)

    for z_hole_start, z_hole_end in list(_list2range(zz_lst)):
        z_ref_start, z_ref_end = z_hole_start - 1, z_hole_end
        slice_ref_start, slice_ref_end = data[:, :, z_ref_start], data[:, :, z_ref_end]

        hole_cur_lst = list(range(z_hole_start, z_hole_end))
        lenght_hole = len(hole_cur_lst) + 1
        phys_lenght_hole = lenght_hole * z_spaccing

        denom_interpolation = (lenght_hole + 1)

        if phys_lenght_hole < 10:
            logger.warning('Filling a hole in the segmentation around z_slice #:' + str(z_ref_start))

            for idx_z, z_hole_cur in enumerate(hole_cur_lst):
                num_interpolation = (lenght_hole - idx_z - 1) * slice_ref_start  # Contribution of the bottom ref slice
                num_interpolation += (idx_z + 1) * slice_ref_end  # Contribution of the top ref slice

                slice_interpolation = num_interpolation * 1. / denom_interpolation
                slice_interpolation = (slice_interpolation > 0).astype(np.int)

                data_interpol[:, :, z_hole_cur] = slice_interpolation

    return data_interpol


def _list2range(lst):
    tmplst = lst[:]
    tmplst.sort()
    start = tmplst[0]

    currentrange = [start, start + 1]

    for item in tmplst[1:]:
        if currentrange[1] == item:  # contiguous
            currentrange[1] += 1
        else:  # new range start
            yield list(currentrange)
            currentrange = [item, item + 1]

    yield list(currentrange)  # last range


def _remove_blobs(data):
    """Remove false positive blobs, likely occuring in brain sections."""
    labeled_obj, num_obj = label(data)
    if num_obj > 1:  # If there is more than one connected object
        bigger_obj = (labeled_obj == (np.bincount(labeled_obj.flat)[1:].argmax() + 1))

        data2clean = np.copy(data)

        # remove blobs only above the bigger connected object
        z_max = np.max(np.where(bigger_obj)[2])
        data2clean[:, :, :z_max + 1] = 0

        labeled_obj2clean, num_obj2clean = label(data2clean)
        if num_obj2clean:  # If there is connected object above the biffer connected one
            for obj_id in range(1, num_obj2clean + 1):
                # if the blob has a volume < 10% of the bigger connected object, then remove it
                if np.sum(labeled_obj2clean == obj_id) < 0.1 * np.sum(bigger_obj):
                    logger.info('Removing small objects above slice#' + str(z_max))
                    data[np.where(labeled_obj2clean == obj_id)] = 0

    return data


def _remove_extrem_holes(z_lst, end_z, start_z=0):
    """Remove extrem holes from the list of holes so that we will not interpolate on the extrem slices."""
    if start_z in z_lst:
        while start_z in z_lst:
            z_lst = z_lst[1:]
            start_z += 1
        if len(z_lst):
            z_lst.pop(0)

    if end_z in z_lst:
        while end_z in z_lst:
            z_lst = z_lst[:-1]
            end_z -= 1

    return z_lst


def post_processing_volume_wise(im_in):
    """Post processing function."""
    data_in = im_in.data.astype(np.int)

    data_in = _remove_blobs(data_in)

    zz_zeros = [zz for zz in range(im_in.dim[2]) if 1 not in list(np.unique(data_in[:, :, zz]))]
    zz_holes = _remove_extrem_holes(zz_zeros, im_in.dim[2] - 1, 0)
    # filling z_holes, i.e. interpolate for z_slice not segmented
    im_in.data = _fill_z_holes(zz_holes, data_in, im_in.dim[6]) if len(zz_holes) else data_in
    return im_in
