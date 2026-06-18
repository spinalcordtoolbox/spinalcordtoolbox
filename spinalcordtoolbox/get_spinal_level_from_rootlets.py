"""
Functions to get spinal levels from rootlets segmentation

Copyright (c) 2025 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE

Authors: Sandrine Bédard, Jan Valosek, Théo Matthieu
"""

import os
import numpy as np
from spinalcordtoolbox.image import Image, zeros_like, add_suffix
import spinalcordtoolbox.labels as sct_labels
from spinalcordtoolbox.scripts import sct_label_utils
from spinalcordtoolbox.labels import project_centerline, label_regions_from_reference
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline


def intersect_seg_and_rootlets(fname_seg, fname_rootlets, dilate_size):
    """
    Intersect the spinal cord segmentation and the spinal nerve rootlet segmentation.
    :param fname_seg: path to the spinal cord segmentation or image object
    :param fname_rootlets: path to the spinal nerve rootlet segmentation or image object
    :param dilate_size: size of spinal cord segmentation dilation in pixels
    :return: im_intersect: Image object of the intersection between the spinal cord segmentation and the spinal nerve
    rootlet segmentation
    """

    # Reorient to RPI
    im_rootlets = Image(fname_rootlets).change_orientation('RPI')
    im_seg = Image(fname_seg).change_orientation('RPI')

    # Convert dilate size from mm to pixels
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    dilate_size = int((dilate_size / px))

    # Dilate the SC segmentation using sct_maths
    fname_seg_dil = add_suffix(fname_seg, '_dil')
    os.system('sct_maths -i ' + im_seg.absolutepath + ' -o ' + fname_seg_dil + ' -dilate ' + str(dilate_size))

    # Load the dilated SC segmentation
    im_seg_dil = Image(fname_seg_dil).change_orientation('RPI')
    im_seg_dil_data = im_seg_dil.data

    # Intersect the rootlets and the dilated SC segmentation
    intersect_data = im_rootlets.data * im_seg_dil_data
    # Save the intersection using the Image class
    im_intersect = zeros_like(im_rootlets)
    im_intersect.data = intersect_data
    return im_intersect


def project_rootlets_to_segmentation(fname_seg, fname_rootlets, fname_intersect, rootlets_levels):
    """"
    Project the nerve rootlets intersection on the spinal cord segmentation
    :param im_rootlets: Image object of the spinal nerve rootlet segmentation
    :param im_seg: Image object of the spinal cord segmentation
    :param im_intersect: Image object of the intersection between the spinal cord segmentation and the spinal nerve
    rootlet segmentation
    :param rootlets_levels: list of the spinal nerve rootlets levels
    :param fname_rootlets: path to the spinal nerve rootlet segmentation
    :return: fname_spinal_levels: path to the spinal levels segmentation
    :return: start_end_slices: list of the spinal levels start and end slices
    """
    im_seg = Image(fname_seg).change_orientation('RPI')
    im_intersect = Image(fname_intersect).change_orientation('RPI')

    start_end_slices = dict()

    # Loop across the rootlets levels
    for level in rootlets_levels:
        # Get the list of slices where the level is present
        slices_list = np.unique(np.where(im_intersect.data == level)[2])
        # Skip the level if it is not present in the intersection
        if len(slices_list) != 0:
            min_slice = min(slices_list)
            max_slice = max(slices_list)
            start_end_slices[level] = {'start': min_slice, 'end': max_slice}

    # In the intersection image, keep only values at start slice for each level, set other slices to 0
    im_intersect_start = zeros_like(im_intersect)
    for level in rootlets_levels:
        start_slice = start_end_slices[level]['start']
        print(start_slice)
        im_intersect_start.data[:, :, start_slice-1] = im_intersect.data[:, :, start_slice]  # Shift of one as then with + 1 and projection this is inside the subsequent level
    fname_rostral_com = add_suffix(fname_rootlets, '_com')
    # Get a signle point per level at the start slice
    sct_labels.cubic_to_point(Image(im_intersect_start)).save(fname_rostral_com)
    # Add 1 to the levels so that the projection is done correctly (level 2 should project to level 3, etc, as this is located at the bottom slice of level and not the top slice)
    sct_label_utils.main(['-i', fname_rostral_com, '-add', '1', '-o', add_suffix(fname_rostral_com, '_add1')])
    fname_rostral_com_add1 = add_suffix(fname_rostral_com, '_add1')
    # Project the rootlets points on the spinal cord centerlline
    rostral_rootlets_projected = project_centerline(Image(im_seg), Image(fname_rostral_com_add1))
    rostral_rootlets_projected.save(add_suffix(fname_rostral_com_add1, '_projected'), mutable=True)

    # Use the projected single point labels to extract a labeled centerline from the input segmentation
    ctl_projected = label_regions_from_reference(Image(im_seg), rostral_rootlets_projected, centerline=True)
    seg_projected = label_regions_from_reference(Image(im_seg), rostral_rootlets_projected, centerline=False)
    ctl_projected.save(add_suffix(fname_rostral_com_add1, '_projected_centerline'), mutable=True)
    seg_projected.save(add_suffix(fname_rostral_com_add1, '_projected_seg'), mutable=True)

    # Mask the projected centerline with coverage of start of lowest level and end of most upper level
    im_ctl_data = ctl_projected.data
    min_level = min(rootlets_levels)
    max_level = max(rootlets_levels)
    # Find index of start and end slices of min and max levels on projected centerline,
    # since with projection, they may have changed compared to original segmentation
    max_level_start = np.min(np.where(im_ctl_data == max_level)[2])
    im_ctl_data[:, :, :(max_level_start)] = 0  # Set everything below to zero
    im_ctl_data[:, :, start_end_slices[min_level]['end']:] = 0  # Set everything above to zero
    ctl_projected.data = im_ctl_data
    fname_ctl_projected = add_suffix(fname_rostral_com_add1, '_projected_centerline_masked')
    ctl_projected.save(fname_ctl_projected, mutable=True)
    # Do the same thing on seg_projected:
    im_seg_data = seg_projected.data
    im_seg_data[:, :, :(max_level_start)] = 0  # Set everything below to zero
    im_seg_data[:, :, start_end_slices[min_level]['end']:] = 0  # Set everything above to zero
    seg_projected.data = im_seg_data
    fname_seg_projected = add_suffix(fname_rostral_com_add1, '_projected_seg_masked')
    seg_projected.save(fname_seg_projected, mutable=True)
    return fname_ctl_projected, fname_seg_projected


def project_rootlets_to_segmentation2(fname_seg, fname_rootlets, fname_intersect, rootlets_levels):
    """"
    Project the nerve rootlets intersection on the spinal cord segmentation
    :param im_rootlets: Image object of the spinal nerve rootlet segmentation
    :param im_seg: Image object of the spinal cord segmentation
    :param im_intersect: Image object of the intersection between the spinal cord segmentation and the spinal nerve
    rootlet segmentation
    :param rootlets_levels: list of the spinal nerve rootlets levels
    :param fname_rootlets: path to the spinal nerve rootlet segmentation
    :return: fname_spinal_levels: path to the spinal levels segmentation
    :return: start_end_slices: list of the spinal levels start and end slices
    """
    im_seg = Image(fname_seg).change_orientation('RPI')
    im_intersect = Image(fname_intersect).change_orientation('RPI')

    start_end_slices = dict()

    # Loop across the rootlets levels
    for level in rootlets_levels:
        # Get the list of slices where the level is present
        slices_list = np.unique(np.where(im_intersect.data == level)[2])
        # Skip the level if it is not present in the intersection
        if len(slices_list) != 0:
            min_slice = min(slices_list)
            max_slice = max(slices_list)
            start_end_slices[level] = {'start': min_slice, 'end': max_slice}
    # Get middle of end and start of adjacent levels to define the start and end slices of each level
    levels_slices_start = dict()
    for i in range(len(rootlets_levels)):
        level = rootlets_levels[i]
        # compute start
        if i == len(rootlets_levels) - 1:
            # For the last level, keep original start
            start = start_end_slices[level]['start']
        else:
            # For other levels, if next level exists use midpoint, otherwise keep original
            next_level = rootlets_levels[i + 1]
            if next_level in start_end_slices:
                start = int((start_end_slices[level]['start'] + start_end_slices[next_level]['end']) / 2)
            else:
                start = start_end_slices[level]['start']
        # compute end
        if i == 0:
            # For the first level, keep original start
            end = start_end_slices[level]['end']
        else:
            # For other levels, start from the previous level's end
            prev_level = rootlets_levels[i - 1]
            end = levels_slices_start[prev_level]['start']

        levels_slices_start[level] = {'start': start, 'end': end}
    # Place the level values in the spinal cord segmentation 
    im_rootlets_levels = im_seg.copy()
    for level in rootlets_levels:
        start_slice = levels_slices_start[level]['start']
        end_slice = levels_slices_start[level]['end']
        print(f"Level {level}: Start slice = {start_slice}, End slice = {end_slice}")
        im_rootlets_levels.data[:, :, start_slice:end_slice] = level * im_rootlets_levels.data[:, :, start_slice:end_slice]

    # Use the projected single point labels to extract a labeled centerline from the input segmentation
    #ctl_projected = label_regions_from_reference(Image(im_seg), rostral_rootlets_projected, centerline=True)
    #seg_projected = label_regions_from_reference(Image(im_seg), rostral_rootlets_projected, centerline=False)
    #ctl_projected.save(add_suffix(fname_rostral_com_add1, '_projected_centerline'), mutable=True)
    fname_seg_projected = add_suffix(fname_seg, '_spinal_level')
    im_rootlets_levels.save(fname_seg_projected, mutable=True)
    # Mask the projected centerline with coverage of start of lowest level and end of most upper level
    # im_ctl_data = ctl_projected.data
    min_level = min(rootlets_levels)
    max_level = max(rootlets_levels)
    # # Find index of start and end slices of min and max levels on projected centerline,
    # # since with projection, they may have changed compared to original segmentation
    max_level_start = np.min(np.where(im_rootlets_levels.data == max_level)[2])
    # im_ctl_data[:, :, :(max_level_start)] = 0  # Set everything below to zero
    # im_ctl_data[:, :, start_end_slices[min_level]['end']:] = 0  # Set everything above to zero
    # ctl_projected.data = im_ctl_data
    # fname_ctl_projected = add_suffix(fname_rostral_com_add1, '_projected_centerline_masked')
    # ctl_projected.save(fname_ctl_projected, mutable=True)
    # # Do the same thing on seg_projected:
    im_seg_data = im_rootlets_levels.data
    im_seg_data[:, :, :(max_level_start)] = 0  # Set everything below to zero
    im_seg_data[:, :, levels_slices_start[min_level]['end']:] = 0  # Set everything above to zero
    im_rootlets_levels.data = im_seg_data
    fname_seg_projected = add_suffix(fname_seg_projected, '_masked')
    im_rootlets_levels.save(fname_seg_projected, mutable=True)

    # project on centerline
    im_rootlets_levels_ctl = project_centerline(Image(im_seg), Image(im_rootlets_levels))
    fname_seg_projected_ctl = add_suffix(fname_seg_projected, '_masked_ctl')
    im_rootlets_levels_ctl.save(fname_seg_projected_ctl, mutable=True)

    return fname_seg_projected_ctl, fname_seg_projected