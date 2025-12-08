"""
Functions to get spinal levels from rootlets segmentation

Copyright (c) 2025 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import os
import numpy as np

#import numpy as np
from spinalcordtoolbox.image import Image, zeros_like, add_suffix
import spinalcordtoolbox.labels as sct_labels
from spinalcordtoolbox.scripts import sct_label_utils
from spinalcordtoolbox.labels import project_centerline, label_regions_from_reference


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
 #   fname_intersect = add_suffix(fname_rootlets, '_intersect')
  #  im_intersect.save(fname_intersect)
   # print(f'Intersection saved in {fname_intersect}.')

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
    im_rootlets = Image(fname_rootlets).change_orientation('RPI')
    im_seg = Image(fname_seg).change_orientation('RPI')
    im_intersect = Image(fname_intersect).change_orientation('RPI')
    im_spinal_levels_data = np.copy(im_seg.data)

    start_end_slices = dict()

    # Loop across the rootlets levels
    for level in rootlets_levels:
        # Get the list of slices where the level is present
        slices_list = np.unique(np.where(im_intersect.data == level)[2])
        # Skip the level if it is not present in the intersection
        if len(slices_list) != 0:
            min_slice = min(slices_list)
            max_slice = max(slices_list)
            # get center of mass of rootlets seg on min slice
           # com = np.round(np.mean(np.where(im_rootlets.data[:, :, min_slice] == level)[0])).astype(int)
            # project COM on spinal cord centerline
            start_end_slices[level] = {'start': min_slice, 'end': max_slice}
            # Color the SC segmentation with the level
            #im_spinal_levels_data[:, :, min_slice:max_slice+1][im_seg.data[:, :, min_slice:max_slice+1] == 1] = level

    # Set zero to the slices with no intersection
   # im_spinal_levels_data[im_spinal_levels_data == 1] = 0
    # In the intersection image, keep only values at start slice for each level, set other slices to 0
    im_intersect_start = zeros_like(im_intersect)
    for level in rootlets_levels:
        start_slice = start_end_slices[level]['start']
        print(start_slice)
        im_intersect_start.data[:, :, start_slice-1] = im_intersect.data[:, :, start_slice] # Shift of one as then with + 1 and projection this is inside the subsequent level
    fname_rostral_com = add_suffix(fname_rootlets, '_com')
    sct_labels.cubic_to_point(Image(im_intersect_start)).save(fname_rostral_com)
    sct_label_utils.main(['-i', fname_rostral_com, '-add', '1', '-o', add_suffix(fname_rostral_com, '_add1')])
    fname_rostral_com_add1 = add_suffix(fname_rostral_com, '_add1')
    rostral_rootlets_projected = project_centerline(Image(im_seg), Image(fname_rostral_com_add1))
    rostral_rootlets_projected.save(add_suffix(fname_rostral_com_add1, '_projected'), mutable=True)

    # Project the levels on the spinal cord centerline:
    ctl_projected = label_regions_from_reference(Image(im_seg), rostral_rootlets_projected, centerline=True)
    seg_projected = label_regions_from_reference(Image(im_seg), rostral_rootlets_projected, centerline=False)
    ctl_projected.save(add_suffix(fname_rostral_com_add1, '_projected_centerline'), mutable=True)
    seg_projected.save(add_suffix(fname_rostral_com_add1, '_projected_seg'), mutable=True)
    # Mask the projected centerline to keep stop of most upper level and start of lower level
    im_ctl_data = ctl_projected.data
    min_level = min(rootlets_levels)
    max_level = max(rootlets_levels)
    print(start_end_slices[max_level]['start']+1, start_end_slices[min_level]['end'])
    im_ctl_data[:, :, :start_end_slices[max_level]['start']] = 0
    im_ctl_data[:, :, start_end_slices[min_level]['end']:] = 0
    ctl_projected.data = im_ctl_data
    fname_ctl_projected = add_suffix(fname_rostral_com_add1, '_projected_centerline_masked')
    ctl_projected.save(fname_ctl_projected, mutable=True)
    # Do the same thing on seg_projected:
    print(start_end_slices[max_level]['start']+1, start_end_slices[min_level]['end'])
    im_seg_data = seg_projected.data
    im_seg_data[:, :, :start_end_slices[max_level]['start']] = 0
    im_seg_data[:, :, start_end_slices[min_level]['end']:] = 0
    seg_projected.data = im_seg_data
    fname_seg_projected = add_suffix(fname_rostral_com_add1, '_projected_seg_masked')
    seg_projected.save(fname_seg_projected, mutable=True)
    return fname_ctl_projected, fname_seg_projected


def main():
    fname_rootlets = '/Users/sandrinebedard/spinalcordtoolbox/sct_course_data/single_subject/data/t2/t2_rootlets.nii.gz'
    fname_seg = '/Users/sandrinebedard/spinalcordtoolbox/sct_course_data/single_subject/data/t2/t2_seg.nii.gz'
    dilate_size = 3 # TODO this should in mm or resample image to isotropic before dilating

    # Load input images using the SCT Image class
    im_rootlets = Image(fname_rootlets).change_orientation('RPI')
    im_seg = Image(fname_seg).change_orientation('RPI')
    fname_intersect = intersect_seg_and_rootlets(im_rootlets, fname_seg, fname_rootlets, dilate_size)
    im_intersect = Image(fname_intersect).change_orientation('RPI')
    # Get unique values in the rootlets segmentation larger than 0
    rootlets_levels = np.unique(im_rootlets.data[np.where(im_rootlets.data > 0)])

    # Project the nerve rootlets intersection on the spinal cord segmentation to obtain spinal levels
    fname_spinal_levels, start_end_slices = project_rootlets_to_segmentation(im_rootlets, im_seg, im_intersect,
                                                                             rootlets_levels, fname_rootlets)


if __name__ == '__main__':
    main()