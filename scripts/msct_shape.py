#!/usr/bin/env python
########################################################################################################################
#
# This file contains useful functions for shape analysis based on spinal cord segmentation.
# The main input of these functions is a small image containing the binary spinal cord segmentation,
# ideally centered in the image.
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Modified: 2016-12-20
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

from __future__ import print_function, absolute_import, division

import os
import time
import math
from collections import OrderedDict
from random import randint
from itertools import compress

import numpy as np
import scipy.ndimage

import tqdm
from skimage import measure, filters

import sct_utils as sct
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
from msct_types import Centerline
from sct_straighten_spinalcord import smooth_centerline


def smoothing(image, sigma=1.0):
    return filters.gaussian(image, sigma=sigma)


def properties2d(image, resolution=None):
    label_img = measure.label(np.transpose(image))
    regions = measure.regionprops(label_img)
    areas = [r.area for r in regions]
    ix = np.argsort(areas)
    if len(regions) != 0:
        sc_region = regions[ix[-1]]
        try:
            ratio_minor_major = sc_region.minor_axis_length / sc_region.major_axis_length
        except ZeroDivisionError:
            ratio_minor_major = 0.0

        area = sc_region.area  # TODO: increase precision (currently single decimal)
        diameter = sc_region.equivalent_diameter
        major_l = sc_region.major_axis_length
        minor_l = sc_region.minor_axis_length
        if resolution is not None:
            area *= resolution[0] * resolution[1]
            # TODO: compute length depending on resolution. Here it assume the patch has the same X and Y resolution
            diameter *= resolution[0]
            major_l *= resolution[0]
            minor_l *= resolution[0]

            size_grid = 8.0 / resolution[0]  # assuming the maximum spinal cord radius is 8 mm
        else:
            size_grid = int(2.4 * sc_region.major_axis_length)

        """
        import matplotlib.pyplot as plt
        plt.imshow(label_img)
        plt.text(1, 1, sc_region.orientation, color='white')
        plt.show()
        """

        # y0, x0 = sc_region.centroid
        # orientation = sc_region.orientation
        #
        # resolution_grid = 0.25
        # x_grid, y_grid = np.mgrid[-size_grid:size_grid:resolution_grid, -size_grid:size_grid:resolution_grid]
        # coordinates_grid = np.array(list(zip(x_grid.ravel(), y_grid.ravel())))
        # coordinates_grid_image = np.array([[x0 + math.cos(orientation) * coordinates_grid[i, 0], y0 - math.sin(orientation) * coordinates_grid[i, 1]] for i in range(coordinates_grid.shape[0])])
        #
        # square = scipy.ndimage.map_coordinates(image, coordinates_grid_image.T, output=np.float32, order=0, mode='constant', cval=0.0)
        # square_image = square.reshape((len(x_grid), len(x_grid)))
        #
        # size_half = square_image.shape[1] / 2
        # left_image = square_image[:, :size_half]
        # right_image = np.fliplr(square_image[:, size_half:])
        #
        # dice_symmetry = np.sum(left_image[right_image == 1]) * 2.0 / (np.sum(left_image) + np.sum(right_image))

        """DEBUG
        import matplotlib.pyplot as plt
        plt.imshow(square_image)
        plt.text(3, 3, dice, color='white')
        plt.show()
        """

        sc_properties = {'area': area,
                         'bbox': sc_region.bbox,
                         'centroid': sc_region.centroid,
                         'eccentricity': sc_region.eccentricity,
                         'equivalent_diameter': diameter,
                         'euler_number': sc_region.euler_number,
                         'inertia_tensor': sc_region.inertia_tensor,
                         'inertia_tensor_eigvals': sc_region.inertia_tensor_eigvals,
                         'minor_axis_length': minor_l,
                         'major_axis_length': major_l,
                         'moments': sc_region.moments,
                         'moments_central': sc_region.moments_central,
                         'orientation': sc_region.orientation * 180.0 / math.pi,
                         'perimeter': sc_region.perimeter,
                         'ratio_minor_major': ratio_minor_major,
                         'solidity': sc_region.solidity  # convexity measure
                         # 'symmetry': dice_symmetry
                         }
    else:
        sc_properties = None

    return sc_properties


def assign_AP_and_RL_diameter(properties):
    """
    This script checks the orientation of the spinal cord and inverts axis if necessary to make sure the major axis is
    always labeled as right-left (RL), and the minor antero-posterior (AP).
    :param properties: dictionary generated by properties2d()
    :return: properties updated with new fields: AP_diameter, RL_diameter
    """
    if -45.0 < properties['orientation'] < 45.0:
        properties['RL_diameter'] = properties['major_axis_length']
        properties['AP_diameter'] = properties['minor_axis_length']
    else:
        properties['RL_diameter'] = properties['minor_axis_length']
        properties['AP_diameter'] = properties['major_axis_length']
    return properties


def compute_properties_along_centerline(im_seg, smooth_factor=5.0, interpolation_mode=0, algo_fitting='hanning',
                                        window_length=50, size_patch=7, remove_temp_files=1, verbose=1):
    """
    Compute shape property along spinal cord centerline. This algorithm computes the centerline,
    oversample it, extract 2D patch orthogonal to the centerline, compute the shape on the 2D patches, and finally
    undersample the shape information in order to match the input slice #.
    :param im_seg:
    :param smooth_factor:
    :param interpolation_mode:
    :param algo_fitting:
    :param window_length:
    :param remove_temp_files:
    :param verbose:
    :return:
    """

    # List of properties to output (in the right order)
    property_list = ['area',
                     'equivalent_diameter',
                     'AP_diameter',
                     'RL_diameter',
                     'ratio_minor_major',
                     'eccentricity',
                     'solidity',
                     'orientation']

    # TODO: make sure fname_segmentation and fname_disks are in the same space
    path_tmp = sct.tmp_create(basename="compute_properties_along_centerline", verbose=verbose)

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    im_seg.change_orientation("RPI", generate_path=True).save(path_tmp, mutable=True)
    fname_segmentation_orient = im_seg.absolutepath

    # # Change orientation of the input centerline into RPI
    # sct.printv('\nOrient centerline to RPI orientation...', verbose)
    # # im_seg = Image(file_data + ext_data)
    # fname_segmentation_orient = 'segmentation_rpi.nii.gz'
    # image = set_orientation(im_seg, 'RPI')
    # image.setFileName(fname_segmentation_orient)
    # image.save()

    # Initiating some variables
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    # Define the resampling resolution. Here, we take the minimum of half the pixel size along X or Y in order to have
    # sufficient precision upon resampling. Since we want isotropic resamping, we take the min between the two dims.
    resolution = min(float(px) / 2, float(py) / 2)
    # resolution = 0.5
    properties = {key: [] for key in property_list}
    properties['incremental_length'] = []
    properties['distance_from_C1'] = []
    properties['vertebral_level'] = []
    properties['z_slice'] = []

    # compute the spinal cord centerline based on the spinal cord segmentation
    number_of_points = nz  # 5 * nz
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = \
        smooth_centerline(fname_segmentation_orient, algo_fitting=algo_fitting, window_length=window_length,
                          verbose=verbose, nurbs_pts_number=number_of_points, all_slices=False, phys_coordinates=True,
                          remove_outliers=True)
    centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline,
                            x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

    sct.printv('Computing spinal cord shape along the spinal cord...')
    with tqdm.tqdm(total=centerline.number_of_points) as pbar:

        # Extracting patches perpendicular to the spinal cord and computing spinal cord shape
        for index in range(centerline.number_of_points):
            # value_out = -5.0
            value_out = 0.0
            # TODO: instead of iterating along the cord centerline, it would be better to simply iterate along Z, and
            # correct for angulation using the cosine. The current approach has 2 issues:
            # - the centerline is not homogeneously sampled along z (which is the reason it is oversampled)
            # - computationally expensive
            # - requires resampling to higher resolution --> to check: maybe required with cosine approach
            current_patch = centerline.extract_perpendicular_square(im_seg, index, size=size_patch,
                                                                    resolution=resolution,
                                                                    interpolation_mode=interpolation_mode,
                                                                    border='constant', cval=value_out)

            # check for pixels close to the spinal cord segmentation that are out of the image
            patch_zero = np.copy(current_patch)
            patch_zero[patch_zero == value_out] = 0.0
            # patch_borders = dilation(patch_zero) - patch_zero

            """
            if np.count_nonzero(patch_borders + current_patch == value_out + 1.0) != 0:
                c = image.transfo_phys2pix([centerline.points[index]])[0]
                print('WARNING: no patch for slice', c[2])
                continue
            """
            # compute shape properties on 2D patch
            sc_properties = properties2d(patch_zero, [resolution, resolution])
            # assign AP and RL to minor or major axis, depending on the orientation
            sc_properties = assign_AP_and_RL_diameter(sc_properties)
            # loop across properties and assign values for function output
            if sc_properties is not None:
                properties['incremental_length'].append(centerline.incremental_length[index])
                properties['z_slice'].append(im_seg.transfo_phys2pix([centerline.points[index]])[0][2])
                for property_name in property_list:
                    properties[property_name].append(sc_properties[property_name])
            else:
                c = im_seg.transfo_phys2pix([centerline.points[index]])[0]
                sct.printv('WARNING: no properties for slice', c[2])

            pbar.update(1)

    # smooth the spinal cord shape with a gaussian kernel if required
    # TODO: remove this smoothing
    # TODO: not all properties can be smoothed
    if smooth_factor != 0.0:  # smooth_factor is in mm
        import scipy
        window = scipy.signal.hann(smooth_factor / np.mean(centerline.progressive_length))
        for property_name in property_list:
            properties[property_name] = scipy.signal.convolve(properties[property_name], window, mode='same') / np.sum(window)

    # Display properties on the referential space. Requires intervertebral disks
    if verbose == 2:
        x_increment = 'distance_from_C1'
        import matplotlib.pyplot as plt
        # Display the image and plot all contours found
        fig, axes = plt.subplots(len(property_list), sharex=True, sharey=False)
        for k, property_name in enumerate(property_list):
            axes[k].plot(properties[x_increment], properties[property_name])
            axes[k].set_ylabel(property_name)

        axes[-1].set_xlabel('Position along the spinal cord (in mm)')

        plt.show()

    # extract all values for shape properties to be averaged across the oversampled centerline in order to match the
    # input slice #
    sorting_values = []
    for label in properties['z_slice']:
        if label not in sorting_values:
            sorting_values.append(label)
    # average spinal cord shape properties
    averaged_shape = OrderedDict()
    for property_name in property_list:
        averaged_shape[property_name] = []
        for label in sorting_values:
            averaged_shape[property_name].append(np.mean(
                [item for i, item in enumerate(properties[property_name]) if
                 properties['z_slice'][i] == label]))

    # Removing temporary folder
    os.chdir(curdir)
    if remove_temp_files:
        sct.rmtree(path_tmp)

    return averaged_shape


"""
Example of script that averages spinal cord shape from multiple subjects/patients, in a common reference frame (PAM50)
def prepare_data():

    folder_dataset = '/Volumes/data_shared/sct_testing/large/'
    import isct_test_function
    import json
    json_requirements = 'gm_model=0'
    data_subjects, subjects_name = sct_pipeline.generate_data_list(folder_dataset, json_requirements=json_requirements)

    fname_seg_images = []
    fname_disks_images = []
    group_images = []

    for subject_folder in data_subjects:
        if os.path.exists(os.path.join(subject_folder, 't2')):
            if os.path.exists(os.path.join(subject_folder, 't2', 't2_seg_manual.nii.gz')) and os.path.exists(os.path.join(subject_folder, 't2', 't2_disks_manual.nii.gz')):
                fname_seg_images.append(os.path.join(subject_folder, 't2', 't2_seg_manual.nii.gz'))
                fname_disks_images.append(os.path.join(subject_folder, 't2', 't2_disks_manual.nii.gz'))
                json_file = io.open(os.path.join(subject_folder, 'dataset_description.json'))
                dic_info = json.load(json_file)
                json_file.close()
                # pass keys and items to lower case
                dic_info = dict((k.lower(), v.lower()) for k, v in dic_info.items())
                if dic_info['pathology'] == 'HC':
                    group_images.append('b')
                else:
                    group_images.append('r')

    sct.printv('Number of images', len(fname_seg_images))

    property_list = ['area',
                     'equivalent_diameter',
                     'ratio_major_minor',
                     'eccentricity',
                     'solidity']

    average_properties(fname_seg_images, property_list, fname_disks_images, group_images, verbose=1)

"""
