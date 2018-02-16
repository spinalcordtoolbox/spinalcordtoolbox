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


import numpy as np
import sct_utils as sct
import os
import time
import math
from random import randint
from skimage import measure, filters
import shutil
import matplotlib.pyplot as plt
from itertools import compress
from sct_image import Image, set_orientation
from msct_types import Centerline
from sct_straighten_spinalcord import smooth_centerline


def find_contours(image, threshold=0.5, smooth_sigma=0.0, verbose=1):
    image_input = image
    if smooth_sigma != 0.0:
        image_input = smoothing(image_input, sigma=smooth_sigma, verbose=verbose)
    sct.printv(np.min(image_input), np.max(image_input))
    contours = measure.find_contours(image_input, threshold)

    if verbose == 2:
        import matplotlib.pyplot as plt
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(image_input, interpolation='nearest', cmap=plt.cm.gray)

        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    return contours


def smoothing(image, sigma=1.0, verbose=1):
    return filters.gaussian(image, sigma=sigma)


def properties2d(image, resolution=None, verbose=1):
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

        area = sc_region.area
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

        y0, x0 = sc_region.centroid
        orientation = sc_region.orientation

        resolution_grid = 0.25
        x_grid, y_grid = np.mgrid[-size_grid:size_grid:resolution_grid, -size_grid:size_grid:resolution_grid]
        coordinates_grid = np.array(list(zip(x_grid.ravel(), y_grid.ravel())))
        coordinates_grid_image = np.array([[x0 + math.cos(orientation) * coordinates_grid[i, 0], y0 - math.sin(orientation) * coordinates_grid[i, 1]] for i in range(coordinates_grid.shape[0])])

        from scipy.ndimage import map_coordinates
        square = map_coordinates(image, coordinates_grid_image.T, output=np.float32, order=0, mode='constant', cval=0.0)
        square_image = square.reshape((len(x_grid), len(x_grid)))

        size_half = square_image.shape[1] / 2
        left_image = square_image[:, :size_half]
        right_image = np.fliplr(square_image[:, size_half:])

        dice_symmetry = np.sum(left_image[right_image == 1]) * 2.0 / (np.sum(left_image) + np.sum(right_image))

        """
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
                         'solidity': sc_region.solidity,  # convexity measure
                         'symmetry': dice_symmetry
                         }
    else:
        sc_properties = None

    return sc_properties


def average_properties(fname_seg_images, property_list, fname_disks_images, group_images, verbose=1):
    if len(fname_seg_images) != len(fname_disks_images):
        raise ValueError('ERROR: each segmentation image must be accompagnied by a disk image')

    # variables
    xtick_disks = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    regions_labels = {'-1': 'PONS', '0': 'MO',
                      '1': 'C1', '2': 'C2', '3': 'C3', '4': 'C4', '5': 'C5', '6': 'C6', '7': 'C7',
                      '8': 'T1', '9': 'T2', '10': 'T3', '11': 'T4', '12': 'T5', '13': 'T6', '14': 'T7', '15': 'T8', '16': 'T9', '17': 'T10', '18': 'T11', '19': 'T12',
                      '20': 'L1', '21': 'L2', '22': 'L3', '23': 'L4', '24': 'L5',
                      '25': 'S1', '26': 'S2', '27': 'S3', '28': 'S4', '29': 'S5',
                      '30': 'Co'}
    convert_vertlabel2disklabel = {'PONS': 'Pons', 'MO': 'Pons-MO',
                          'C1': 'MO-C1', 'C2': 'C1-C2', 'C3': 'C2-C3', 'C4': 'C3-C4', 'C5': 'C4-C5', 'C6': 'C5-C6', 'C7': 'C6-C7',
                          'T1': 'C7-T1', 'T2': 'T1-T2', 'T3': 'T2-T3', 'T4': 'T3-T4', 'T5': 'T4-T5', 'T6': 'T5-T6', 'T7': 'T6-T7', 'T8': 'T7-T8', 'T9': 'T8-T9',
                          'T10': 'T9-T10', 'T11': 'T10-T11', 'T12': 'T11-T12',
                          'L1': 'T12-L1', 'L2': 'L1-L2', 'L3': 'L2-L3', 'L4': 'L3-L4', 'L5': 'L4-L5',
                          'S1': 'L5-S1', 'S2': 'S1-S2', 'S3': 'S2-S3', 'S4': 'S3-S4', 'S5': 'S4-S5',
                          'Co': 'S5-Co'}
    xlabel_disks = [convert_vertlabel2disklabel[regions_labels[str(label)]] for label in xtick_disks]

    if verbose == 2:
        # Display the image and plot all contours found
        fig, axes = plt.subplots(len(property_list), sharex=True, sharey=False)

        xlim = [min(xtick_disks), max(xtick_disks)]

    for i, fname_seg in enumerate(fname_seg_images):
        sct.printv(fname_seg)
        fname_disks = fname_disks_images[i]
        properties_along_centerline = compute_properties_along_centerline(fname_seg, property_list, fname_disks, verbose)

        centerline = properties_along_centerline['centerline']

        mask_points = np.array([True if isinstance(item, str) else False for item in centerline.l_points])
        dist_points_rel = list(compress(centerline.dist_points_rel, mask_points))
        l_points = list(compress(centerline.l_points, mask_points))

        relative_position = [dist_points_rel[k] + centerline.labels_regions[l_points[k]] for k in range(len(l_points))]
        relative_position = [item - 51 if item >= 51 else item for item in relative_position]
        relative_position = [item - 50 if item >= 50 else item for item in relative_position]

        if verbose == 2:
            labels = [centerline.labels_regions[l_points[k]] for k in range(len(l_points))]
            xlim = [min(labels), max(labels)]
            for k, property_name in enumerate(property_list):
                axes[k].plot(relative_position, list(compress(properties_along_centerline[property_name], mask_points)), color=group_images[i])
                axes[k].set_xlim(xlim)

    if verbose == 2:
        for k, property_name in enumerate(property_list):
            axes[k].set_ylabel(property_name)
        plt.xticks(xtick_disks, xlabel_disks, rotation=30)
        axes[-1].set_xlim(xlim)
        sct.printv('\nAffichage des resultats', verbose=verbose)
        plt.savefig('shape_results.png')
        plt.show()


def compute_properties_along_centerline(fname_seg_image, property_list, fname_disks_image=None, smooth_factor=5.0, interpolation_mode=0, remove_temp_files=1, verbose=1):

    # Check list of properties
    # If diameters is in the list, compute major and minor axis length and check orientation
    compute_diameters = False
    property_list_local = list(property_list)
    if 'diameters' in property_list_local:
        compute_diameters = True
        property_list_local.remove('diameters')
        property_list_local.append('major_axis_length')
        property_list_local.append('minor_axis_length')
        property_list_local.append('orientation')

    # TODO: make sure fname_segmentation and fname_disks are in the same space
    path_tmp = sct.tmp_create(basename="compute_properties_along_centerline", verbose=verbose)

    sct.copy(fname_seg_image, path_tmp)
    if fname_disks_image is not None:
        sct.copy(fname_disks_image, path_tmp)

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    fname_segmentation = os.path.abspath(fname_seg_image)
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)

    # Change orientation of the input centerline into RPI
    sct.printv('\nOrient centerline to RPI orientation...', verbose)
    im_seg = Image(file_data + ext_data)
    fname_segmentation_orient = 'segmentation_rpi' + ext_data
    image = set_orientation(im_seg, 'RPI')
    image.setFileName(fname_segmentation_orient)
    image.save()

    # Initiating some variables
    nx, ny, nz, nt, px, py, pz, pt = image.dim
    resolution = 0.5
    properties = {key: [] for key in property_list_local}
    properties['incremental_length'] = []
    properties['distance_from_C1'] = []
    properties['vertebral_level'] = []
    properties['z_slice'] = []

    # compute the spinal cord centerline based on the spinal cord segmentation
    number_of_points = 5 * nz
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(fname_segmentation_orient, algo_fitting='nurbs', verbose=verbose, nurbs_pts_number=number_of_points, all_slices=False, phys_coordinates=True, remove_outliers=True)
    centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

    # Compute vertebral distribution along centerline based on position of intervertebral disks
    if fname_disks_image is not None:
        fname_disks = os.path.abspath(fname_disks_image)
        path_data, file_data, ext_data = sct.extract_fname(fname_disks)
        im_disks = Image(file_data + ext_data)
        fname_disks_orient = 'disks_rpi' + ext_data
        image_disks = set_orientation(im_disks, 'RPI')
        image_disks.setFileName(fname_disks_orient)
        image_disks.save()

        image_disks = Image(fname_disks_orient)
        coord = image_disks.getNonZeroCoordinates(sorting='z', reverse_coord=True)
        coord_physical = []
        for c in coord:
            c_p = image_disks.transfo_pix2phys([[c.x, c.y, c.z]])[0]
            c_p.append(c.value)
            coord_physical.append(c_p)
        centerline.compute_vertebral_distribution(coord_physical)

    sct.printv('Computing spinal cord shape along the spinal cord...')
    timer_properties = sct.Timer(number_of_iteration=centerline.number_of_points)
    timer_properties.start()
    # Extracting patches perpendicular to the spinal cord and computing spinal cord shape
    for index in range(centerline.number_of_points):
        # value_out = -5.0
        value_out = 0.0
        current_patch = centerline.extract_perpendicular_square(image, index, resolution=resolution, interpolation_mode=interpolation_mode, border='constant', cval=value_out)

        # check for pixels close to the spinal cord segmentation that are out of the image
        from skimage.morphology import dilation
        patch_zero = np.copy(current_patch)
        patch_zero[patch_zero == value_out] = 0.0
        patch_borders = dilation(patch_zero) - patch_zero

        """
        if np.count_nonzero(patch_borders + current_patch == value_out + 1.0) != 0:
            c = image.transfo_phys2pix([centerline.points[index]])[0]
            print('WARNING: no patch for slice', c[2])
            timer_properties.add_iteration()
            continue
        """

        sc_properties = properties2d(patch_zero, [resolution, resolution])
        if sc_properties is not None:
            properties['incremental_length'].append(centerline.incremental_length[index])
            if fname_disks_image is not None:
                properties['distance_from_C1'].append(centerline.dist_points[index])
                properties['vertebral_level'].append(centerline.l_points[index])
            properties['z_slice'].append(image.transfo_phys2pix([centerline.points[index]])[0][2])
            for property_name in property_list_local:
                properties[property_name].append(sc_properties[property_name])
        else:
            c = image.transfo_phys2pix([centerline.points[index]])[0]
            print('WARNING: no properties for slice', c[2])

        timer_properties.add_iteration()
    timer_properties.stop()

    # Adding centerline to the properties for later use
    properties['centerline'] = centerline

    # We assume that the major axis is in the right-left direction
    # this script checks the orientation of the spinal cord and invert axis if necessary to make sure the major axis is right-left
    if compute_diameters:
        diameter_major = properties['major_axis_length']
        diameter_minor = properties['minor_axis_length']
        orientation = properties['orientation']
        for i, orientation_item in enumerate(orientation):
            if -45.0 < orientation_item < 45.0:
                continue
            else:
                temp = diameter_minor[i]
                properties['minor_axis_length'][i] = diameter_major[i]
                properties['major_axis_length'][i] = temp

        properties['RL_diameter'] = properties['major_axis_length']
        properties['AP_diameter'] = properties['minor_axis_length']
        del properties['major_axis_length']
        del properties['minor_axis_length']

    # smooth the spinal cord shape with a gaussian kernel if required
    # TODO: not all properties can be smoothed
    if smooth_factor != 0.0:  # smooth_factor is in mm
        import scipy
        window = scipy.signal.hann(smooth_factor / np.mean(centerline.progressive_length))
        for property_name in property_list_local:
            properties[property_name] = scipy.signal.convolve(properties[property_name], window, mode='same') / np.sum(window)

    if compute_diameters:
        property_list_local.remove('major_axis_length')
        property_list_local.remove('minor_axis_length')
        property_list_local.append('RL_diameter')
        property_list_local.append('AP_diameter')
        property_list = property_list_local

    # Display properties on the referential space. Requires intervertebral disks
    if verbose == 2:
        x_increment = 'distance_from_C1'
        if fname_disks_image is None:
            x_increment = 'incremental_length'

        # Display the image and plot all contours found
        fig, axes = plt.subplots(len(property_list_local), sharex=True, sharey=False)
        for k, property_name in enumerate(property_list_local):
            axes[k].plot(properties[x_increment], properties[property_name])
            axes[k].set_ylabel(property_name)

        if fname_disks_image is not None:
            properties['distance_disk_from_C1'] = centerline.distance_from_C1label  # distance between each disk and C1 (or first disk)
            xlabel_disks = [centerline.convert_vertlabel2disklabel[label] for label in properties['distance_disk_from_C1']]
            xtick_disks = [properties['distance_disk_from_C1'][label] for label in properties['distance_disk_from_C1']]
            plt.xticks(xtick_disks, xlabel_disks, rotation=30)
        else:
            axes[-1].set_xlabel('Position along the spinal cord (in mm)')

        plt.show()

    # Removing temporary folder
    os.chdir(curdir)
    shutil.rmtree(path_tmp, ignore_errors=True)

    return property_list, properties


def surface(volume, threshold=0.5, verbose=1):
    verts, faces = measure.marching_cubes(volume, threshold)

    if verbose == 2:
        import visvis as vv
        vv.mesh(np.fliplr(verts), faces)
        vv.use().Run()


def shape_pca(data):
    return


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
