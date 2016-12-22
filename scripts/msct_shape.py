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
from skimage import measure
from skimage import filters


def find_contours(image, threshold=0.5, smooth_sigma=0.0, verbose=1):
    image_input = image
    if smooth_sigma != 0.0:
        image_input = smoothing(image_input, sigma=smooth_sigma, verbose=verbose)
    print np.min(image_input), np.max(image_input)
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


def properties2d(image, verbose=1):
    label_img = measure.label(image)
    regions = measure.regionprops(label_img)
    #areas = [r.area for r in regions]
    #ix = np.argsort(areas)
    if len(regions) != 0:
        sc_region = regions[0]
        sc_properties = {'area': sc_region.area,
                         'bbox': sc_region.bbox,
                         'centroid': sc_region.centroid,
                         'eccentricity': sc_region.eccentricity,
                         'equivalent_diameter': sc_region.equivalent_diameter,
                         'euler_number': sc_region.euler_number,
                         'inertia_tensor': sc_region.inertia_tensor,
                         'inertia_tensor_eigvals': sc_region.inertia_tensor_eigvals,
                         'minor_axis_length': sc_region.minor_axis_length,
                         'major_axis_length': sc_region.major_axis_length,
                         'moments': sc_region.moments,
                         'moments_central': sc_region.moments_central,
                         'orientation': sc_region.orientation,
                         'perimeter': sc_region.perimeter,
                         'ratio_major_minor': sc_region.major_axis_length / sc_region.minor_axis_length,
                         'solidity': sc_region.solidity  # convexity measure
                         }
    else:
        sc_properties = None

    return sc_properties


def z_property(volume, property_list, verbose=1):
    number_of_slices = volume.shape[2]
    properties = {key: [] for key in property_list}
    properties['slice_id'] = []

    for i in range(0, number_of_slices, 1):
        sc_properties = properties2d(volume[:, :, i])
        if sc_properties is not None:
            properties['slice_id'].append(i)
            for property_name in property_list:
                properties[property_name].append(sc_properties[property_name])

    if verbose == 2:
        import matplotlib.pyplot as plt
        # Display the image and plot all contours found
        fig, axes = plt.subplots(len(property_list), sharex=True, sharey=False)
        for k, property_name in enumerate(property_list):
            axes[k].plot(properties['slice_id'], properties[property_name])
            axes[k].set_ylabel(property_name)
        plt.show()

    return properties


def surface(volume, threshold=0.5, verbose=1):
    verts, faces = measure.marching_cubes(volume, threshold)

    if verbose == 2:
        """
        from mayavi import mlab
        mlab.triangular_mesh([vert[0] for vert in verts],
                             [vert[1] for vert in verts],
                             [vert[2] for vert in verts],
                             faces)
        mlab.show()
        """

        import visvis as vv
        vv.mesh(np.fliplr(verts), faces)
        vv.use().Run()


def shape_pca(data):
    return
