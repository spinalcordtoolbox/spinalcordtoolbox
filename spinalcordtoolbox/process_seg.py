#!/usr/bin/env python
# -*- coding: utf-8
# Functions processing segmentation data

from __future__ import absolute_import

import os, math

import numpy as np
from skimage import measure, filters, transform
from tqdm import tqdm

import sct_utils as sct
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.aggregate_slicewise import Metric
# TODO don't import SCT stuff outside of spinalcordtoolbox/
from spinalcordtoolbox.centerline.core import get_centerline
from msct_types import Centerline

# TODO: only use logging, don't use printing, pass images, not filenames, do imports at beginning of file, no chdir()
# TODO: add degree for poly fitting


def compute_shape(segmentation, algo_fitting='bspline', angle_correction=True, verbose=1):
    """
    Compute morphometric measures of the spinal cord in the transverse (axial) plane from the segmentation.
    The segmentation could be binary or weighted for partial volume [0,1].
    :param segmentation: input segmentation. Could be either an Image or a file name.
    :param algo_fitting:
    :param angle_correction:
    :param verbose:
    :return metrics: Dict of class Metric()
    """
    # List of properties to output (in the right order)
    property_list = ['area',
                     'diameter_AP',
                     'diameter_RL',
                     'eccentricity',
                     'solidity',
                     'orientation',
                     'angle_AP',
                     'angle_RL'
                     ]

    im_seg = Image(segmentation).change_orientation('RPI')

    # Getting image dimensions
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim

    # Extract min and max index in Z direction
    data_seg = im_seg.data
    X, Y, Z = (data_seg > 0).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # Initialize dictionary of property_list, with 1d array of nan (default value if no property for a given slice).
    shape_properties = {key: np.full_like(np.empty(nz), np.nan, dtype=np.double) for key in property_list}

    if angle_correction:
        # compute the spinal cord centerline based on the spinal cord segmentation
        _, arr_ctl, arr_ctl_der = get_centerline(im_seg, algo_fitting=algo_fitting, minmax=False, verbose=verbose)

    # Loop across z and compute shape analysis
    for iz in tqdm(range(min_z_index, max_z_index + 1), unit='iter', unit_scale=False, desc="Compute shape analysis",
                   ascii=True, ncols=80):
        # Extract 2D patch
        current_patch = im_seg.data[:, :, iz]
        if angle_correction:
            # Extract tangent vector to the centerline (i.e. its derivative)
            tangent_vect = np.array([arr_ctl_der[0][iz - min_z_index] * px,
                                     arr_ctl_der[1][iz - min_z_index] * py,
                                     pz])
            # Normalize vector by its L2 norm
            tangent_vect = tangent_vect / np.linalg.norm(tangent_vect)
            # Compute the angle between the centerline and the normal vector to the slice (i.e. u_z)
            v0 = [tangent_vect[0], tangent_vect[2]]
            v1 = [0, 1]
            angle_x_rad = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
            v0 = [tangent_vect[1], tangent_vect[2]]
            v1 = [0, 1]
            angle_y_rad = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
            # Apply affine transformation to account for the angle between the cord centerline and the normal to the patch
            tform = transform.AffineTransform(scale=(np.cos(angle_x_rad), np.cos(angle_y_rad)))
            # TODO: make sure pattern does not go extend outside of image border
            current_patch_scaled = transform.warp(current_patch,
                                                  tform.inverse,
                                                  output_shape=current_patch.shape,
                                                  order=1,
                                                  )
        else:
            current_patch_scaled = current_patch
            angle_x_rad, angle_y_rad = 0.0, 0.0
        # compute shape properties on 2D patch
        shape_property = properties2d(current_patch_scaled, [px, py])
        if shape_property is not None:
            # Add custom fields
            shape_property['angle_AP'] = angle_x_rad * 180.0 / math.pi
            shape_property['angle_RL'] = angle_y_rad * 180.0 / math.pi
            # Loop across properties and assign values for function output
            for property_name in property_list:
                shape_properties[property_name][iz] = shape_property[property_name]
        else:
            sct.log.warning('No properties for slice: '.format([iz]))

        """ DEBUG
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(current_patch_scaled)
        ax.grid()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.savefig('tmp_fig.png')
        """
    metrics = {}
    for key, value in shape_properties.items():
        # Making sure all entries added to metrics have results
        if not value == []:
            metrics[key] = Metric(data=np.array(value), label=key)

    return metrics


def properties2d(image, dim):
    """
    Compute shape property of the input 2D image. Accounts for partial volume information.
    :param image: 2D input image of uint8 type that has a single object, weighted for partial volume.
    :param dim: [px, py]: Physical dimension of the image (in mm). X,Y respectively correspond to AP,RL.
    :return:
    """
    # TODO: first, see where the object is, then crop, then upsample --> faster execution
    upscale = 5  # upscale factor for resampling the input image
    # Oversample image to reach sufficient precision when computing shape metrics on the binary mask
    image_r = transform.pyramid_expand(image, upscale=upscale, sigma=None, order=1)
    # Binarize image using threshold at 0. Necessary input for measure.regionprops
    image_bin = np.array(image_r > 0.5, dtype='uint8')
    # Get all closed binary regions from the image (normally there is only one)
    regions = measure.regionprops(image_bin, intensity_image=image_r)
    # Check number of regions
    if len(regions) == 0:
        sct.log.warning('The slice seems empty.')
        return None
    elif len(regions) > 1:
        sct.log.warning('There is more than one object on this slice.')
        return None
    region = regions[0]
    # Compute area with weighted segmentation and adjust area with physical pixel size
    area = np.sum(image_r) * dim[0] * dim[1] / upscale ** 2
    # Compute ellipse orientation, rotated by 90deg because image axis are inverted, modulo pi, in deg
    orientation = (region.orientation + math.pi / 2 % math.pi) * 180.0 / math.pi
    # Find RL and AP diameter based on major/minor axes and cord orientation=
    [diameter_AP, diameter_RL] = \
        find_AP_and_RL_diameter(region.major_axis_length, region.minor_axis_length, orientation,
                                [i / upscale for i in dim])
    # TODO: compute major_axis_length/minor_axis_length by summing weighted voxels along axis
    # Fill up dictionary
    properties = {'area': area,
                  'diameter_AP': diameter_AP,
                  'diameter_RL': diameter_RL,
                  'centroid': region.centroid,
                  'eccentricity': region.eccentricity,
                  'orientation': orientation,
                  'solidity': region.solidity  # convexity measure
    }

    return properties


def find_AP_and_RL_diameter(major_axis, minor_axis, orientation, dim):
    """
    This script checks the orientation of the and assigns the major/minor axis to the appropriate dimension, right-
    left (RL) or antero-posterior (AP). It also multiplies by the pixel size in mm.
    :param major_axis: major ellipse axis length calculated by regionprops
    :param minor_axis: minor ellipse axis length calculated by regionprops
    :param orientation: orientation in degree
    :param dim: pixel size in mm.
    :return: diameter_AP, diameter_RL
    """
    if -45.0 < orientation < 45.0:
        diameter_AP = minor_axis
        diameter_RL = major_axis
    else:
        diameter_AP = major_axis
        diameter_RL = minor_axis
    # Adjust with pixel size
    diameter_AP *= dim[0]
    diameter_RL *= dim[1]
    return diameter_AP, diameter_RL
