#!/usr/bin/env python
# -*- coding: utf-8
# Functions processing segmentation data

from __future__ import absolute_import

import math
import platform
import numpy as np
from skimage import measure, transform
from tqdm import tqdm
import logging
import nibabel
import pandas as pd
from scipy.ndimage import label, generate_binary_structure

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.aggregate_slicewise import Metric, func_bin, func_std
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.resampling import resample_nib


def compute_shape(segmentation, angle_correction=True, param_centerline=None, verbose=1):
    """
    Compute morphometric measures of the spinal cord in the transverse (axial) plane from the segmentation.
    The segmentation could be binary or weighted for partial volume [0,1].
    :param segmentation: input segmentation. Could be either an Image or a file name.
    :param angle_correction:
    :param param_centerline: see centerline.core.ParamCenterline()
    :param verbose:
    :return metrics: Dict of class Metric(). If a metric cannot be calculated, its value will be nan.
    :return fit_results: class centerline.core.FitResults()
    """
    # List of properties to output (in the right order)
    property_list = ['area',
                     'angle_AP',
                     'angle_RL',
                     'diameter_AP',
                     'diameter_RL',
                     'eccentricity',
                     'orientation',
                     'solidity',
                     'length'
                     ]

    im_seg = Image(segmentation).change_orientation('RPI')
    # Getting image dimensions. x, y and z respectively correspond to RL, PA and IS.
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    pr = min([px, py])
    # Resample to isotropic resolution in the axial plane. Use the minimum pixel dimension as target dimension.
    im_segr = resample_nib(im_seg, new_size=[pr, pr, pz], new_size_type='mm', interpolation='linear')

    # Update dimensions from resampled image.
    nx, ny, nz, nt, px, py, pz, pt = im_segr.dim

    # Extract min and max index in Z direction
    data_seg = im_segr.data
    X, Y, Z = (data_seg > 0).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # Initialize dictionary of property_list, with 1d array of nan (default value if no property for a given slice).
    shape_properties = {key: np.full_like(np.empty(nz), np.nan, dtype=np.double) for key in property_list}

    fit_results = None

    if angle_correction:
        angle_RL_rad, angle_AP_rad, _ = get_angle_correction(im_seg=im_segr)

    # Loop across z and compute shape analysis
    for iz in tqdm(range(min_z_index, max_z_index + 1), unit='iter', unit_scale=False, desc="Compute shape analysis",
                   ascii=True, ncols=80):
        # Extract 2D patch
        current_patch = im_segr.data[:, :, iz]
        if angle_correction:
            # Apply affine transformation to account for the angle between the centerline and the normal to the patch
            tform = transform.AffineTransform(scale=(np.cos(angle_RL_rad[iz - min_z_index]),
                                                     np.cos(angle_AP_rad[iz - min_z_index])))
            # Convert to float64, to avoid problems in image indexation causing issues when applying transform.warp
            current_patch = current_patch.astype(np.float64)
            # TODO: make sure pattern does not go extend outside of image border
            current_patch_scaled = transform.warp(current_patch,
                                                  tform.inverse,
                                                  output_shape=current_patch.shape,
                                                  order=1,
                                                  )
        else:
            current_patch_scaled = current_patch
            angle_AP_rad, angle_RL_rad = 0.0, 0.0
        # compute shape properties on 2D patch
        shape_property = _properties2d(current_patch_scaled, [px, py])
        if shape_property is not None:
            # Add custom fields
            shape_property['angle_AP'] = angle_AP_rad * 180.0 / math.pi
            shape_property['angle_RL'] = angle_RL_rad * 180.0 / math.pi
            shape_property['length'] = pz / (np.cos(angle_AP_rad) * np.cos(angle_RL_rad))
            # Loop across properties and assign values for function output
            for property_name in property_list:
                shape_properties[property_name][iz] = shape_property[property_name]
        else:
            logging.warning('\nNo properties for slice: {}'.format(iz))

        """ DEBUG
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(current_patch_scaled)
        ax.grid()
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        fig.savefig('tmp_fig.png')
        """
    metrics = {}
    for key, value in shape_properties.items():
        # Making sure all entries added to metrics have results
        if not value == []:
            metrics[key] = Metric(data=np.array(value), label=key)

    return metrics, fit_results


def _properties2d(image, dim):
    """
    Compute shape property of the input 2D image. Accounts for partial volume information.
    :param image: 2D input image in uint8 or float (weighted for partial volume) that has a single object.
    :param dim: [px, py]: Physical dimension of the image (in mm). X,Y respectively correspond to AP,RL.
    :return:
    """
    upscale = 5  # upscale factor for resampling the input image (for better precision)
    pad = 3  # padding used for cropping
    # Check if slice is empty
    if not image.any():
        logging.debug('The slice is empty.')
        return None
    # Normalize between 0 and 1 (also check if slice is empty)
    image_norm = (image - image.min()) / (image.max() - image.min())
    # Convert to float64
    image_norm = image_norm.astype(np.float64)
    # Binarize image using threshold at 0. Necessary input for measure.regionprops
    image_bin = np.array(image_norm > 0.5, dtype='uint8')
    # Get all closed binary regions from the image (normally there is only one)
    regions = measure.regionprops(image_bin, intensity_image=image_norm)
    # Check number of regions
    if len(regions) > 1:
        logging.debug('There is more than one object on this slice.')
        return None
    region = regions[0]
    # Get bounding box of the object
    minx, miny, maxx, maxy = region.bbox
    # Use those bounding box coordinates to crop the image (for faster processing)
    image_crop = image_norm[np.clip(minx-pad, 0, image_bin.shape[0]): np.clip(maxx+pad, 0, image_bin.shape[0]),
                 np.clip(miny-pad, 0, image_bin.shape[1]): np.clip(maxy+pad, 0, image_bin.shape[1])]
    # Oversample image to reach sufficient precision when computing shape metrics on the binary mask
    image_crop_r = transform.pyramid_expand(image_crop, upscale=upscale, sigma=None, order=1)
    # Binarize image using threshold at 0. Necessary input for measure.regionprops
    image_crop_r_bin = np.array(image_crop_r > 0.5, dtype='uint8')
    # Get all closed binary regions from the image (normally there is only one)
    regions = measure.regionprops(image_crop_r_bin, intensity_image=image_crop_r)
    region = regions[0]
    # Compute area with weighted segmentation and adjust area with physical pixel size
    area = np.sum(image_crop_r) * dim[0] * dim[1] / upscale ** 2
    # Compute ellipse orientation, modulo pi, in deg, and between [0, 90]
    orientation = fix_orientation(region.orientation)
    # Find RL and AP diameter based on major/minor axes and cord orientation=
    [diameter_AP, diameter_RL] = \
        _find_AP_and_RL_diameter(region.major_axis_length, region.minor_axis_length, orientation,
                                 [i / upscale for i in dim])
    # TODO: compute major_axis_length/minor_axis_length by summing weighted voxels along axis
    # Deal with https://github.com/neuropoly/spinalcordtoolbox/issues/2307
    if any(x in platform.platform() for x in ['Darwin-15', 'Darwin-16']):
        solidity = np.nan
    else:
        solidity = region.solidity
    # Fill up dictionary
    properties = {'area': area,
                  'diameter_AP': diameter_AP,
                  'diameter_RL': diameter_RL,
                  'centroid': region.centroid,
                  'eccentricity': region.eccentricity,
                  'orientation': orientation,
                  'solidity': solidity  # convexity measure
    }

    return properties


def fix_orientation(orientation):
    """Re-map orientation from skimage.regionprops from [-pi/2,pi/2] to [0,90] and rotate by 90deg because image axis
    are inverted"""
    orientation_new = orientation * 180.0 / math.pi
    if 360 <= abs(orientation_new) <= 540:
        orientation_new = 540 - abs(orientation_new)
    if 180 <= abs(orientation_new) <= 360:
        orientation_new = 360 - abs(orientation_new)
    if 90 <= abs(orientation_new) <= 180:
        orientation_new = 180 - abs(orientation_new)
    return abs(orientation_new)


def _find_AP_and_RL_diameter(major_axis, minor_axis, orientation, dim):
    """
    This script checks the orientation of the and assigns the major/minor axis to the appropriate dimension, right-
    left (RL) or antero-posterior (AP). It also multiplies by the pixel size in mm.
    :param major_axis: major ellipse axis length calculated by regionprops
    :param minor_axis: minor ellipse axis length calculated by regionprops
    :param orientation: orientation in degree. Ranges between [0, 90]
    :param dim: pixel size in mm.
    :return: diameter_AP, diameter_RL
    """
    if 0 <= orientation < 45.0:
        diameter_AP = minor_axis
        diameter_RL = major_axis
    else:
        diameter_AP = major_axis
        diameter_RL = minor_axis
    # Adjust with pixel size
    diameter_AP *= dim[0]
    diameter_RL *= dim[1]
    return diameter_AP, diameter_RL


def get_angle_correction(im_seg):
    """Measure spinal cord angle with respect to slice.

    Compute the angle about RL axis between the centerline and the normal vector to the slice.
    Same for the AP and IS axes.

    :param im_seg: Spinal cord segmentation image.
    :return: three numpy arrays, angles in radians, np.nan when no segmentation.
    """
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim

    # Get range of slices where the segmentation is present
    _, _, Z = (im_seg.data > 0).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # fit centerline, smooth it and return the first derivative (in physical space)
    _, arr_ctl, arr_ctl_der, _ = get_centerline(im_seg, param=ParamCenterline(), verbose=1)
    x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = arr_ctl_der

    # Init angles in the three dimensions, shape: (nz,), init with np.nan
    angle_RL = np.full_like(np.empty(nz), np.nan, dtype=np.double)
    angle_AP = np.full_like(np.empty(nz), np.nan, dtype=np.double)
    angle_IS = np.full_like(np.empty(nz), np.nan, dtype=np.double)

    # Loop across slices where segmentation is present
    for iz in range(min_z_index, max_z_index+1):
        # normalize the tangent vector to the centerline (i.e. its derivative)
        vect = np.array([x_centerline_deriv[iz - min_z_index] * px,
                         y_centerline_deriv[iz - min_z_index] * py,
                         pz])
        # Normalize vector by its L2 norm
        norm = np.linalg.norm(vect)
        tangent_vect = vect / norm

        # Compute the angle about RL axis between the centerline and the normal vector to the slice
        v0 = [tangent_vect[1], tangent_vect[2]]
        v1 = [0, 1]
        angle_RL[iz] = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))

        # Compute the angle about AP axis between the centerline and the normal vector to the slice
        v0 = [tangent_vect[0], tangent_vect[2]]
        v1 = [0, 1]
        angle_AP[iz] = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))

        # Compute the angle between the normal vector of the plane and the vector z
        angle_IS[iz] = np.arccos(np.vdot(tangent_vect, np.array([0, 0, 1])))

    return angle_RL, angle_AP, angle_IS


def analyze_binary_objects(fname_mask, fname_voi=None, fname_ref=None, path_template=None, path_ofolder="./analyze_lesion", verbose=1):
    """
    Analyze lesions or tumours by computing statistics on binary mask.

    :param fname_mask: Lesion binary mask filename.
    :param fname_voi: Volume of interest binary mask filename.
    :param fname_ref: Image filename from which to extract average values within lesions.
    :param path_template: Path to folder containing the atlas/template registered to the anatomical image.
    :param path_ofolder: Output folder.
    :param verbose: Verbose.
    :return: XX
    """
    im_mask = Image(fname_mask)

    # Check if input data is binary and not empty
    if not im_mask.is_binary() or im_mask.is_empty():
        logging.error("ERROR input file %s is not binary file or is empty".format(fname_mask))

    # re-orient image to RPI
    logging.info("Reorient the image to RPI, if necessary...")
    original_orientation = im_mask.orientation
    im_mask.change_orientation('RPI')

    logging.info("Label the different lesions on the input mask...")
    # Binary structure
    bin_struct = generate_binary_structure(3, 2)  # 18-connectivity
    # Label connected regions of the masked image
    im_labeled = im_mask.copy()
    im_labeled.data, num_lesion = label(im_mask.data.copy(), structure=bin_struct)

    columns_result = ["lesion_id", "volume", "length_IS", "max_equivalent_diameter"]

    if fname_voi is not None:
        im_voi = Image(fname_voi)
        im_voi.change_orientation('RPI')
        logging.info("Compute angle correction for CSA based on cord angle with respect to slice...")
        # Spinal cord angle with respect to I-S slice
        _, _, angle_correction_IS_rad = get_angle_correction(im_seg=im_voi.copy())
        # Convert angles in degrees
        angle_correction = np.degrees(angle_correction_IS_rad)
        # TODO: convert angles to degrees (math.degrees)
    else:
        angle_correction = np.full_like(np.empty(im_mask.dim[2]), 1.0, dtype=np.double)

    if fname_ref is not None:
        im_ref = Image(fname_ref).change_orientation('RPI')
        logging.info("Load raw data...")
        columns_result += ['mean_intensity', 'std_intensity']
    else:
        im_ref = None

    # Indexes of I-S slices where VOI is present
    z_voi = [z for z in list(angle_correction) if z != np.nan]

    # Initialise result dictionary
    df_results = pd.DataFrame.from_dict(columns=columns_result)

    # Voxel size
    px, py, pz = im_labeled.dim[4:7]

    # Compute metrics for each lesion
    for lesion_id in range(1, num_lesion + 1):
        im_lesion_id = im_labeled.copy()
        data_lesion_id = (im_lesion_id.data == lesion_id).astype(np.int)

        # Indexes of I-S slices where lesion_id is present
        z_lesion_cur = [z for z in z_voi if np.any(data_lesion_id[:, :, z])]

        # Volume
        volume = np.sum(data_lesion_id) * px * py * pz
        # Inf-Sup length
        length_is_zz = [np.cos(angle_correction[zz]) * pz[2] for zz in z_lesion_cur]
        length_is = np.sum(length_is_zz)
        # Maximum equivalent diameter
        list_area = [np.sum(data_lesion_id[:, :, zz]) * np.cos(angle_correction[zz]) * px * py for zz in z_lesion_cur]
        max_equiv_diameter = 2 * np.sqrt(max(list_area) / (4 * np.pi))

        # Info
        logging.info('\tVolume: ' + str(np.round(volume, 2)) + ' mm3')
        logging.info('\t(S-I) length: ' + str(np.round(length_is, 2)) + ' mm')
        logging.info('\tMax. equivalent diameter : ' + str(np.round(max_equiv_diameter, 2)) + ' mm')

        list_results = [lesion_id, volume, length_is, max_equiv_diameter]

        # Intensity
        if im_ref is not None:
            avg, _ = func_bin(im_ref.data, data_lesion_id)
            std, _ = func_std(im_ref.data, data_lesion_id)
            logging.info('\tMean+/-Std values in Reference image:' + str(np.round(avg, 2))
                         + '+/-' + str(np.round(std, 2)))
            list_results += [avg, std]

        df_results.loc[len(df_results)] = list_results

        del im_lesion_id
