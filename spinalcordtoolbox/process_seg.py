"""
Functions processing segmentation data

Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import math
import platform
import numpy as np
from skimage import measure, transform
from scipy.ndimage import map_coordinates
import logging

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.aggregate_slicewise import Metric
from spinalcordtoolbox.centerline.core import get_centerline
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.registration.algorithms import compute_pca
from spinalcordtoolbox.utils.shell import parse_num_list_inv
from spinalcordtoolbox.utils.sys import sct_progress_bar

from spinalcordtoolbox.process_seg_debug import create_ap_diameter_plots

# NB: We use a threshold to check if an array is empty, instead of checking if it's exactly 0. This is because
# resampling can change 0 -> ~0 (e.g. 1e-16). See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3402
NEAR_ZERO_THRESHOLD = 1e-6
KEYS_DEFAULT = ['area',
                'angle_AP',
                'angle_RL',
                'diameter_AP',
                'diameter_AP_ellipse',
                'diameter_RL',
                'eccentricity',
                'orientation',
                'solidity',
                'length'
                ]


def compute_shape(segmentation, angle_correction=True, centerline_path=None, param_centerline=None,
                  verbose=1, remove_temp_files=1):
    """
    Compute morphometric measures of the spinal cord in the transverse (axial) plane from the segmentation.
    The segmentation could be binary or weighted for partial volume [0,1].

    :param segmentation: input segmentation. Could be either an Image or a file name.
    :param angle_correction:
    :param centerline_path: path to image file to be used as a centerline for computing angle correction.
    :param param_centerline: see centerline.core.ParamCenterline()
    :param verbose:
    :param remove_temp_files: int: Whether to remove temporary files. 0 = no, 1 = yes.
    :return metrics: Dict of class Metric(). If a metric cannot be calculated, its value will be nan.
    :return fit_results: class centerline.core.FitResults()
    """
    # List of properties that are always available
    property_list = KEYS_DEFAULT

    im_seg = Image(segmentation).change_orientation('RPI')

    # Getting image dimensions. x, y and z respectively correspond to RL, PA and IS.
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    pr = 0.1   # we use a fixed value to be independent from the input image resolution
    # Resample to isotropic resolution in the axial plane. Use the minimum pixel dimension as target dimension.
    im_segr = resample_nib(im_seg, new_size=[pr, pr, pz], new_size_type='mm', interpolation='linear')

    # Update dimensions from resampled image.
    nx, ny, nz, nt, px, py, pz, pt = im_segr.dim
    # Extract min and max index in Z direction
    data_seg = im_segr.data
    X, Y, Z = (data_seg > NEAR_ZERO_THRESHOLD).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # Initialize dictionary of property_list, with 1d array of nan (default value if no property for a given slice).
    shape_properties = {key: np.full(nz, np.nan, dtype=np.double) for key in property_list}

    fit_results = None

    if angle_correction:
        # allow the centerline image to be bypassed (in case `im_seg` is irregularly shaped, e.g. GM/WM)
        if centerline_path:
            im_centerline = Image(centerline_path).change_orientation('RPI')
            im_centerline_r = resample_nib(im_centerline, new_size=[pr, pr, pz], new_size_type='mm',
                                           interpolation='linear')
        else:
            im_centerline_r = im_segr
        # compute the spinal cord centerline based on the spinal cord segmentation
        _, arr_ctl, arr_ctl_der, fit_results = get_centerline(im_centerline_r, param=param_centerline, verbose=verbose,
                                                              remove_temp_files=remove_temp_files)
        # the third column of `arr_ctl` contains the integer slice numbers, and the first two
        # columns of `arr_ctl_der` contain the x and y components of the centerline derivative
        deriv = {int(z_ref): arr_ctl_der[:2, index] for index, z_ref in enumerate(arr_ctl[2])}

        # check for slices in the input mask not covered by the centerline
        missing_slices = sorted(set(range(min_z_index, max_z_index + 1)).difference(deriv.keys()))
        if missing_slices:
            raise ValueError(
                f"The provided centerline does not cover slice(s) {parse_num_list_inv(missing_slices)} "
                "of the input mask. Please supply a '-centerline' covering all the slices, or disable angle "
                "correction ('-angle-corr 0')."
            ) from None

    # Loop across z and compute shape analysis
    for iz in sct_progress_bar(range(min_z_index, max_z_index + 1), unit='iter', unit_scale=False, desc="Compute shape analysis",
                               ncols=80):
        # Extract 2D patch
        current_patch = im_segr.data[:, :, iz]
        if angle_correction:
            # Extract tangent vector to the centerline (i.e. its derivative)
            tangent_vect = np.array([deriv[iz][0] * px, deriv[iz][1] * py, pz])
            # Compute the angle about AP axis between the centerline and the normal vector to the slice
            angle_AP_rad = math.atan2(tangent_vect[0], tangent_vect[2])
            # Compute the angle about RL axis between the centerline and the normal vector to the slice
            angle_RL_rad = math.atan2(tangent_vect[1], tangent_vect[2])
            # Apply affine transformation to account for the angle between the centerline and the normal to the patch
            tform = transform.AffineTransform(scale=(np.cos(angle_RL_rad), np.cos(angle_AP_rad)))
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

        # Compute shape properties for this slice
        shape_property = _properties2d(current_patch_scaled, [px, py], iz, verbose=verbose)
        if shape_property is not None:
            # Add custom fields
            shape_property['angle_AP'] = angle_AP_rad * 180.0 / math.pi     # convert to degrees
            shape_property['angle_RL'] = angle_RL_rad * 180.0 / math.pi     # convert to degrees
            shape_property['length'] = pz / (np.cos(angle_AP_rad) * np.cos(angle_RL_rad))
            # Loop across properties and assign values for function output
            for property_name in property_list:
                shape_properties[property_name][iz] = shape_property.get(property_name, np.nan)

    metrics = {}
    for key, value in shape_properties.items():
        # Making sure all entries added to metrics have results
        value = np.array(value)
        if value.size > 0:
            metrics[key] = Metric(data=value, label=key)

    return metrics, fit_results


def _properties2d(seg, dim, iz, verbose=1):
    """
    Compute shape property of the input 2D segmentation. Accounts for partial volume information.
    :param seg: 2D input segmentation in uint8 or float (weighted for partial volume) that has a single object. seg.shape[0] --> RL; seg.shape[1] --> PA
    :param dim: [px, py]: Physical dimension of the segmentation (in mm). X,Y respectively correspond to RL,PA.
    :param iz: Integer slice number (z index) of the segmentation. Used for plotting purposes.
    :param angle_hog: Optional angle in radians to rotate the segmentation to align with AP/RL axes.
    :return:
    """
    pad = 3  # padding used for cropping
    # Check if slice is empty
    if np.all(seg < NEAR_ZERO_THRESHOLD):
        logging.debug('The slice is empty.')
        return None
    # Normalize between 0 and 1 (also check if slice is empty)
    seg_norm = (seg - seg.min()) / (seg.max() - seg.min())
    # Convert to float64
    seg_norm = seg_norm.astype(np.float64)
    # Binarize segmentation using threshold at 0. Necessary input for measure.regionprops
    # Note: even when the input segmentation is binary, it might be soft now due to the angle correction
    seg_bin = np.array(seg_norm > 0.5, dtype='uint8')
    # Get all closed binary regions from the segmentation (normally there is only one)
    regions = measure.regionprops(seg_bin, intensity_image=seg_norm)
    # Check number of regions
    if len(regions) > 1:
        logging.debug('There is more than one object on this slice.')
        return None
    region = regions[0]
    # Get bounding box of the object
    minx, miny, maxx, maxy = region.bbox
    # Use those bounding box coordinates to crop the segmentation (for faster processing)
    seg_crop = seg_norm[np.clip(minx-pad, 0, seg_bin.shape[0]): np.clip(maxx+pad, 0, seg_bin.shape[0]),
                        np.clip(miny-pad, 0, seg_bin.shape[1]): np.clip(maxy+pad, 0, seg_bin.shape[1])]
    seg_crop_r = seg_crop
    # Binarize segmentation using threshold at 0.5 Necessary input for measure.regionprops
    seg_crop_r_bin = np.array(seg_crop_r > 0.5, dtype='uint8')
    # Get all closed binary regions from the segmentation (normally there is only one)
    regions = measure.regionprops(seg_crop_r_bin, intensity_image=seg_crop_r)
    region = regions[0]
    # Compute area with weighted segmentation and adjust area with physical pixel size
    area = np.sum(seg_crop_r) * dim[0] * dim[1]
    # Compute ellipse orientation, modulo pi, in deg, and between [0, 90]
    orientation = fix_orientation(region.orientation)
    # Find RL and AP diameter based on major/minor axes and cord orientation # TODO: remove
    [diameter_AP, diameter_RL] = \
        _find_AP_and_RL_diameter(region.major_axis_length, region.minor_axis_length, orientation,
                                 dim)
    # Deal with https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2307
    if any(x in platform.platform() for x in ['Darwin-15', 'Darwin-16']):
        solidity = np.nan
    else:
        solidity = region.solidity

    # Fill up dictionary
    properties = {
        'area': area,
        'diameter_AP_ellipse': diameter_AP,
        'diameter_RL': diameter_RL,
        'centroid': region.centroid,        # Why do we store this? It is not used in the code.
        'eccentricity': region.eccentricity,
        'orientation': -region.orientation,  # in radians
        'solidity': solidity,  # convexity measure
    }
    # Rotate the segmentation by the orientation to align with AP/RL axes
    seg_crop_r_rotated = _rotate_segmentation_by_angle(seg_crop_r, -region.orientation)

    # Measure diameters along AP and RL axes in the rotated segmentation
    rotated_properties = _measure_ap_diameter(seg_crop_r, seg_crop_r_rotated, dim, region.orientation,
                                              iz, properties, verbose)
    # Update the properties dictionary with the rotated properties
    properties.update(rotated_properties)
    properties['orientation'] = -properties['orientation'] * 180.0 / math.pi  # convert to degrees

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


def _rotate_segmentation_by_angle(seg_crop_r, angle):
    """
    Rotate the segmentation by the angle (HOG angle found from the image) to align with AP/RL axes.

    :param seg_crop_r: 2D input segmentation
    :param angle: Rotation angle in radians (positive values correspond to counter-clockwise rotation)
    :return seg_crop_r_rotated: Rotated segmentation
    """
    # get center of mass (which is computed in the PCA function)
    _, _, [y0, x0] = compute_pca(seg_crop_r)        # same as `y0, x0 = region.centroid`
    # Create coordinate grid
    rows, cols = seg_crop_r.shape
    Y, X = np.mgrid[0:rows, 0:cols]
    # Center the coordinates
    Xc = X - x0
    Yc = Y - y0
    # Rotate coordinates
    # If the angle is negative, the rotation is clockwise (from Left to Right).
    # If it is positive, the rotation is counter-clockwise (from Right to Left).
    Xr = Xc * np.cos(angle) - Yc * np.sin(angle)
    Yr = Xc * np.sin(angle) + Yc * np.cos(angle)
    # Shift the rotated coordinates back to their original position in the image. This ensures that the rotated
    # segmentation is positioned correctly in the output image, with the rotation happening around the center of the
    # object rather than around the origin of the coordinate system.
    Xr = Xr + x0
    Yr = Yr + y0

    # Create coordinate mapping for interpolation
    coords = np.column_stack([np.ravel(Yr), np.ravel(Xr)])

    # Apply transformation
    seg_crop_r_rotated = map_coordinates(seg_crop_r,
                                         [coords[:, 0], coords[:, 1]],
                                         order=1).reshape(seg_crop_r.shape)      # order of the spline interpolation --> order=1: linear interpolation

    return seg_crop_r_rotated


def _measure_ap_diameter(seg_crop_r, seg_crop_r_rotated, dim, angle, iz, properties, verbose):
    """
    Measure the AP diameter in the rotated segmentation.
    This function counts the number of pixels along the AP axis in the rotated segmentation and converts them
    to physical dimensions using the provided pixel size.

    :param seg_crop_r: Original cropped segmentation (used only for plotting).
    :param seg_crop_r_rotated: Rotated segmentation (after applying angle) used to measure diameters. seg.shape[0] --> RL; seg.shape[1] --> PA
    :param dim: Physical dimensions of the segmentation (in mm). X,Y respectively correspond to RL,PA.
    :param angle: Rotation angle in radians (HOG angle found from the image)
    :param iz: Integer slice number (z index) of the segmentation. Used for plotting purposes.
    :param properties: Dictionary containing the properties of the original (not-rotated) segmentation. Used for plotting purposes.
    :param verbose: Verbosity level. If 2, debug figures are created.
    :return result: Dictionary containing the measured diameters and pixel counts along AP and RL axes.
    """
    # Get center of mass (which is computed in the PCA function); centermass_src: [RL, AP] (assuming RPI orientation)
    # Note: I'm using [rl0, ap0] instead of [y0, x0] to make it easier to track the axes as numpy handle them in a bit unintuitive way :-D
    rotated_bin = np.array(seg_crop_r_rotated > 0.5, dtype='uint8')  # binarize the rotated segmentation for PCA
    # Check if segmentation is empty after rotation
    if np.all(rotated_bin < NEAR_ZERO_THRESHOLD):
        logging.debug('The rotated segmentation is empty.')
        return {
            'ap_pixel_count': np.nan,
            'diameter_AP': np.nan,
        }
    else:
        _, _, [rl0, ap0] = compute_pca(rotated_bin)    # same as `y0, x0 = region.centroid`
        rl0_r, ap0_r = round(rl0), round(ap0)

        # Note: seg_crop_r_rotated is soft (due to the rotation) so we sum its values to account for softness
        # Sum non-zero pixels along AP axis, i.e., the number of pixels in the row corresponding to the center of mass along the RL axis
        # Compute AP diameter average acrross 3 mm extent centered at rl0_r

        # Use centroid for AP diameter
        extent_avg = 30  # extent of 3 mm (because seg was resampled to 0.1mm) used for averaging the minimum (to account for noise)
        indices = np.array([i for i in range(rl0_r - extent_avg//2, rl0_r + extent_avg//2)])
        # Ensure indices are within the segmentation bounds
        indices = indices[(indices >= 0) & (indices < seg_crop_r_rotated.shape[0])]
        ap_pixels = np.sum(seg_crop_r_rotated[indices, :], axis=1).mean()
        coord_ap = rl0_r

        ap_diameter = ap_pixels * dim[1]

        # Store all the rotated properties
        result = {
            'ap_pixel_count': ap_pixels,
            'diameter_AP': ap_diameter,
        }

        # Debug plotting
        if verbose == 2:
            create_ap_diameter_plots(angle, ap0_r, ap_diameter, dim, iz, properties, rl0_r, properties["diameter_RL"],
                                     seg_crop_r_rotated, seg_crop_r, coord_ap)

    return result
