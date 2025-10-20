"""
Functions processing segmentation data

Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import os
import math
import platform
import numpy as np
from skimage import measure, transform
import skimage
from scipy.ndimage import map_coordinates, gaussian_filter1d
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.aggregate_slicewise import Metric
from spinalcordtoolbox.centerline.core import get_centerline
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.registration.algorithms import compute_pca, find_angle_hog
from spinalcordtoolbox.utils.shell import parse_num_list_inv
from spinalcordtoolbox.utils.sys import sct_progress_bar

# NB: We use a threshold to check if an array is empty, instead of checking if it's exactly 0. This is because
# resampling can change 0 -> ~0 (e.g. 1e-16). See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3402
NEAR_ZERO_THRESHOLD = 1e-6


def compute_shape(segmentation, image=None, angle_correction=True, centerline_path=None, param_centerline=None,
                  verbose=1, remove_temp_files=1, filter_size=5):
    """
    Compute morphometric measures of the spinal cord in the transverse (axial) plane from the segmentation.
    The segmentation could be binary or weighted for partial volume [0,1].

    :param segmentation: input segmentation. Could be either an Image or a file name.
    :param image: input image. Could be either an Image or a file name.
    :param angle_correction:
    :param centerline_path: path to image file to be used as a centerline for computing angle correction.
    :param param_centerline: see centerline.core.ParamCenterline()
    :param verbose:
    :param remove_temp_files: int: Whether to remove temporary files. 0 = no, 1 = yes.
    :param filter_size: int: size of the gaussian filter for regularization along z for rotation angle. 0: no regularization
    :return metrics: Dict of class Metric(). If a metric cannot be calculated, its value will be nan.
    :return fit_results: class centerline.core.FitResults()
    """
    # List of properties that are always available
    property_list = ['area',
                     'angle_AP',
                     'angle_RL',
                     'diameter_AP',
                     'diameter_AP_ellipse',
                     'diameter_RL',
                     'diameter_RL_ellipse',
                     'eccentricity',
                     'orientation_abs',
                     'orientation',
                     'solidity',
                     'length'
                     ]

    im_seg = Image(segmentation).change_orientation('RPI')
    # Check if the input image is provided (i.e., image is not None)
    if image is not None:
        # HOG-related properties that are only available when image (`sct_process_segmentation -i`) is provided
        # TODO: consider whether to use this workaround or include the columns even when image is not provided and use NaN
        hog_properties = ['centermass_x',
                          'centermass_y',
                          'angle_hog']
        # Add quadrant area properties
        quadrant_keys = [
            'area_quadrant_anterior_left',
            'area_quadrant_anterior_right',
            'area_quadrant_posterior_left',
            'area_quadrant_posterior_right',
        ]
        # Add symmetry properties
        symmetry_keys = [
            'symmetry_dice_RL',
            'symmetry_hausdorff_RL',
            'symmetry_difference_RL',
            'symmetry_dice_AP',
            'symmetry_hausdorff_AP',
            'symmetry_difference_AP',
        ]
        # Add HOG-related properties and symmetry to the property list when image is provided
        property_list = property_list[:1] + hog_properties + quadrant_keys + symmetry_keys + property_list[1:]

        im = Image(image).change_orientation('RPI')
        # Make sure the input image and segmentation have the same dimensions
        if im_seg.dim[:3] != im.dim[:3]:
            raise ValueError(
                f"The input segmentation image ({im_seg.path}) and the input image ({im.path}) do not have the same "
                f"dimensions. Please provide images with the same dimensions."
            )

    # Getting image dimensions. x, y and z respectively correspond to RL, PA and IS.
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    pr = 0.1
    # Resample to isotropic resolution in the axial plane. Use the minimum pixel dimension as target dimension.
    im_segr = resample_nib(im_seg, new_size=[pr, pr, pz], new_size_type='mm', interpolation='linear')
    im_r = resample_nib(im, new_size=[pr, pr, pz], new_size_type='mm', interpolation='linear') if image is not None else None

    # Update dimensions from resampled image.
    nx, ny, nz, nt, px, py, pz, pt = im_segr.dim
    # Extract min and max index in Z direction
    data_seg = im_segr.data
    X, Y, Z = (data_seg > NEAR_ZERO_THRESHOLD).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # Initialize dictionary of property_list, with 1d array of nan (default value if no property for a given slice).
    shape_properties = {key: np.full(nz, np.nan, dtype=np.double) for key in property_list}

    # Initialize lists to store slice indices and angles
    z_indices = []
    angle_hog_values = []
    centermass_values = []
    current_patches = {}

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
                "The provided angle correction centerline does not cover slice(s) "
                f"{parse_num_list_inv(missing_slices)} of the input mask. Please "
                "supply a more extensive '-angle-corr-centerline', or disable angle "
                "correction ('-angle-corr 0')."
            ) from None

    # Loop across z and compute shape analysis
    for iz in sct_progress_bar(range(min_z_index, max_z_index + 1), unit='iter', unit_scale=False, desc="Compute shape analysis",
                               ncols=80):
        # Extract 2D patch
        current_patch = im_segr.data[:, :, iz]
        # Special check for the edge case when segmentation has only a single pixel (e.g., in the lumbar region),
        # in this case we skip the slice as we cannot compute PCA
        if np.count_nonzero(current_patch) <= 1:
            logging.warning(f'Skipping slice {iz} as the segmentation contains only a single pixel.')
            continue

        current_patch_im = im_r.data[:, :, iz] if image is not None else None
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
            if image is not None:
                current_patch_im = current_patch_im.astype(np.float64)
                current_patch_im_scaled = transform.warp(current_patch_im,
                                                         tform.inverse,
                                                         output_shape=current_patch_im.shape,
                                                         order=1,
                                                         )
        else:
            current_patch_scaled = current_patch
            current_patch_im_scaled = current_patch_im if current_patch_im is not None else None
            angle_AP_rad, angle_RL_rad = 0.0, 0.0
        # Store the data for this slice
        z_indices.append(iz)

        # Store basic properties and angles to be used later after regularization
        if image is not None:
            # compute PCA and get center or mass based on segmentation; centermass_src: [RL, AP] (assuming RPI orientation)
            coord_src, pca_src, centermass_src = compute_pca(current_patch_scaled)
            # Finds the angle of the image
            # TODO: explore different sigma values for the HOG method, i.e., the influence how far away pixels will vote for the orientation.
            # TODO: double-check if sigma is in voxel or mm units.
            # TODO: do we want to use the same sigma for all slices? As the spinal cord sizes vary across the z-axis.
            # TODO: Use the angle line found by the HOG method to compute symmetry (based on right and left CSA).
            angle_hog, conf_src = find_angle_hog(current_patch_im_scaled, centermass_src,
                                                 px, py, angle_range=40)    # 40 is taken from registration.algorithms.register2d_centermassrot

            angle_hog_values.append(angle_hog)
            centermass_values.append(centermass_src)
            # Store the patches to use later after regularization
            current_patches[iz] = {
                'patch': current_patch_scaled,
                'angle_AP_rad': angle_AP_rad,
                'angle_RL_rad': angle_RL_rad
            }
        else:
            angle_hog = None
        # Compute shape properties for this slice
        shape_property = _properties2d(current_patch_scaled, [px, py], iz, angle_hog=angle_hog, verbose=verbose)
        if image is None or filter_size < 0:  # TODO and regularization term filter_size < 0
            # If regularization is disabled or no image is provided,
            # loop through stored patches and compute properties the regular way
            if shape_property is not None:
                # Add custom fields
                shape_property['angle_AP'] = angle_AP_rad * 180.0 / math.pi     # convert to degrees
                shape_property['angle_RL'] = angle_RL_rad * 180.0 / math.pi     # convert to degrees
                shape_property['length'] = pz / (np.cos(angle_AP_rad) * np.cos(angle_RL_rad))
                # Loop across properties and assign values for function output

                if image is not None:
                    # Get the index of the current slice in our stored arrays
                    idx = z_indices.index(iz)
                    angle_hog = angle_hog_values[idx]
                    centermass_src = centermass_values[idx]
                    # Add custom fields
                    shape_property['centermass_x'] = centermass_src[0]
                    shape_property['centermass_y'] = centermass_src[1]
                    shape_property['angle_hog'] = -angle_hog * 180.0 / math.pi     # degrees, and change sign to match negative if left rotation

                    # Loop across properties and assign values for function output
                for property_name in property_list:
                    shape_properties[property_name][iz] = shape_property[property_name]
            else:
                logging.warning(f'\nNo properties for slice: {iz}')

    # Apply regularization to HOG angles along the z-axis if filter_size > 0
    # The code snippet below is taken from algorithms.register2d_centermassrot -- maybe it could be extracted into a
    # function and reused
    if image is not None and filter_size > 0 and len(z_indices) > 0:
        # Convert lists to numpy arrays
        z_indices_array = np.array(z_indices)
        angle_hog_array = np.array(angle_hog_values)
        # Apply Gaussian filter to regularize the angles
        angle_hog_regularized = gaussian_filter1d(angle_hog_array, filter_size)
        if verbose == 2:
            plt.figure(figsize=(10, 6))
            plt.plot(z_indices_array, 180 * angle_hog_array / np.pi, 'ob', label='Original HOG angles')
            plt.plot(z_indices_array, 180 * angle_hog_regularized / np.pi, 'r', linewidth=2, label='Regularized HOG angles')
            plt.grid()
            plt.xlabel('z slice')
            plt.ylabel('Angle (deg)')
            plt.title(f"Regularized HOG angle estimation (filter_size: {filter_size})")
            plt.legend()
            fname_out = os.path.join('process_seg_regularize_hog_rotation.png')
            plt.savefig(fname_out, dpi=300)
            plt.close()
            logging.info(f"Saved regularized HOG angles visualization to: {fname_out}")

        # Now compute shape properties using the regularized angles
        for i, iz in enumerate(z_indices_array):
            angle_hog = angle_hog_regularized[i]
            current_patch_scaled = current_patches[iz]['patch']
            angle_AP_rad = current_patches[iz]['angle_AP_rad']
            angle_RL_rad = current_patches[iz]['angle_RL_rad']

            # Get centermass for this slice
            centermass_src = centermass_values[i]

            # Compute shape properties with regularized angle_hog
            shape_property = _properties2d(current_patch_scaled, [px, py], iz, angle_hog=angle_hog, verbose=verbose)

            if shape_property is not None:
                # Add custom fields
                shape_property['centermass_x'] = centermass_src[0]
                shape_property['centermass_y'] = centermass_src[1]
                shape_property['angle_hog'] = -angle_hog * 180.0 / math.pi     # degrees, and change sign to match negative if left rotation
                shape_property['angle_AP'] = angle_AP_rad * 180.0 / math.pi     # convert to degrees
                shape_property['angle_RL'] = angle_RL_rad * 180.0 / math.pi     # convert to degrees
                shape_property['length'] = pz / (np.cos(angle_AP_rad) * np.cos(angle_RL_rad))
                # Loop across properties and assign values for function output
                for property_name in property_list:
                    shape_properties[property_name][iz] = shape_property[property_name]
            else:
                logging.warning(f'\nNo properties for slice: {iz}')
    metrics = {}
    for key, value in shape_properties.items():
        # Making sure all entries added to metrics have results
        value = np.array(value)
        if value.size > 0:
            metrics[key] = Metric(data=value, label=key)

    return metrics, fit_results


def _properties2d(seg, dim, iz, angle_hog=None, verbose=1):
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
        'diameter_RL_ellipse': diameter_RL,
        'diameter_RL': diameter_RL,
        'centroid': region.centroid,        # Why do we store this? It is not used in the code.
        'eccentricity': region.eccentricity,
        'orientation_abs': orientation,     # in degrees
        'orientation': -region.orientation,  # in radians
        'solidity': solidity,  # convexity measure
    }
    # Rotate the segmentation by the orientation to align with AP/RL axes
    seg_crop_r_rotated = _rotate_segmentation_by_angle(seg_crop_r, -region.orientation)

    # Measure diameters along AP and RL axes in the rotated segmentation
    rotated_properties = _measure_rotated_diameters(seg_crop_r, seg_crop_r_rotated, dim, region.orientation,
                                                    iz, properties, verbose)
    # Update the properties dictionary with the rotated properties
    properties.update(rotated_properties)
    properties['orientation'] = -properties['orientation'] * 180.0 / math.pi  # convert to degrees

    if angle_hog is not None:
        # If angle_hog is provided, rotate the segmentation by this angle to align with AP/RL axes, and use for symmetry assessment
        seg_crop_r_rotated_hog = _rotate_segmentation_by_angle(seg_crop_r, angle_hog)
        # Compute quadrant areas and RL and AP symmetry
        quadrant_areas = compute_quadrant_areas(seg_crop_r_rotated_hog, region.centroid, orientation_deg=0,
                                                area=area, diameter_AP=diameter_AP, diameter_RL=diameter_RL,
                                                dim=dim, iz=iz, verbose=verbose)
        # Update the properties dictionary with the rotated properties
        properties.update(quadrant_areas)
        symmetry_metrics = _calculate_symmetry(seg_crop_r_rotated_hog, region.centroid, iz=iz, dim=dim, verbose=verbose)
        properties.update(symmetry_metrics)
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


def _measure_rotated_diameters(seg_crop_r, seg_crop_r_rotated, dim, angle, iz, properties, verbose):
    """
    Measure the AP and RL diameters in the rotated segmentation.
    This function counts the number of pixels along the AP and RL axes in the rotated segmentation and converts them
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
        extent_avg = 30  # extent used for averaging the minimum (to account for noise)
        indices = np.array([i for i in range(rl0_r - extent_avg//2, rl0_r + extent_avg//2 + 1)])
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
            _debug_plotting_hog(angle, ap0_r, ap_diameter, dim, iz, properties, rl0_r, properties["diameter_RL_ellipse"],
                                seg_crop_r_rotated, seg_crop_r, coord_ap)

    return result


def _calculate_symmetry(seg_crop_r_rotated, centroid, dim, iz=None, verbose=1):
    """
    Compute symmetry metrics by flipping the segmentation along the RL and AP axes and comparing with the original.
    Calculates Dice coefficient, symmetric difference, and Hausdorff distance for both axes.
    :param seg_crop_r_rotated: Rotated segmentation (after applying angle) used to measure diameters. seg.shape[0] --> RL; seg.shape[1] --> PA
    :param centroid: (y, x) coordinates of the centroid in the upsampled image space.
    :param dim: [px, py] pixel dimensions in mm.
    :param iz: Optional slice index for debug plotting.
    :param verbose: Verbosity level for debug plotting.
    :return: symmetry_metrics: Dictionary with symmetry metrics for RL and AP axes.
    """
    y0, x0 = centroid
    y0 = int(round(y0))
    x0 = int(round(x0))
    # Create an empty array for flipped version of AP axis
    seg_crop_r_flipped_AP = np.zeros_like(seg_crop_r_rotated)

    # Get bounding box of the segmentation (on the cropped segmentation, not the full one)
    coords = np.argwhere(seg_crop_r_rotated > 0)
    # Center of segmentation along RL axis (x)

    # Flip around segmentation center
    for y, x in coords[:, :2]:
        x_mirror = 2 * x0 - x
        if 0 <= x_mirror < seg_crop_r_rotated.shape[1]:
            seg_crop_r_flipped_AP[y, x_mirror] = seg_crop_r_rotated[y, x]
    # Erase half of the segmentation to avoid bias
    seg_crop_r_flipped_AP[:, :x0] = 0
    seg_crop_r_rotated_cut = np.copy(seg_crop_r_rotated)
    seg_crop_r_rotated_cut[:, :x0] = 0
    # Compute the intersection and union for the Dice coefficient
    intersection_AP = np.sum(seg_crop_r_rotated_cut * seg_crop_r_flipped_AP)
    union_AP = np.sum(seg_crop_r_rotated_cut) + np.sum(seg_crop_r_flipped_AP)

    # Compute the Dice coefficient
    symmetry_dice_AP = 2 * intersection_AP / union_AP if union_AP > 0 else 0
    symmetric_difference_AP = (union_AP - (2*intersection_AP)) * dim[0] * dim[1]

    # Compute Hausdorff distance as additional metric
    hausdorff_distance_AP = skimage.metrics.hausdorff_distance(seg_crop_r_rotated_cut > 0.5, seg_crop_r_flipped_AP > 0.5) * dim[0]
    coords_AP = skimage.metrics.hausdorff_pair(seg_crop_r_rotated_cut > 0.5, seg_crop_r_flipped_AP > 0.5)  # For debug plots

    # Create an empty array for flipped version of RL axis
    seg_crop_r_flipped_RL = np.zeros_like(seg_crop_r_rotated)

    # Flip around segmentation center (RL axis)
    for y, x in coords[:, :2]:
        y_mirror = 2 * y0 - y
        if 0 <= y_mirror < seg_crop_r_rotated.shape[0]:
            seg_crop_r_flipped_RL[y_mirror, x] = seg_crop_r_rotated[y, x]
    seg_crop_r_flipped_RL[:y0, :] = 0
    seg_crop_r_rotated_cut_RL = np.copy(seg_crop_r_rotated)
    seg_crop_r_rotated_cut_RL[:y0, :] = 0
    # Compute the intersection and union for the Dice coefficient
    intersection_RL = np.sum(seg_crop_r_rotated_cut_RL * seg_crop_r_flipped_RL)
    union_RL = np.sum(seg_crop_r_rotated_cut_RL) + np.sum(seg_crop_r_flipped_RL)

    # Compute the Dice coefficient
    symmetry_dice_RL = 2 * intersection_RL / union_RL if union_RL > 0 else 0
    symmetric_difference_RL = (union_RL - (2*intersection_RL)) * dim[0] * dim[1]

    # Compute Hausdorff distance as additional metric
    hausdorff_distance_RL = skimage.metrics.hausdorff_distance(seg_crop_r_rotated_cut_RL > 0.5, seg_crop_r_flipped_RL > 0.5) * dim[0]
    coords_RL = skimage.metrics.hausdorff_pair(seg_crop_r_rotated_cut_RL > 0.5, seg_crop_r_flipped_RL > 0.5)

    symmetry_metrics = {
        'symmetry_dice_RL': symmetry_dice_RL,
        'symmetry_dice_AP': symmetry_dice_AP,
        'symmetry_hausdorff_RL': hausdorff_distance_RL,
        'symmetry_hausdorff_AP': hausdorff_distance_AP,
        'symmetry_difference_RL': symmetric_difference_RL,
        'symmetry_difference_AP': symmetric_difference_AP,
    }

    # Create a debug plot
    if verbose == 2:
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(seg_crop_r_rotated > 0.5, cmap='gray', vmin=0, vmax=0.1, alpha=1)
        plt.imshow(seg_crop_r_rotated_cut_RL > 0.5, cmap='Reds', vmin=0, vmax=0.1, alpha=0.7)
        plt.imshow(seg_crop_r_flipped_RL > 0.5, cmap='Blues', vmin=0, vmax=0.1, alpha=0.6)
        plt.plot(x0, y0, 'go', markersize=5, label='Centroid')
        if coords_RL is not None and len(coords_RL) == 2:
            (y1, x1), (y2, x2) = coords_RL
            ax2 = plt.gca()
            ax2.plot([x1, x2], [y1, y2], 'y-', linewidth=2, label='Hausdorff distance')
            ax2.plot([x1, x2], [y1, y2], 'yo', markersize=5)
        # plt.title('RL dice')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(seg_crop_r_rotated > 0.5, cmap='gray', vmin=0, vmax=0.1, alpha=1)
        plt.imshow(seg_crop_r_rotated_cut > 0.5, cmap='Reds', vmin=0, vmax=0.1, alpha=0.7)
        plt.imshow(seg_crop_r_flipped_AP > 0.5, cmap='Blues', vmin=0, vmax=0.1, alpha=0.4)
        plt.plot(x0, y0, 'go', markersize=5, label='Centroid')

        # Plot Hausdorff pair points and line for AP dice
        if coords_AP is not None and len(coords_AP) == 2:
            (y1, x1), (y2, x2) = coords_AP
            ax2 = plt.gca()
            ax2.plot([x1, x2], [y1, y2], 'y-', linewidth=2, label='Hausdorff distance')
            ax2.plot([x1, x2], [y1, y2], 'yo', markersize=5)
        # plt.title('AP dice')
        plt.axis('off')
        # Move the legend outside of the subplots
        plt.legend(loc='lower center', bbox_to_anchor=(-0.1, -0.1), ncol=2)
        plt.suptitle(
            f'Symmetry Dice RL: {symmetry_dice_RL:.3f}, AP: {symmetry_dice_AP:.3f}\n'
            f'Hausdorff RL (mm): {hausdorff_distance_RL:.3f}, AP: {hausdorff_distance_AP:.3f}\n'
            f'Symmetric diff RL (mm²): {symmetric_difference_RL:.3f}, AP: {symmetric_difference_AP:.3f}'
        )
        if not os.path.exists('debug_figures_symmetry'):
            os.makedirs('debug_figures_symmetry')
        fname_out = os.path.join('debug_figures_symmetry', f'process_seg_symmetry_dice_z{iz:03d}.png')
        plt.savefig(fname_out, dpi=300)
        plt.close()
        logging.info(f"Saved symmetry Dice visualization to: {fname_out}")

    return symmetry_metrics


def compute_quadrant_areas(image_crop_r: np.ndarray, centroid: tuple[float, float], orientation_deg: float,
                           area: float, diameter_AP: float, diameter_RL: float,
                           dim: list[float], iz: int, verbose=1) -> tuple[dict, dict]:
    """
    Compute the cross-sectional area of the four spinal cord quadrants in the axial plane.
    Also calculates the symmetry of the spinal cord in the right-left (RL) and anterior-posterior (AP) directions.
    The function rotates the coordinate system based on the spinal cord orientation and
    partitions the segmentation into four quadrants: posterior right, anterior right,
    posterior left, and anterior left. It then calculates the area of each quadrant in mm².
    :param image_crop_r: 2D upsampled non-binary (due to the angle correction) segmentation mask of the spinal cord.
    :param centroid: (y, x) coordinates of the centroid in the upsampled image space.
    :param orientation_deg: Orientation angle of the spinal cord in degrees (from regionprops).
    :param area: Total area of the spinal cord in mm² (used for symmetry calculations).
    :param diameter_AP: AP diameter of the spinal cord in mm (used for debug plots).
    :param diameter_RL: RL diameter of the spinal cord in mm (used for debug plots).
    :param dim: [px, py] pixel dimensions in mm. X,Y respectively correspond to AP, RL.
    :param iz: Slice index used for filename in debug plot.
    :return: quadrant_areas (dict)
        quadrant_areas is a dictionary with the area in mm² for each quadrant:
                 {
                    'area_quadrant_posterior_right': float,
                    'area_quadrant_anterior_right': float,
                    'area_quadrant_posterior_left': float,
                    'area_quadrant_anterior_left': float
                 }
    """
    y0, x0 = centroid
    orientation_rad = np.radians(orientation_deg)
    rows, cols = image_crop_r.shape
    Y, X = np.mgrid[0:rows, 0:cols]

    # Translate coordinate grid to centroid
    Xc = X - x0
    Yc = Y - y0

    # Rotate coordinates to align with AP/RL axes
    # This rotation is needed to accurately define anatomical quadrants.
    # Without the rotation, quadrants would be based on image axes (top/bottom/left/right) rather than true anatomical
    # orientation (posterior/anterior/left/right) resulting in unprecise area calculations whenever the spinal cord is
    # not perfectly aligned with the image axes.
    # TODO remove this rotation as the segmentation is already rotated
    Xr = Xc * np.cos(-orientation_rad) - Yc * np.sin(-orientation_rad)
    Yr = Xc * np.sin(-orientation_rad) + Yc * np.cos(-orientation_rad)

    # Apply quadrant masks - use the intensity values directly to account for the mask softness
    post_r_mask = (Yr < 0) & (Xr < 0)    # Posterior Right
    ant_r_mask = (Yr < 0) & (Xr >= 0)    # Anterior Right
    post_l_mask = (Yr >= 0) & (Xr < 0)   # Posterior Left
    ant_l_mask = (Yr >= 0) & (Xr >= 0)   # Anterior Left

    # Calculate physical area in mm²
    pixel_area_mm2 = (dim[0] * dim[1])

    # Sum areas for each quadrant - multiply by intensity values to account for the mask softness
    quadrant_areas = {
        'area_quadrant_posterior_right': np.sum(image_crop_r[post_r_mask]) * pixel_area_mm2,
        'area_quadrant_anterior_right': np.sum(image_crop_r[ant_r_mask]) * pixel_area_mm2,
        'area_quadrant_posterior_left': np.sum(image_crop_r[post_l_mask]) * pixel_area_mm2,
        'area_quadrant_anterior_left': np.sum(image_crop_r[ant_l_mask]) * pixel_area_mm2
    }

    # Calculate AP and RL symmetry
    left_area = quadrant_areas.get('area_quadrant_anterior_left', 0) + quadrant_areas.get('area_quadrant_posterior_left', 0)
    right_area = quadrant_areas.get('area_quadrant_anterior_right', 0) + quadrant_areas.get('area_quadrant_posterior_right', 0)
    anterior_area = quadrant_areas.get('area_quadrant_anterior_left', 0) + quadrant_areas.get('area_quadrant_anterior_right', 0)
    posterior_area = quadrant_areas.get('area_quadrant_posterior_left', 0) + quadrant_areas.get('area_quadrant_posterior_right', 0)

    # """"DEBUG
    if verbose == 2:
        def _add_diameter_lines(ax, centroid, diameter_AP, diameter_RL, orientation_rad, dim):
            """
            Helper function to add diameter lines to a matplotlib axis.
            """
            y0, x0 = centroid

            radius_ap = (diameter_AP / dim[0]) * 0.5
            radius_rl = (diameter_RL / dim[1]) * 0.5

            dx_ap = radius_ap * np.cos(orientation_rad)
            dy_ap = radius_ap * np.sin(orientation_rad)
            dx_rl = radius_rl * -np.sin(orientation_rad)
            dy_rl = radius_rl * np.cos(orientation_rad)

            ax.plot([x0 - dx_ap, x0 + dx_ap], [y0 - dy_ap, y0 + dy_ap], 'r--', linewidth=2, label='AP diameter')
            ax.plot([x0 - dx_rl, x0 + dx_rl], [y0 - dy_rl, y0 + dy_rl], 'b--', linewidth=2, label='RL diameter')

            # Add centroid
            ax.plot(x0, y0, '.g', markersize=15)

        def _setup_axis(ax, title, xlabel='y\nPosterior-Anterior (PA)', ylabel='x\nLeft-Right (LR)'):
            """
            Helper function to set up common axis properties.
            """
            ax.grid()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        # Create masks for halves (combining quadrants)
        right_mask = post_r_mask | ant_r_mask  # Right half (posterior + anterior right)
        left_mask = post_l_mask | ant_l_mask   # Left half (posterior + anterior left)
        anterior_mask = ant_r_mask | ant_l_mask  # Anterior half (right + left anterior)
        posterior_mask = post_r_mask | post_l_mask  # Posterior half (right + left posterior)

        # Calculate areas for halves
        right_area = quadrant_areas['area_quadrant_posterior_right'] + quadrant_areas['area_quadrant_anterior_right']
        left_area = quadrant_areas['area_quadrant_posterior_left'] + quadrant_areas['area_quadrant_anterior_left']
        anterior_area = quadrant_areas['area_quadrant_anterior_right'] + quadrant_areas['area_quadrant_anterior_left']
        posterior_area = quadrant_areas['area_quadrant_posterior_right'] + quadrant_areas['area_quadrant_posterior_left']

        # Create figure with 1x3 subplots
        fig = Figure(figsize=(18, 6))
        FigureCanvas(fig)

        # ---------------------------------
        # Plot 1: Quadrants
        # ---------------------------------
        ax1 = fig.add_subplot(1, 3, 1)

        # Plot each quadrant mask with a different color
        ax1.imshow(np.where(post_r_mask, image_crop_r, np.nan), cmap='Reds', vmin=0, vmax=1, alpha=1)
        ax1.imshow(np.where(ant_r_mask, image_crop_r, np.nan), cmap='Blues', vmin=0, vmax=1, alpha=1)
        ax1.imshow(np.where(post_l_mask, image_crop_r, np.nan), cmap='Greens', vmin=0, vmax=1, alpha=1)
        ax1.imshow(np.where(ant_l_mask, image_crop_r, np.nan), cmap='Purples', vmin=0, vmax=1, alpha=1)

        ax1.imshow(image_crop_r > 0.5, cmap='gray', interpolation='nearest', vmin=0, vmax=1, alpha=.4)

        _add_diameter_lines(ax1, centroid, diameter_AP, diameter_RL, orientation_rad, dim, )
        _setup_axis(ax1, 'Quadrants')
        offset = 20  # pixel offset from centroid for annotation placement
        ax1.text(x0 - offset, y0 - offset, f"PR:\n{quadrant_areas['area_quadrant_posterior_right']:.2f} mm²", color='red', 
                 fontsize=10, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax1.text(x0 + offset, y0 - offset, f"AR:\n{quadrant_areas['area_quadrant_anterior_right']:.2f} mm²", color='blue',
                 fontsize=10, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax1.text(x0 - offset, y0 + offset, f"PL:\n{quadrant_areas['area_quadrant_posterior_left']:.2f} mm²", color='green',
                 fontsize=10, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax1.text(x0 + offset, y0 + offset, f"AL:\n{quadrant_areas['area_quadrant_anterior_left']:.2f} mm²", color='purple',
                 fontsize=10, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax1.legend(loc='upper right')

        # ---------------------------------
        # Plot 2: Right-Left Symmetry
        # ---------------------------------
        ax2 = fig.add_subplot(1, 3, 2)

        # Plot each half with a different color
        ax2.imshow(np.where(right_mask, image_crop_r, np.nan), cmap='Reds', vmin=0, vmax=1, alpha=1, label='Right')
        ax2.imshow(np.where(left_mask, image_crop_r, np.nan), cmap='Blues', vmin=0, vmax=1, alpha=1, label='Left')
        ax2.imshow(image_crop_r > 0.5, cmap='gray', interpolation='nearest', vmin=0, vmax=1, alpha=.4)

        _add_diameter_lines(ax2, centroid, diameter_AP, diameter_RL, orientation_rad, dim)
        # _add_ellipse(ax2, centroid, diameter_AP, diameter_RL, orientation_rad, dim, upscale)
        _setup_axis(ax2, 'Right-Left Symmetry')
        ax2.text(x0, y0 - offset, f"Right:\n{right_area:.2f} mm²", color='red', fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax2.text(x0, y0 + offset, f"Left:\n{left_area:.2f} mm²", color='blue', fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # ---------------------------------
        # Plot 3: Anterior-Posterior Symmetry
        # ---------------------------------
        ax3 = fig.add_subplot(1, 3, 3)

        # Plot each half with a different color
        ax3.imshow(np.where(anterior_mask, image_crop_r, np.nan), cmap='Greens', vmin=0, vmax=1, alpha=1, label='Anterior')
        ax3.imshow(np.where(posterior_mask, image_crop_r, np.nan), cmap='Purples', vmin=0, vmax=1, alpha=1, label='Posterior')
        ax3.imshow(image_crop_r > 0.5, cmap='gray', interpolation='nearest', vmin=0, vmax=1, alpha=.4)

        _add_diameter_lines(ax3, centroid, diameter_AP, diameter_RL, orientation_rad, dim)
        # _add_ellipse(ax3, centroid, diameter_AP, diameter_RL, orientation_rad, dim, upscale)
        _setup_axis(ax3, 'Anterior-Posterior Symmetry')
        ax3.text(x0 - offset, y0, f"Posterior:\n{posterior_area:.2f} mm²", color='purple',
                 fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax3.text(x0 + offset, y0, f"Anterior:\n{anterior_area:.2f} mm²", color='green',
                 fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Save figure
        os.makedirs('debug_figures_area_quadrants', exist_ok=True)
        fig.tight_layout()
        fig.savefig(f'debug_figures_area_quadrants/cord_quadrant_tmp_fig_slice_{iz:03d}.png', dpi=150)
        # """

    return quadrant_areas


def _debug_plotting_hog(angle_hog, ap0_r, ap_diameter, dim, iz, properties, rl0_r, rl_diameter,
                        rotated_bin, seg_crop_r, coord_ap):
    """
    """
    def _add_labels(ax):
        """Add A, P, R, L labels"""
        bbox_params = dict(facecolor='black', alpha=1)
        ax.text(ap0_r, seg_crop_r.shape[0] * 0.95, 'L', color='white', fontsize=12, ha='center', va='center',
                bbox=bbox_params, alpha=0.8)
        ax.text(seg_crop_r.shape[1] * 0.95, rl0_r, 'A', color='white', fontsize=12, ha='center', va='center',
                bbox=bbox_params, alpha=0.8)
        ax.text(ap0_r, seg_crop_r.shape[0] * 0.05, 'R', color='white', fontsize=12, ha='center', va='center',
                bbox=bbox_params, alpha=0.8)
        ax.text(seg_crop_r.shape[1] * 0.05, rl0_r, 'P', color='white', fontsize=12, ha='center', va='center',
                bbox=bbox_params, alpha=0.8)

    def _add_ellipse(ax, x0, y0):
        """Add an ellipse to the plot."""
        ellipse = Ellipse(
            (x0, y0),
            width=properties['diameter_AP_ellipse'] / dim[0],
            height=properties['diameter_RL_ellipse'] / dim[1],
            angle=properties['orientation']*180.0/math.pi,
            edgecolor='orange',
            facecolor='none',
            linewidth=2,
            label="Ellipse fitted using skimage.regionprops, angle: {:.2f}".format(-properties['orientation']*180.0/math.pi)
        )
        ax.add_patch(ellipse)

    # Plot the original and rotated segmentation
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    fig = plt.figure(figsize=(6, 8))
    ax1 = fig.add_subplot(111)
    # 1. Original segmentation
    seg_crop_r_bin = np.array(seg_crop_r > 0.5, dtype='uint8')  # binarize the original segmentation
    ax1.imshow(seg_crop_r_bin, cmap='gray', alpha=0.6, label='Original Segmentation')
    # ax1.imshow(seg_crop_r_bin, cmap='Reds', alpha=1, label='Original Segmentation', vmin=0, vmax=1.3)
    # Add ellipse fitted using skimage.regionprops
    _, _, [y0, x0] = compute_pca(seg_crop_r)
    # Center of mass in the original segmentation
    ax1.plot(x0, y0, 'ko', markersize=10, label='Original Segmentation Center of Mass')
    _add_ellipse(ax1, x0, y0)
    # Draw AP and RL axes through the center of mass of the original segmentation
    ax1.arrow(ap0_r, rl0_r, np.sin(angle_hog + (90 * math.pi / 180)) * 25,
              np.cos(angle_hog + (90 * math.pi / 180)) * 25, color='black', width=0.1,
              head_width=1, label=f'HOG angle = {angle_hog * 180 / math.pi:.1f}°')  # convert to degrees
    # Add AP and RL diameters from the original segmentation obtained using skimage.regionprops
    radius_ap = (properties['diameter_AP_ellipse'] / dim[0]) * 0.5
    radius_rl = (properties['diameter_RL_ellipse'] / dim[1]) * 0.5
    dx_ap = radius_ap * np.cos(properties['orientation'])
    dy_ap = radius_ap * np.sin(properties['orientation'])
    dx_rl = radius_rl * -np.sin(properties['orientation'])
    dy_rl = radius_rl * np.cos(properties['orientation'])
    ax1.plot([x0 - dx_ap, x0 + dx_ap], [y0 - dy_ap, y0 + dy_ap], color='blue', linestyle='--', linewidth=2,
             label=f'AP diameter (skimage.regionprops) = {properties["diameter_AP_ellipse"]:.2f} mm')
    ax1.plot([x0 - dx_rl, x0 + dx_rl], [y0 - dy_rl, y0 + dy_rl], color='blue', linestyle='solid', linewidth=2,
             label=f'RL diameter (skimage.regionprops) = {properties["diameter_RL_ellipse"]:.2f} mm')
    # Add A, P, R, L labels
    _add_labels(ax1)

    # 2. Rotated segmentation by angle_hog
    ax1.imshow(rotated_bin, cmap='Reds', alpha=0.8, label='Rotated Segmentation')
    # Center of mass
    ax1.plot(ap0_r, rl0_r, 'bo', markersize=10, label='Rotated Segmentation Center of Mass')
    rotated_bin_bin = np.array(rotated_bin > 0.5, dtype='uint8')  # binarize the rotated segmentation
    right = np.nonzero(rotated_bin_bin[:, ap0_r])[0][0]
    left = np.nonzero(rotated_bin_bin[:, ap0_r])[0][-1]
    if rotated_bin_bin[coord_ap, :].size > 0 and np.any(rotated_bin_bin[coord_ap, :]):
        anterior = np.nonzero(rotated_bin_bin[coord_ap, :])[0][0]
        posterior = np.nonzero(rotated_bin_bin[coord_ap, :])[0][-1]
    else:
        anterior = posterior = np.nan
    
    ax1.plot([anterior, posterior], [coord_ap, coord_ap], color='red', linestyle='--', linewidth=2,
             label=f'AP Diameter (rotated segmentation) = {ap_diameter:.2f} mm, coord_ap={coord_ap}')
    ax1.plot([ap0_r, ap0_r], [left, right], color='red', linestyle='solid', linewidth=2,
             label=f'RL Diameter (rotated segmentation) = {rl_diameter:.2f} mm')

    # Plot horizontal and vertical grid lines
    ax1.grid(which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, framealpha=1.0, fontsize=8)
    ax1.set_title(f'Slice {iz}\nOriginal segmentation and Segmentation rotated by HOG angle')

    plt.tight_layout()
    # plt.show()
    # Save the figure
    if not os.path.exists('debug_figures_diameters'):
        os.makedirs('debug_figures_diameters')
    fname_out = os.path.join('debug_figures_diameters', f'slice_{iz}.png')
    fig.savefig(fname_out, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory
    print(f'Saved debug figure for slice {iz} with segmentation properties to {fname_out}')
