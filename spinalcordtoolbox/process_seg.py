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
from scipy.ndimage import map_coordinates, gaussian_filter1d
import logging
import matplotlib.pyplot as plt

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
                     'diameter_RL',
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
        hog_properties = ['diameter_AP',
                          'diameter_RL',
                          'centermass_x',
                          'centermass_y',
                          'angle_hog']
        # Add HOG-related properties to the property list when image is provided
        property_list = property_list[:1] + hog_properties + property_list[1:]
        im = Image(image).change_orientation('RPI')
        # Make sure the input image and segmentation have the same dimensions
        if im_seg.dim[:3] != im.dim[:3]:
            raise ValueError(
                f"The input segmentation image ({im_seg.path}) and the input image ({im.path}) do not have the same "
                f"dimensions. Please provide images with the same dimensions."
            )

    # Getting image dimensions. x, y and z respectively correspond to RL, PA and IS.
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    pr = min([px, py])
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

            # Store the data for this slice
            z_indices.append(iz)
            angle_hog_values.append(angle_hog)
            centermass_values.append(centermass_src)
            # Store the patches to use later after regularization
            current_patches[iz] = {
                'patch': current_patch_scaled,
                'angle_AP_rad': angle_AP_rad,
                'angle_RL_rad': angle_RL_rad
            }

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
            if not os.path.exists('hog_debug_figures'):
                os.makedirs('hog_debug_figures')
            fname_out = os.path.join('hog_debug_figures', 'process_seg_regularize_hog_rotation.png')
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
                shape_property['angle_hog'] = angle_hog     # in radians
                shape_property['angle_AP'] = angle_AP_rad * 180.0 / math.pi     # convert to degrees
                shape_property['angle_RL'] = angle_RL_rad * 180.0 / math.pi     # convert to degrees
                shape_property['length'] = pz / (np.cos(angle_AP_rad) * np.cos(angle_RL_rad))

                # Loop across properties and assign values for function output
                for property_name in property_list:
                    shape_properties[property_name][iz] = shape_property[property_name]
            else:
                logging.warning(f'\nNo properties for slice: {iz}')
    else:
        # If regularization is disabled or no image is provided,
        # loop through stored patches and compute properties the regular way
        for iz in z_indices:
            current_patch_scaled = current_patches[iz]['patch']
            angle_AP_rad = current_patches[iz]['angle_AP_rad']
            angle_RL_rad = current_patches[iz]['angle_RL_rad']

            if image is not None:
                # Get the index of the current slice in our stored arrays
                idx = z_indices.index(iz)
                angle_hog = angle_hog_values[idx]
                centermass_src = centermass_values[idx]

                # Compute shape properties with original angle_hog
                shape_property = _properties2d(current_patch_scaled, [px, py], iz, angle_hog=angle_hog, verbose=verbose)

                if shape_property is not None:
                    # Add custom fields
                    shape_property['centermass_x'] = centermass_src[0]
                    shape_property['centermass_y'] = centermass_src[1]
                    shape_property['angle_hog'] = angle_hog     # in radians
                    shape_property['angle_AP'] = angle_AP_rad * 180.0 / math.pi     # convert to degrees
                    shape_property['angle_RL'] = angle_RL_rad * 180.0 / math.pi     # convert to degrees
                    shape_property['length'] = pz / (np.cos(angle_AP_rad) * np.cos(angle_RL_rad))

                    # Loop across properties and assign values for function output
                    for property_name in property_list:
                        shape_properties[property_name][iz] = shape_property[property_name]
                else:
                    logging.warning(f'\nNo properties for slice: {iz}')
            else:
                # If image is None, don't pass angle_hog
                shape_property = _properties2d(current_patch_scaled, [px, py], iz, verbose=verbose)

                if shape_property is not None:
                    # Add custom fields
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
    upscale = 5  # upscale factor for resampling the input segmentation (for better precision)
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
    # Oversample segmentation to reach sufficient precision when computing shape metrics on the binary mask
    seg_crop_r = transform.pyramid_expand(seg_crop, upscale=upscale, sigma=None, order=1)
    # Binarize segmentation using threshold at 0. Necessary input for measure.regionprops
    seg_crop_r_bin = np.array(seg_crop_r > 0.5, dtype='uint8')
    # Get all closed binary regions from the segmentation (normally there is only one)
    regions = measure.regionprops(seg_crop_r_bin, intensity_image=seg_crop_r)
    region = regions[0]
    # Compute area with weighted segmentation and adjust area with physical pixel size
    area = np.sum(seg_crop_r) * dim[0] * dim[1] / upscale ** 2
    # Compute ellipse orientation, modulo pi, in deg, and between [0, 90]
    orientation = fix_orientation(region.orientation)
    # Find RL and AP diameter based on major/minor axes and cord orientation
    [diameter_AP, diameter_RL] = \
        _find_AP_and_RL_diameter(region.major_axis_length, region.minor_axis_length, orientation,
                                 [i / upscale for i in dim])
    # TODO: compute major_axis_length/minor_axis_length by summing weighted voxels along axis
    # Deal with https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2307
    if any(x in platform.platform() for x in ['Darwin-15', 'Darwin-16']):
        solidity = np.nan
    else:
        solidity = region.solidity
    # Fill up dictionary
    properties = {
        'area': area,
        'diameter_AP': diameter_AP,
        'diameter_RL': diameter_RL,
        'centroid': region.centroid,        # Why do we store this? It is not used in the code.
        'eccentricity': region.eccentricity,
        'orientation_abs': orientation,     # in degrees
        'orientation': -region.orientation,  # in radians
        'solidity': solidity,  # convexity measure
    }
    print('AP ellipse')
    print(properties['diameter_AP'])
    # Rotate the segmentation by the angle_hog to align with AP/RL axes
    seg_crop_r_rotated = _rotate_segmentation_by_angle(seg_crop_r, region.orientation)

    # Measure diameters along AP and RL axes in the rotated segmentation
    rotated_properties = _measure_rotated_diameters(seg_crop_r, seg_crop_r_rotated, dim, region.orientation, upscale,
                                                    iz, properties, verbose)
    print('AP seg')
    print(rotated_properties['diameter_AP'])
    # Update the properties dictionary with the rotated properties
    properties.update(rotated_properties)

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


def _rotate_segmentation_by_angle(seg_crop_r, angle_hog):
    """
    Rotate the segmentation by the angle (HOG angle found from the image) to align with AP/RL axes.

    :param seg_crop_r: 2D input segmentation
    :param angle_hog: Rotation angle in radians (positive values correspond to counter-clockwise rotation)
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
    Xr = Xc * np.cos(angle_hog) - Yc * np.sin(angle_hog)
    Yr = Xc * np.sin(angle_hog) + Yc * np.cos(angle_hog)
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


def _measure_rotated_diameters(seg_crop_r, seg_crop_r_rotated, dim, angle, upscale, iz, properties, verbose):
    """
    Measure the AP and RL diameters in the rotated segmentation.
    This function counts the number of pixels along the AP and RL axes in the rotated segmentation and converts them
    to physical dimensions using the provided pixel size.

    :param seg_crop_r: Original cropped segmentation (used only for plotting).
    :param seg_crop_r_rotated: Rotated segmentation (after applying angle) used to measure diameters. seg.shape[0] --> RL; seg.shape[1] --> PA
    :param dim: Physical dimensions of the segmentation (in mm). X,Y respectively correspond to RL,PA.
    :param angle: Rotation angle in radians (HOG angle found from the image)
    :param upscale: Upscale factor used for resampling the segmentation
    :param iz: Integer slice number (z index) of the segmentation. Used for plotting purposes.
    :param properties: Dictionary containing the properties of the original (not-rotated) segmentation. Used for plotting purposes.
    :param verbose: Verbosity level. If 2, debug figures are created.
    :return result: Dictionary containing the measured diameters and pixel counts along AP and RL axes.
    """
    # Get center of mass (which is computed in the PCA function); centermass_src: [RL, AP] (assuming RPI orientation)
    # Note: I'm using [rl0, ap0] instead of [y0, x0] to make it easier to track the axes as numpy handle them in a bit unintuitive way :-D
    rotated_bin = np.array(seg_crop_r_rotated > 0.5, dtype='uint8') # binarize the rotated segmentation for PCA
    _, _, [rl0, ap0] = compute_pca(rotated_bin)    # same as `y0, x0 = region.centroid`
    rl0_r, ap0_r = round(rl0), round(ap0)

    # Note: seg_crop_r_rotated is soft (due to the rotation) so we sum its values to account for softness
    # Sum non-zero pixels along AP axis, i.e., the number of pixels in the row corresponding to the center of mass along the RL axis
    # TODO: consider taking into account more than one row
    ap_pixels = np.sum(seg_crop_r_rotated[rl0_r, :])
    # Sum non-zero pixels along RL axis, i.e., the number of pixels in the column corresponding to the center of mass along the AP axis
    rl_pixels = np.sum(seg_crop_r_rotated[:, ap0_r])
    # Convert pixels to physical dimensions
    # TODO: double-check dim[0] and dim[1] correspondence to RL and AP diameters
    rl_diameter = rl_pixels * dim[0] / upscale
    ap_diameter = ap_pixels * dim[1] / upscale

    # Store all the rotated properties
    result = {
        'ap_pixel_count': ap_pixels,
        'rl_pixel_count': rl_pixels,
        'diameter_AP': ap_diameter,
        'diameter_RL': rl_diameter,
    }

    # Debug plotting
    if verbose == 2:
        _debug_plotting_hog(angle, ap0_r, ap_diameter, dim, iz, properties, rl0_r, rl_diameter,
                            rotated_bin, seg_crop_r, upscale)

    return result


def _debug_plotting_hog(angle_hog, ap0_r, ap_diameter, dim, iz, properties, rl0_r, rl_diameter,
                        rotated_bin, seg_crop_r, upscale):
    """
    """
    def _add_labels(ax):
        """Add A, P, R, L labels"""
        bbox_params = dict(facecolor='black', alpha=1)
        ax.text(ap0_r, seg_crop_r.shape[0] * 0.95, 'L', color='white', fontsize=12, ha='center', va='center',
                bbox=bbox_params)
        ax.text(seg_crop_r.shape[1] * 0.95, rl0_r, 'A', color='white', fontsize=12, ha='center', va='center',
                bbox=bbox_params)
        ax.text(ap0_r, seg_crop_r.shape[0] * 0.05, 'R', color='white', fontsize=12, ha='center', va='center',
                bbox=bbox_params)
        ax.text(seg_crop_r.shape[1] * 0.05, rl0_r, 'P', color='white', fontsize=12, ha='center', va='center',
                bbox=bbox_params)

    def _add_ellipse(ax, x0, y0):
        """Add an ellipse to the plot."""
        ellipse = Ellipse(
            (x0, y0),
            width=properties['diameter_AP'] * upscale / dim[0],
            height=properties['diameter_RL'] * upscale / dim[1],
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
    ax1.imshow(seg_crop_r_bin, cmap='gray', alpha=1, label='Original Segmentation')
    # Add ellipse fitted using skimage.regionprops
    _, _, [y0, x0] = compute_pca(seg_crop_r)
    # Center of mass in the original segmentation
    ax1.plot(x0, y0, 'ko', markersize=10, label='Original Segmentation Center of Mass')
    _add_ellipse(ax1, x0, y0)
    # Draw AP and RL axes through the center of mass of the original segmentation
    ax1.axhline(y=y0, color='k', linestyle='--', alpha=1, linewidth=1, label='AP axis (original segmentation)')
    ax1.axvline(x=x0, color='k', linestyle='--', alpha=1, linewidth=1, label='RL axis (original segmentation)')
    # Add AP and RL diameters from the original segmentation obtained using skimage.regionprops
    radius_ap = (properties['diameter_AP'] / dim[0]) * 0.5 * upscale
    radius_rl = (properties['diameter_RL'] / dim[1]) * 0.5 * upscale
    dx_ap = radius_ap * np.cos(properties['orientation'])
    dy_ap = radius_ap * np.sin(properties['orientation'])
    dx_rl = radius_rl * -np.sin(properties['orientation'])
    dy_rl = radius_rl * np.cos(properties['orientation'])
    ax1.plot([x0 - dx_ap, x0 + dx_ap], [y0 - dy_ap, y0 + dy_ap], color='blue', linestyle='--', linewidth=2,
             label=f'AP diameter (skimage.regionprops) = {properties["diameter_AP"]:.2f} mm')
    ax1.plot([x0 - dx_rl, x0 + dx_rl], [y0 - dy_rl, y0 + dy_rl], color='blue', linestyle='solid', linewidth=2,
             label=f'RL diameter (skimage.regionprops) = {properties["diameter_RL"]:.2f} mm')
    # Add A, P, R, L labels
    _add_labels(ax1)

    # 2. Rotated segmentation by angle_hog
    ax1.imshow(rotated_bin, cmap='Reds', alpha=0.4, label='Rotated Segmentation')
    # Center of mass
    ax1.plot(ap0_r, rl0_r, 'ro', markersize=10, label='Rotated Segmentation Center of Mass')
    # Draw arrow for the rotation angle
    # Flip sign to match PCA convention
    # See https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/ba30577e80a4e7387498820f0ff30b8965fbf2a4/spinalcordtoolbox/registration/algorithms.py#L834
    # TODO: figure out why the link below flip the angle sign only for src_hog but not for dest_hog
    #angle_hog = -angle_hog
    ax1.arrow(ap0_r, rl0_r, np.sin(angle_hog + (90 * math.pi / 180)) * 25,
              np.cos(angle_hog + (90 * math.pi / 180)) * 25, color='black', width=0.1,
              head_width=1, label=f'HOG angle = {angle_hog * 180 / math.pi:.1f}°')  # convert to degrees
    # Draw AP and RL axes through the center of mass of the rotated segmentation
    ax1.axhline(y=rl0_r, color='k', linestyle='dashdot', alpha=1, linewidth=1, label='AP axis (rotated segmentation)')
    ax1.axvline(x=ap0_r, color='k', linestyle='dashdot', alpha=1, linewidth=1, label='RL axis (rotated segmentation)')
    # Draw lines for the measured AP and RL diameters
    right = np.nonzero(rotated_bin[:, ap0_r])[0][0]
    left = np.nonzero(rotated_bin[:, ap0_r])[0][-1]
    anterior = np.nonzero(rotated_bin[rl0_r, :])[0][0]
    posterior = np.nonzero(rotated_bin[rl0_r, :])[0][-1]
    ax1.plot([anterior, posterior], [rl0_r, rl0_r], color='red', linestyle='--', linewidth=2,
             label=f'AP Diameter (rotated segmentation) = {ap_diameter:.2f} mm')
    ax1.plot([ap0_r, ap0_r], [right, left], color='red', linestyle='solid', linewidth=2,
             label=f'RL Diameter (rotated segmentation) = {rl_diameter:.2f} mm')

    # Plot horizontal and vertical grid lines
    ax1.grid(which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1, framealpha=1.0, fontsize=8)
    ax1.set_title(f'Slice {iz}\nOriginal segmentation and Segmentation rotated by HOG angle')

    plt.tight_layout()
    # plt.show()
    # Save the figure
    if not os.path.exists('hog_debug_figures'):
        os.makedirs('hog_debug_figures')
    fname_out = os.path.join('hog_debug_figures', f'slice_{iz}.png')
    fig.savefig(fname_out, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory
    print(f'Saved debug figure for slice {iz} with segmentation properties to {fname_out}')
