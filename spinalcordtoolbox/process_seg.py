"""
Functions processing segmentation data

Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import math
import platform
import numpy as np
from skimage import measure, transform
import logging

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


def compute_shape(segmentation, image, angle_correction=True, centerline_path=None, param_centerline=None,
                  verbose=1, remove_temp_files=1):
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
    :return metrics: Dict of class Metric(). If a metric cannot be calculated, its value will be nan.
    :return fit_results: class centerline.core.FitResults()
    """
    # List of properties to output (in the right order)
    property_list = ['area',
                     'angle_hog',
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
    im = Image(image).change_orientation('RPI')
    # Make sure the input image and segmentation have the same dimensions
    if im_seg.dim != im.dim:
        raise ValueError(
            f"The input segmentation image ({im_seg.path}) and the input image ({im.path}) do not have the same dimensions. "
            "Please provide images with the same dimensions."
        )

    # Getting image dimensions. x, y and z respectively correspond to RL, PA and IS.
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    pr = min([px, py])
    # Resample to isotropic resolution in the axial plane. Use the minimum pixel dimension as target dimension.
    im_segr = resample_nib(im_seg, new_size=[pr, pr, pz], new_size_type='mm', interpolation='linear')
    im_r = resample_nib(im, new_size=[pr, pr, pz], new_size_type='mm', interpolation='linear')

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
        current_patch_im = im_r.data[:, :, iz]
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
            current_patch_im = current_patch_im.astype(np.float64)
            current_patch_im_scaled = transform.warp(current_patch_im,
                                                     tform.inverse,
                                                     output_shape=current_patch_im.shape,
                                                     order=1,
                                                     )
        else:
            current_patch_scaled = current_patch
            current_patch_im_scaled = current_patch_im
            angle_AP_rad, angle_RL_rad = 0.0, 0.0
        # compute shape properties on 2D patch of the segmentation
        shape_property = _properties2d(current_patch_scaled, [px, py])
        # compute PCA and get center or mass based on segmentation
        coord_src, pca_src, centermass_src = compute_pca(current_patch_scaled)
        # Finds the angle of the image
        angle_hog, conf_src = find_angle_hog(current_patch_im_scaled, centermass_src,
                                             px, py, angle_range=40)    # 40 is taken from registration.algorithms.register2d_centermassrot
        shape_property['angle_hog'] = angle_hog * 180.0 / math.pi  # convert to degrees

        # -------------
        # Debug figure: image, mask, PCA, HOG
        import matplotlib.pyplot as plt
        from matplotlib.patches import Arrow
        import os
        img = current_patch_im_scaled
        mask = current_patch_scaled
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.fliplr(np.rot90(img, 3)), cmap='gray', alpha=1)      # to match the orientation of the center of mass and PCA
        ax.imshow(np.fliplr(np.rot90(mask, 3)), cmap='Reds', alpha=0.4)     # to match the orientation of the center of mass and PCA
        # Plot PCA axes
        center = centermass_src  # (y, x) for plotting
        scale = 10
        scale_pca = 20
        for k, color in enumerate(['lime', 'green']):    # to make PCA components in different colors
            v = pca_src.components_[k]
            ax.arrow(center[0], center[1], v[1] * scale_pca, v[0] * scale_pca, color=color, linestyle='--', width=0.5, head_width=2, label=f'PCA component {k + 1}')
            ax.arrow(center[0], center[1], -v[1] * scale_pca, -v[0] * scale_pca, color=color, linestyle='--', width=0.5, head_width=2, label='')
        # Plot HOG angle
        angle = -angle_hog  # flip sign to match PCA convention
        ax.arrow(center[0], center[1], np.sin(angle) * scale, np.cos(angle) * scale, color='blue', width=0.25,
                 head_width=2, label=f'HOG = {shape_property["angle_hog"]:.1f}°')
        ax.arrow(center[0], center[1], -np.sin(angle) * scale, -np.cos(angle) * scale, color='blue', width=0.25,
                 head_width=2)
        # Plot orientation angle
        angle_or = -shape_property['orientation'] * math.pi / 180.0  # convert to radians
        ax.arrow(center[0], center[1], np.sin(angle_or) * scale, np.cos(angle_or) * scale,
                 color='orange', width=0.5, head_width=2, label=f'Ellipse Orientation = {shape_property["orientation"]:.1f}°')
        ax.arrow(center[0], center[1], -np.sin(angle_or) * scale, -np.cos(angle_or) * scale,
                    color='orange', width=0.5, head_width=2)
        ax.set_title(f'Slice {iz}: Image, Mask, PCA, HOG')
        ax.set_aspect('equal')
        ax.legend(loc='lower right', framealpha=1.0)
        plt.tight_layout()
        #plt.show()
        fname_fig = os.path.join(f'slice_{iz}_hog.png')
        plt.savefig(fname_fig, dpi=300, bbox_inches='tight')
        print(f'Saved debug figure to {fname_fig}')
        plt.close(fig)
        # -------------

        if shape_property is not None:
            # Add custom fields
            shape_property['angle_AP'] = angle_AP_rad * 180.0 / math.pi     # convert to degrees
            shape_property['angle_RL'] = angle_RL_rad * 180.0 / math.pi     # convert to degrees
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
        value = np.array(value)
        if value.size > 0:
            metrics[key] = Metric(data=value, label=key)

    return metrics, fit_results


def _properties2d(seg, dim):
    """
    Compute shape property of the input 2D segmentation. Accounts for partial volume information.
    :param seg: 2D input segmentation in uint8 or float (weighted for partial volume) that has a single object.
    :param dim: [px, py]: Physical dimension of the segmentation (in mm). X,Y respectively correspond to AP,RL.
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
    # Find RL and AP diameter based on major/minor axes and cord orientation=
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
        'centroid': region.centroid,
        'eccentricity': region.eccentricity,
        'orientation': orientation,
        'solidity': solidity,  # convexity measure
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
