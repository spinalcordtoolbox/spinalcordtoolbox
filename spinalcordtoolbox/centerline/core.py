#!/usr/bin/env python
# -*- coding: utf-8
# Core functions dealing with centerline extraction from 3D data.


# TODO: Add feature to born centerline at min/max z, for compatibility with flatten_sagittal

import logging
import numpy as np

from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.centerline import curve_fitting


class ParamCenterline:
    def __init__(self, contrast=None, degree=3):
        self.contrast = contrast
        self.degree = degree  # Degree of polynomial function


def find_and_sort_coord(img):
    """
    Find x,y,z coordinate of centerline and output an array which is sorted along SI direction. Removes any duplicate
    along the SI direction by averaging across the same ind_SI.
    :param img: Image(): Input image. Could be any orientation.
    :return: nx3 numpy array with X, Y, Z coordinates of center of mass
    """
    # TODO: deal with nan, etc.
    # Get indices of non-null values
    arr = np.array(np.where(img.data))
    # Sort indices according to SI axis
    dim_si = [img.orientation.find(x) for x in ['I', 'S'] if img.orientation.find(x) is not -1][0]
    # Loop across SI axis and average coordinates within duplicate SI values
    arr_sorted_avg = [np.array([])] * 3
    for i_si in sorted(set(arr[dim_si])):
        # Get indices corresponding to i_si
        ind_si_all = np.where(arr[dim_si] == i_si)
        if len(ind_si_all[0]):
            # loop across dimensions and average all existing coordinates (equivalent to center of mass)
            for i_dim in range(3):
                arr_sorted_avg[i_dim] = np.append(arr_sorted_avg[i_dim], arr[i_dim][ind_si_all].mean())
    return np.array(arr_sorted_avg)


def get_centerline(im_seg, algo_fitting='polyfit', minmax=True, param=ParamCenterline(), verbose=1):
    """
    Extract centerline from an image (using optic) or from a binary or weighted segmentation (using the center of mass).
    :param im_seg: Image(): Input segmentation or series of points along the centerline.
    :param algo_fitting: str:
        polyfit: Polynomial fitting
        nurbs:
        optic: Automatic segmentation using SVM and HOG. See [Gros et al. MIA 2018].
    :param minmax: Crop output centerline where the segmentation starts/end. If False, centerline will span all slices.
    :param param: ParamCenterline()
    :param verbose: int: verbose level
    :return: im_centerline: Image: Centerline in discrete coordinate (int)
    :return: arr_centerline: 3x1 array: Centerline in continuous coordinate (float) for each slice in RPI orientation.
    :return: arr_centerline_deriv: 3x1 array: Derivatives of x and y centerline wrt. z for each slice in RPI orient.
    """

    if not isinstance(im_seg, Image):
        raise ValueError("Expecting an image")
    # Open image and change to RPI orientation
    native_orientation = im_seg.orientation
    im_seg.change_orientation('RPI')
    px, py, pz = im_seg.dim[4:7]

    # Take the center of mass at each slice to avoid: https://stackoverflow.com/questions/2009379/interpolate-question
    x_mean, y_mean, z_mean = find_and_sort_coord(im_seg)

    # Crop output centerline to where the segmentation starts/end
    if minmax:
        z_ref = np.array(range(z_mean.min().astype(int), z_mean.max().astype(int) + 1))
    else:
        z_ref = np.array(range(im_seg.dim[2]))

    # Choose method
    if algo_fitting == 'polyfit':
        x_centerline_fit, x_centerline_deriv = curve_fitting.polyfit_1d(z_mean, x_mean, z_ref, deg=param.degree)
        y_centerline_fit, y_centerline_deriv = curve_fitting.polyfit_1d(z_mean, y_mean, z_ref, deg=param.degree)

    elif algo_fitting == 'bspline':
        x_centerline_fit, x_centerline_deriv = curve_fitting.bspline(z_mean, x_mean, z_ref, deg=param.degree)
        y_centerline_fit, y_centerline_deriv = curve_fitting.bspline(z_mean, y_mean, z_ref, deg=param.degree)

    elif algo_fitting == 'linear':
        # Simple linear interpolation
        x_centerline_fit = curve_fitting.linear(z_mean, x_mean, z_ref)
        y_centerline_fit = curve_fitting.linear(z_mean, y_mean, z_ref)
        # Compute derivatives using polynomial fit due to undefined derivatives using linear interpolation
        _, x_centerline_deriv = curve_fitting.polyfit_1d(z_mean, x_mean, z_ref, deg=5)
        _, y_centerline_deriv = curve_fitting.polyfit_1d(z_mean, y_mean, z_ref, deg=5)

    elif algo_fitting == 'nurbs':
        from spinalcordtoolbox.centerline.nurbs import b_spline_nurbs
        # Interpolate such that the output centerline has the same length as z_ref
        x_mean_interp = curve_fitting.linear(z_mean, x_mean, z_ref)
        y_mean_interp = curve_fitting.linear(z_mean, y_mean, z_ref)
        x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv, \
            z_centerline_deriv, error = b_spline_nurbs(x_mean_interp, y_mean_interp, z_ref, nbControl=None, point_number=3000,
                                                       all_slices=True)

    elif algo_fitting == 'optic':
        # This method is particular compared to the previous ones, as here we estimate the centerline based on the
        # image itself (not the segmentation). Hence, we can bypass the fitting procedure and centerline creation
        # and directly output results.
        from spinalcordtoolbox.centerline import optic
        im_centerline = optic.detect_centerline(im_seg, param.contrast)
        x_centerline_fit, y_centerline_fit, z_centerline = find_and_sort_coord(im_centerline)
        # Compute derivatives using polynomial fit
        # TODO: Fix below with reorientation of axes
        _, x_centerline_deriv = curve_fitting.polyfit_1d(z_centerline, x_centerline_fit, z_centerline, deg=5)
        _, y_centerline_deriv = curve_fitting.polyfit_1d(z_centerline, y_centerline_fit, z_centerline, deg=5)
        return im_centerline.change_orientation(native_orientation), \
               np.array([x_centerline_fit, y_centerline_fit, z_centerline]), \
               np.array([x_centerline_deriv, y_centerline_deriv, np.ones_like(z_centerline)]),

    else:
        logging.error('algo_fitting "' + algo_fitting + '" does not exist.')
        raise ValueError

    # Display fig of fitted curves
    if verbose == 2:
        from datetime import datetime
        import matplotlib
        matplotlib.use('Agg')  # prevent display figure
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Algo=%s, Deg=%s" % (algo_fitting, param.degree))
        plt.plot(z_ref * pz, x_centerline_fit * px)
        plt.plot(z_ref * pz, x_centerline_fit * px, 'b.')
        plt.plot(z_mean * pz, x_mean * px, 'ro')
        plt.ylabel("X [mm]")
        plt.subplot(2, 1, 2)
        plt.plot(z_ref, y_centerline_fit)
        plt.plot(z_ref, y_centerline_fit, 'b.')
        plt.plot(z_mean, y_mean, 'ro')
        plt.xlabel("Z [mm]")
        plt.ylabel("Y [mm]")
        plt.savefig('fig_centerline_' + datetime.now().strftime("%y%m%d%H%M%S%f") + '_' + algo_fitting + '.png')
        plt.close()

    # Create an image with the centerline
    im_centerline = im_seg.copy()
    im_centerline.data = np.zeros(im_centerline.data.shape)
    # Assign value=1 to centerline. Make sure to clip to avoid array overflow.
    # TODO: check this round and clip-- suspicious
    im_centerline.data[round_and_clip(x_centerline_fit, clip=[0, im_centerline.data.shape[0]]),
                       round_and_clip(y_centerline_fit, clip=[0, im_centerline.data.shape[1]]),
                       z_ref] = 1
    # reorient centerline to native orientation
    im_centerline.change_orientation(native_orientation)
    # TODO: Reorient centerline in native orientation. For now, we output the array in RPI. Note that it is tricky to
    #   reorient in native orientation, because the voxel center is not in the middle, but in the top corner, so this
    #   needs to be taken into accound during reorientation. The code below does not work properly.
    # # Get a permutation and inversion based on native orientation
    # perm, inversion = _get_permutations(im_seg.orientation, native_orientation)
    # # axes inversion (flip)
    # # ctl = np.array([x_centerline_fit[::inversion[0]], y_centerline_fit[::inversion[1]], z_ref[::inversion[2]]])
    # ctl = np.array([x_centerline_fit, y_centerline_fit, z_ref])
    # ctl_deriv = np.array([x_centerline_deriv[::inversion[0]], y_centerline_deriv[::inversion[1]], np.ones_like(z_ref)])
    # return im_centerline, \
    #        np.array([ctl[perm[0]], ctl[perm[1]], ctl[perm[2]]]), \
    #        np.array([ctl_deriv[perm[0]], ctl_deriv[perm[1]], ctl_deriv[perm[2]]])
    return im_centerline, \
           np.array([x_centerline_fit, y_centerline_fit, z_ref]), \
           np.array([x_centerline_deriv, y_centerline_deriv, np.ones_like(z_ref)])


def round_and_clip(arr, clip=None):
    """
    Round to closest int, convert to dtype=int and clip to min/max values allowed by the list length
    :param arr:
    :param clip: [min, max]: Clip values in arr to min and max
    :return:
    """
    if clip:
        return np.clip(arr.round().astype(int), clip[0], clip[1]-1)
    else:
        return arr.round().astype(int)


def _call_viewer_centerline(im_data, interslice_gap=20.0):
    # TODO: _call_viewer_centerline should not be "internal" anymore, i.e., remove the "_"
    """
    Call Qt viewer for manually selecting labels.
    :param im_data:
    :param interslice_gap:
    :return: Image() of labels.
    """
    from spinalcordtoolbox.gui.base import AnatomicalParams
    from spinalcordtoolbox.gui.centerline import launch_centerline_dialog

    if not isinstance(im_data, Image):
        raise ValueError("Expecting an image")

    # Get the number of slice along the (IS) axis
    im_tmp = im_data.copy().change_orientation('RPI')
    _, _, nz, _, _, _, pz, _ = im_tmp.dim
    del im_tmp

    params = AnatomicalParams()
    # setting maximum number of points to a reasonable value
    params.num_points = np.ceil(nz * pz / interslice_gap) + 2
    params.interval_in_mm = interslice_gap
    params.starting_slice = 'top'

    im_mask_viewer = zeros_like(im_data)
    launch_centerline_dialog(im_data, im_mask_viewer, params)

    return im_mask_viewer
