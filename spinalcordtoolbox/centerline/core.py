#!/usr/bin/env python
# -*- coding: utf-8
# Core functions dealing with centerline extraction from 3D data.


import logging
import numpy as np

from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.centerline import curve_fitting

logger = logging.getLogger(__name__)


class ParamCenterline:
    """Default parameters for centerline fitting"""
    def __init__(self, algo_fitting='bspline', degree=5, smooth=20, contrast=None, minmax=True):
        """
        :param algo_fitting:
          polyfit: Polynomial fitting
          bspline: b-spline fitting
          linear: linear interpolation followed by smoothing with Hanning window
          nurbs: Non-Uniform Rational Basis Splines. See [De Leener et al., MIA, 2016]
          optic: Automatic segmentation using SVM and HOG. See [Gros et al., MIA 2018].
        :param degree: Degree of polynomial function. Only for polyfit.
        :param smooth: Degree of smoothness. Only for bspline and linear. When using algo_fitting=linear, it corresponds
        :param contrast: Contrast type for algo_fitting=optic.
        :param minmax: Crop output centerline where the segmentation starts/end. If False, centerline will span all
          slices.
        to the size of a Hanning window (in mm).
        """
        self.algo_fitting = algo_fitting
        self.contrast = contrast
        self.degree = degree
        self.smooth = smooth
        self.minmax = minmax


class FitResults:
    """
    Collection of metrics to assess fitting performance
    """
    def __init__(self):

        class Data:
            """Raw and fitted data"""
            def __init__(self):
                self.xmean = None
                self.xfit = None
                self.ymean = None
                self.yfit = None
                self.zmean = None
                self.zref = None

        self.rmse = None  # RMSE
        self.laplacian_max = None  # Maximum of 2nd derivatives
        self.data = Data()  # Raw and fitted data (for plotting in QC report)
        self.param = None  # ParamCenterline()


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


def get_centerline(im_seg, param=ParamCenterline(), verbose=1):
    """
    Extract centerline from an image (using optic) or from a binary or weighted segmentation (using the center of mass).
    :param im_seg: Image(): Input segmentation or series of points along the centerline.
    :param param: ParamCenterline() class:
    :param verbose: int: verbose level
    :return: im_centerline: Image: Centerline in discrete coordinate (int)
    :return: arr_centerline: 3x1 array: Centerline in continuous coordinate (float) for each slice in RPI orientation.
    :return: arr_centerline_deriv: 3x1 array: Derivatives of x and y centerline wrt. z for each slice in RPI orient.
    :return: fit_results: FitResults class
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
    if param.minmax:
        z_ref = np.array(range(z_mean.min().astype(int), z_mean.max().astype(int) + 1))
    else:
        z_ref = np.array(range(im_seg.dim[2]))
    index_mean = np.array([list(z_ref).index(i) for i in z_mean])

    # Choose method
    if param.algo_fitting == 'polyfit':
        x_centerline_fit, x_centerline_deriv = curve_fitting.polyfit_1d(z_mean, x_mean, z_ref, deg=param.degree)
        y_centerline_fit, y_centerline_deriv = curve_fitting.polyfit_1d(z_mean, y_mean, z_ref, deg=param.degree)
        fig_title = 'Algo={}, Deg={}'.format(param.algo_fitting, param.degree)

    elif param.algo_fitting == 'bspline':
        x_centerline_fit, x_centerline_deriv = curve_fitting.bspline(z_mean, x_mean, z_ref, param.smooth, pz=pz)
        y_centerline_fit, y_centerline_deriv = curve_fitting.bspline(z_mean, y_mean, z_ref, param.smooth, pz=pz)
        fig_title = 'Algo={}, Smooth={}'.format(param.algo_fitting, param.smooth)

    elif param.algo_fitting == 'linear':
        # Simple linear interpolation
        x_centerline_fit, x_centerline_deriv = curve_fitting.linear(z_mean, x_mean, z_ref, param.smooth, pz=pz)
        y_centerline_fit, y_centerline_deriv = curve_fitting.linear(z_mean, y_mean, z_ref, param.smooth, pz=pz)
        fig_title = 'Algo={}, Smooth={}'.format(param.algo_fitting, param.smooth)

    elif param.algo_fitting == 'nurbs':
        from spinalcordtoolbox.centerline.nurbs import b_spline_nurbs
        point_number = 3000
        # Interpolate such that the output centerline has the same length as z_ref
        x_mean_interp, _ = curve_fitting.linear(z_mean, x_mean, z_ref, 0)
        y_mean_interp, _ = curve_fitting.linear(z_mean, y_mean, z_ref, 0)
        x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv, \
            z_centerline_deriv, error = b_spline_nurbs(x_mean_interp, y_mean_interp, z_ref, nbControl=None,
                                                       point_number=point_number, all_slices=True)
        # Normalize derivatives to z_deriv
        x_centerline_deriv = x_centerline_deriv / z_centerline_deriv
        y_centerline_deriv = y_centerline_deriv / z_centerline_deriv
        fig_title = 'Algo={}, NumberPoints={}'.format(param.algo_fitting, point_number)

    elif param.algo_fitting == 'optic':
        # This method is particular compared to the previous ones, as here we estimate the centerline based on the
        # image itself (not the segmentation). Hence, we can bypass the fitting procedure and centerline creation
        # and directly output results.
        from spinalcordtoolbox.centerline import optic
        assert param.contrast is not None
        im_centerline = optic.detect_centerline(im_seg, param.contrast, verbose)
        x_centerline_fit, y_centerline_fit, z_centerline = find_and_sort_coord(im_centerline)
        # Compute derivatives using polynomial fit
        # TODO: Fix below with reorientation of axes
        _, x_centerline_deriv = curve_fitting.polyfit_1d(z_centerline, x_centerline_fit, z_centerline, deg=param.degree)
        _, y_centerline_deriv = curve_fitting.polyfit_1d(z_centerline, y_centerline_fit, z_centerline, deg=param.degree)
        return \
            im_centerline.change_orientation(native_orientation), \
            np.array([x_centerline_fit, y_centerline_fit, z_centerline]), \
            np.array([x_centerline_deriv, y_centerline_deriv, np.ones_like(z_centerline)]), \
            None
    else:
        logger.error('algo_fitting "' + param.algo_fitting + '" does not exist.')
        raise ValueError

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
    im_seg.change_orientation(native_orientation)
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

    # Compute fitting metrics
    fit_results = FitResults()
    fit_results.rmse = np.sqrt(np.mean((x_mean - x_centerline_fit[index_mean]) ** 2) * px +
                               np.mean((y_mean - y_centerline_fit[index_mean]) ** 2) * py)
    fit_results.laplacian_max = np.max([
        np.absolute(np.gradient(np.array(x_centerline_deriv * px))).max(),
        np.absolute(np.gradient(np.array(y_centerline_deriv * py))).max()])
    fit_results.data.zmean = z_mean
    fit_results.data.zref = z_ref
    fit_results.data.xmean = x_mean
    fit_results.data.xfit = x_centerline_fit
    fit_results.data.ymean = y_mean
    fit_results.data.yfit = y_centerline_fit
    fit_results.param = param

    # Display fig of fitted curves
    if verbose == 2:
        from datetime import datetime
        import matplotlib
        matplotlib.use('Agg')  # prevent display figure
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 10))
        plt.subplot(3, 1, 1)
        plt.title(fig_title + '\nRMSE[mm]={:0.2f}, LaplacianMax={:0.2f}'.format(fit_results.rmse, fit_results.laplacian_max))
        plt.plot(z_mean * pz, x_mean * px, 'ro')
        plt.plot(z_ref * pz, x_centerline_fit * px, 'k')
        plt.plot(z_ref * pz, x_centerline_fit * px, 'k.')
        plt.ylabel("X [mm]")
        plt.legend(['Reference', 'Fitting', 'Fitting points'])

        plt.subplot(3, 1, 2)
        plt.plot(z_mean * pz, y_mean * py, 'ro')
        plt.plot(z_ref * pz, y_centerline_fit * py, 'b')
        plt.plot(z_ref * pz, y_centerline_fit * py, 'b.')
        plt.xlabel("Z [mm]")
        plt.ylabel("Y [mm]")
        plt.legend(['Reference', 'Fitting', 'Fitting points'])

        plt.subplot(3, 1, 3)
        plt.plot(z_ref * pz, x_centerline_deriv * px, 'k.')
        plt.plot(z_ref * pz, y_centerline_deriv * py, 'b.')
        plt.grid(axis='y', color='grey', linestyle=':', linewidth=1)
        plt.axhline(color='grey', linestyle='-', linewidth=1)
        # plt.plot(z_ref * pz, z_centerline_deriv * pz, 'r.')
        plt.ylabel("dX/dZ, dY/dZ")
        plt.xlabel("Z [mm]")
        plt.legend(['X-deriv', 'Y-deriv'])

        plt.savefig('fig_centerline_' + datetime.now().strftime("%y%m%d-%H%M%S%f") + '_' + param.algo_fitting + '.png')
        plt.close()

    return im_centerline, \
           np.array([x_centerline_fit, y_centerline_fit, z_ref]), \
           np.array([x_centerline_deriv, y_centerline_deriv, np.ones_like(z_ref)]), \
           fit_results


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
