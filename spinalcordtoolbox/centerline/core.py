#!/usr/bin/env python
# -*- coding: utf-8
# Core functions dealing with centerline extraction from 3D data.

# TODO: create class ParamCenterline

import numpy as np
from spinalcordtoolbox.image import Image


class ParamCenterline:
    def __init__(self, contrast=None):
        self.contrast = contrast


def get_centerline(segmentation, algo_fitting='polyfit', param=ParamCenterline(), verbose=1):
    """
    Extract centerline from a binary or weighted segmentation by computing the center of mass slicewise.
    :param segmentation: input segmentation or series of points along the centerline. Could be an Image or a file name.
    :param algo_fitting: str:
        nurbs:
        hanning:
        polyfit: Polynomial fitting
    :param param: ParamCenterline()
    :param verbose: int: verbose level
    :return: im_centerline: Image: Centerline in discrete coordinate (int)
    :return: arr_centerline: nparray: Centerline in continuous coordinate (float) for each slice
    """
    # Open image and change to RPI orientation
    im_seg = Image(segmentation)
    native_orientation = im_seg.orientation
    im_seg.change_orientation('RPI')
    x, y, z = np.where(im_seg.data)

    # Choose method
    if algo_fitting == 'polyfit':
        from spinalcordtoolbox.centerline import curve_fitting
        z_centerline = np.array(range(im_seg.dim[2]))
        x_centerline_fit, x_centerline_deriv = curve_fitting.polyfit_1d(z, x, z_centerline, deg=3)
        y_centerline_fit, y_centerline_deriv = curve_fitting.polyfit_1d(z, y, z_centerline, deg=3)

    elif algo_fitting == 'nurbs':
        from spinalcordtoolbox.centerline.nurbs import b_spline_nurbs
        z_centerline = np.array(range(im_seg.dim[2]))
        # TODO: do something about nbControl: previous default was -1.
        x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv, \
            z_centerline_deriv, error = b_spline_nurbs(x, y, z, nbControl=None)

    elif algo_fitting == 'optic':
        from spinalcordtoolbox.centerline import optic
        img_ctl = optic.detect_centerline(im_seg, param.contrast)
        x_centerline_fit, y_centerline_fit, z_centerline = np.where(img_ctl.data)

    # Display fig of fitted curves
    if verbose == 2:
        from datetime import datetime
        import matplotlib
        matplotlib.use('Agg')  # prevent display figure
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title("Fitting algo: %s" % algo_fitting)
        plt.subplot(2, 1, 1)
        plt.plot(z, x, 'ro')
        plt.plot(z_centerline, x_centerline_fit)
        plt.xlabel("Z")
        plt.ylabel("X")
        plt.subplot(2, 1, 2)
        plt.plot(z, y, 'ro')
        plt.plot(z_centerline, y_centerline_fit)
        plt.xlabel("Z")
        plt.ylabel("Y")
        plt.savefig('fig_centerline_' + algo_fitting +'_' + datetime.now().strftime("%y%m%d%H%M%S%f") + '.png')
        plt.close()

    # Create an image with the centerline
    im_centerline = im_seg.copy()
    im_centerline.data = np.zeros(im_centerline.data.shape)
    # assign value=1 to centerline
    im_centerline.data[x_centerline_fit.round().astype(int), x_centerline_fit.round().astype(int), :] = 1
    # reorient centerline to native orientation
    im_centerline.change_orientation(native_orientation)
    # TODO: reorient output array in native orientation
    return im_centerline, np.array([x_centerline_fit, y_centerline_fit, z_centerline])
