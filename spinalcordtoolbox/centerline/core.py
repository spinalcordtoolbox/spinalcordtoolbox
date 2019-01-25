#!/usr/bin/env python
# -*- coding: utf-8
# Core functions dealing with centerline extraction

import numpy as np
from spinalcordtoolbox.image import Image

def get_centerline(segmentation, algo_fitting='polyfit'):
    """
    Extract centerline from a binary or weighted segmentation by computing the center of mass slicewise.
    :param segmentation: input segmentation or series of points along the centerline. Could be an Image or a file name.
    :param algo_fitting: str:
        nurbs:
        hanning:
        polyfit: Polynomial fitting
    :return: im_centerline: Image: Centerline in discrete coordinate (int)
    :return: arr_centerline: nparray: Centerline in continuous coordinate (float) for each slice
    """
    # TODO: output continuous centerline (and add in unit test)
    # open image and change to RPI orientation
    im_seg = Image(segmentation)
    native_orientation = im_seg.orientation
    im_seg.change_orientation('RPI')
    # nx, ny, nz, nt, px, py, pz, pt = im_seg.dim

    # Extract min and max index in Z direction
    # data_seg = im_seg.data
    # X, Y, Z = (data_seg > 0).nonzero()
    # min_z_index, max_z_index = min(Z), max(Z)

    if algo_fitting == 'polyfit':
        from spinalcordtoolbox.centerline import curve_fitting
        x, y, z = np.where(im_seg.data)
        z_centerline = np.array(range(im_seg.dim[2]))
        x_centerline_fit, x_centerline_deriv = curve_fitting.polyfit_1d(z, x, z_centerline, deg=3)
        y_centerline_fit, y_centerline_deriv = curve_fitting.polyfit_1d(z, y, z_centerline, deg=3)

    elif algo_fitting == 'nurbs':
        from spinalcordtoolbox.centerline.nurbs import b_spline_nurbs
        x, y, z = np.where(im_seg.data)
        z_centerline = np.array(range(im_seg.dim[2]))
        # TODO: do something about nbControl: previous default was -1.
        x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv, \
            z_centerline_deriv, error = b_spline_nurbs(x, y, z, nbControl=None)

    # if verbose == 2:
    #     # TODO: code below does not work
    #     import matplotlib.pyplot as plt
    #
    #     # Creation of a vector x that takes into account the distance between the labels
    #     nz_nonz = len(z_centerline_voxel)
    #     x_display = [0 for i in range(x_centerline_voxel.shape[0])]
    #     y_display = [0 for i in range(y_centerline_voxel.shape[0])]
    #     for i in range(0, nz_nonz, 1):
    #         x_display[int(z_centerline_voxel[i] - z_centerline_voxel[0])] = x_centerline[i]
    #         y_display[int(z_centerline_voxel[i] - z_centerline_voxel[0])] = y_centerline[i]
    #
    #     plt.figure(1)
    #     plt.subplot(2, 1, 1)
    #     plt.plot(z_centerline_voxel, x_display, 'ro')
    #     plt.plot(z_centerline_voxel, x_centerline_voxel)
    #     plt.xlabel("Z")
    #     plt.ylabel("X")
    #     plt.title("x and x_fit coordinates")
    #
    #     plt.subplot(2, 1, 2)
    #     plt.plot(z_centerline_voxel, y_display, 'ro')
    #     plt.plot(z_centerline_voxel, y_centerline_voxel)
    #     plt.xlabel("Z")
    #     plt.ylabel("Y")
    #     plt.title("y and y_fit coordinates")
    #     plt.show()

    # Create an image with the centerline
    im_centerline = im_seg.copy()
    im_centerline.data = np.zeros(im_centerline.data.shape)
    # assign value=1 to centerline
    im_centerline.data[x_centerline_fit.round().astype(int), x_centerline_fit.round().astype(int), :] = 1
    # reorient centerline to native orientation
    im_centerline.change_orientation(native_orientation)
    # TODO: reorient output array in native orientation
    return im_centerline, np.array([x_centerline_fit, y_centerline_fit, z_centerline])

    #
    # # output csv with centerline coordinates
    # fname_centerline_csv = file_out + '.csv'
    # f_csv = open(fname_centerline_csv, 'w')
    # f_csv.write('x,y,z\n')  # csv header
    # for i in range(min_z_index, max_z_index + 1):
    #     f_csv.write("%d,%d,%d\n" % (int(i),
    #                                 x_centerline_voxel[i - min_z_index],
    #                                 y_centerline_voxel[i - min_z_index]))
    # f_csv.close()
    # TODO: display open syntax for csv

    # create a .roi file
    # fname_roi_centerline = optic.centerline2roi(fname_image=fname_centerline,
    #                                             folder_output='./',
    #                                             verbose=verbose)

    # Remove temporary files
    # if remove_temp_files:
    #     sct.printv('\nRemove temporary files...', verbose)
    #     sct.rmtree(path_tmp)