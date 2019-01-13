#!/usr/bin/env python
# -*- coding: utf-8
# Functions processing segmentation data

from __future__ import absolute_import

import os, math

import numpy as np

import sct_utils as sct
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.aggregate_slicewise import Metric
# TODO don't import SCT stuff outside of spinalcordtoolbox/
from sct_straighten_spinalcord import smooth_centerline
import msct_shape
from msct_types import Centerline
from .centerline import optic

# TODO: only use logging, don't use printing, pass images, not filenames, do imports at beginning of file, no chdir()

# on v3.2.2 and earlier, the following volumes were output by default, which was a waste of time (people don't use it)
OUTPUT_CSA_VOLUME = 0
OUTPUT_ANGLE_VOLUME = 0


def compute_csa(segmentation, algo_fitting='hanning', type_window='hanning', window_length=80, angle_correction=True,
                use_phys_coord=True, remove_temp_files=1, verbose=1):
    """
    Compute CSA.
    Note: segmentation can be binary or weighted for partial volume effect.
    :param segmentation: input segmentation. Could be either an Image or a file name.
    :param algo_fitting:
    :param type_window:
    :param window_length:
    :param angle_correction:
    :param use_phys_coord:
    :return metrics: Dict of class process_seg.Metric()
    """
    # create temporary folder
    path_tmp = sct.tmp_create()
    # open image and save in temp folder
    im_seg = msct_image.Image(segmentation).save(path_tmp, )

    # change orientation to RPI
    im_seg.change_orientation('RPI')
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    fname_seg = os.path.join(path_tmp, 'segmentation_RPI.nii.gz')
    im_seg.save(fname_seg)

    # Extract min and max index in Z direction
    data_seg = im_seg.data
    X, Y, Z = (data_seg > 0).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # if angle correction is required, get segmentation centerline
    # Note: even if angle_correction=0, we should run the code below so that z_centerline_voxel is defined (later used
    # with option -vert). See #1791
    if use_phys_coord:
        # fit centerline, smooth it and return the first derivative (in physical space)
        x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = \
            smooth_centerline(fname_seg, algo_fitting=algo_fitting, type_window=type_window,
                              window_length=window_length, nurbs_pts_number=3000, phys_coordinates=True,
                              verbose=verbose, all_slices=False)
        centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv,
                                y_centerline_deriv, z_centerline_deriv)

        # average centerline coordinates over slices of the image
        x_centerline_fit_rescorr, y_centerline_fit_rescorr, z_centerline_rescorr, x_centerline_deriv_rescorr, \
        y_centerline_deriv_rescorr, z_centerline_deriv_rescorr = centerline.average_coordinates_over_slices(im_seg)

        # compute Z axis of the image, in physical coordinate
        axis_X, axis_Y, axis_Z = im_seg.get_directions()

    else:
        # fit centerline, smooth it and return the first derivative (in voxel space but FITTED coordinates)
        x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = \
            smooth_centerline(fname_seg, algo_fitting=algo_fitting, type_window=type_window,
                              window_length=window_length,
                              nurbs_pts_number=3000, phys_coordinates=False, verbose=verbose, all_slices=True)

        # correct centerline fitted coordinates according to the data resolution
        x_centerline_fit_rescorr, y_centerline_fit_rescorr, z_centerline_rescorr, \
        x_centerline_deriv_rescorr, y_centerline_deriv_rescorr, z_centerline_deriv_rescorr = \
            x_centerline_fit * px, y_centerline_fit * py, z_centerline * pz, \
            x_centerline_deriv * px, y_centerline_deriv * py, z_centerline_deriv * pz

        axis_Z = [0.0, 0.0, 1.0]

    # Compute CSA
    sct.printv('\nCompute CSA...', verbose)

    # Initialize 1d array with nan. Each element corresponds to a slice.
    csa = np.full_like(np.empty(nz), np.nan, dtype=np.double)
    angles = np.full_like(np.empty(nz), np.nan, dtype=np.double)

    for iz in range(min_z_index, max_z_index + 1):
        if angle_correction:
            # in the case of problematic segmentation (e.g., non continuous segmentation often at the extremities),
            # display a warning but do not crash
            try:
                # normalize the tangent vector to the centerline (i.e. its derivative)
                tangent_vect = normalize(np.array(
                    [x_centerline_deriv_rescorr[iz - min_z_index], y_centerline_deriv_rescorr[iz - min_z_index],
                     z_centerline_deriv_rescorr[iz - min_z_index]]))

            except IndexError:
                sct.printv(
                    'WARNING: Your segmentation does not seem continuous, which could cause wrong estimations at the '
                    'problematic slices. Please check it, especially at the extremities.',
                    type='warning')

            # compute the angle between the normal vector of the plane and the vector z
            angle = np.arccos(np.vdot(tangent_vect, axis_Z))
        else:
            angle = 0.0

        # compute the number of voxels, assuming the segmentation is coded for partial volume effect between 0 and 1.
        number_voxels = np.sum(data_seg[:, :, iz])

        # compute CSA, by scaling with voxel size (in mm) and adjusting for oblique plane
        csa[iz] = number_voxels * px * py * np.cos(angle)
        angles[iz] = math.degrees(angle)

    # Remove temporary files
    if remove_temp_files:
        sct.printv('\nRemove temporary files...')
        sct.rmtree(path_tmp)

    # prepare output
    metrics = {'csa': Metric(data=csa, label='CSA [mm^2]'),
               'angle': Metric(data=angles, label='Angle between cord axis and z [deg]')}
    return metrics


def compute_shape(segmentation, algo_fitting='hanning', window_length=50, remove_temp_files=1, verbose=1):
    """
    This function characterizes the shape of the spinal cord, based on the segmentation
    Shape properties are computed along the spinal cord and averaged per z-slices.
    Option is to provide intervertebral disks to average shape properties over vertebral levels (fname_discs).
    WARNING: the segmentation needs to be binary.
    :param segmentation: input segmentation. Could be either an Image or a file name.
    :param algo_fitting:
    :param window_length:
    :param remove_temp_files:
    :param verbose:
    :return metrics: Dict of class Metric()
    """
    im_seg = msct_image.Image(segmentation).change_orientation('RPI')
    # Extract min and max index in Z direction
    data_seg = im_seg.data
    X, Y, Z = (data_seg > 0).nonzero()
    # min_z_index, max_z_index = min(Z), max(Z)

    shape_properties = msct_shape.compute_properties_along_centerline(im_seg=im_seg,
                                                                      smooth_factor=0.0,
                                                                      interpolation_mode=0,
                                                                      algo_fitting=algo_fitting,
                                                                      window_length=window_length,
                                                                      remove_temp_files=remove_temp_files,
                                                                      verbose=verbose)
    metrics = {}
    for key, value in shape_properties.items():
        # Making sure all entries added to metrics have results
        if not value == []:
            metrics[key] = Metric(value=np.array(value), label=key)

    return metrics


def extract_centerline(segmentation, verbose=0, algo_fitting='hanning', type_window='hanning',
                       window_length=5, use_phys_coord=True, file_out='centerline'):
    """
    Extract centerline from a binary or weighted segmentation by computing the center of mass slicewise.
    :param segmentation: input segmentation. Could be either an Image or a file name.
    :param verbose:
    :param algo_fitting:
    :param type_window:
    :param window_length:
    :param use_phys_coord: TODO: Explain the pros/cons of use_phys_coord.
    :param file_out:
    :return: None
    """
    # TODO: output continuous centerline (and add in unit test)
    # TODO: centerline coordinate should have the same orientation as the input image
    # TODO: no need for unecessary i/o. Everything could be done in RAM

    # Create temp folder
    path_tmp = sct.tmp_create()
    # Open segmentation volume
    im_seg = msct_image.Image(segmentation)
    # im_seg.change_orientation('RPI', generate_path=True)
    native_orientation = im_seg.orientation
    im_seg.change_orientation("RPI", generate_path=True).save(path_tmp, mutable=True)
    fname_tmp_seg = im_seg.absolutepath

    # extract centerline and smooth it
    if use_phys_coord:
        # fit centerline, smooth it and return the first derivative (in physical space)
        x_centerline_fit, y_centerline_fit, z_centerline, \
        x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
            fname_tmp_seg, algo_fitting=algo_fitting, type_window=type_window, window_length=window_length,
            nurbs_pts_number=3000, phys_coordinates=True, verbose=verbose, all_slices=False)
        centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv,
                                y_centerline_deriv, z_centerline_deriv)

        # average centerline coordinates over slices of the image (floating point)
        x_centerline_fit_rescorr, y_centerline_fit_rescorr, z_centerline_rescorr, \
        x_centerline_deriv_rescorr, y_centerline_deriv_rescorr, z_centerline_deriv_rescorr = \
            centerline.average_coordinates_over_slices(im_seg)

        # compute z_centerline in image coordinates (discrete)
        voxel_coordinates = im_seg.transfo_phys2pix(
            [[x_centerline_fit_rescorr[i], y_centerline_fit_rescorr[i], z_centerline_rescorr[i]] for i in
             range(len(z_centerline_rescorr))])
        x_centerline_voxel = [coord[0] for coord in voxel_coordinates]
        y_centerline_voxel = [coord[1] for coord in voxel_coordinates]
        z_centerline_voxel = [coord[2] for coord in voxel_coordinates]

    else:
        # fit centerline, smooth it and return the first derivative (in voxel space but FITTED coordinates)
        x_centerline_voxel, y_centerline_voxel, z_centerline_voxel, \
        x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
            'segmentation_RPI.nii.gz', algo_fitting=algo_fitting, type_window=type_window, window_length=window_length,
            nurbs_pts_number=3000, phys_coordinates=False, verbose=verbose, all_slices=True)

    if verbose == 2:
        # TODO: code below does not work
        import matplotlib.pyplot as plt

        # Creation of a vector x that takes into account the distance between the labels
        nz_nonz = len(z_centerline_voxel)
        x_display = [0 for i in range(x_centerline_voxel.shape[0])]
        y_display = [0 for i in range(y_centerline_voxel.shape[0])]
        for i in range(0, nz_nonz, 1):
            x_display[int(z_centerline_voxel[i] - z_centerline_voxel[0])] = x_centerline[i]
            y_display[int(z_centerline_voxel[i] - z_centerline_voxel[0])] = y_centerline[i]

        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(z_centerline_voxel, x_display, 'ro')
        plt.plot(z_centerline_voxel, x_centerline_voxel)
        plt.xlabel("Z")
        plt.ylabel("X")
        plt.title("x and x_fit coordinates")

        plt.subplot(2, 1, 2)
        plt.plot(z_centerline_voxel, y_display, 'ro')
        plt.plot(z_centerline_voxel, y_centerline_voxel)
        plt.xlabel("Z")
        plt.ylabel("Y")
        plt.title("y and y_fit coordinates")
        plt.show()

    # Create an image with the centerline
    # TODO: write the center of mass, not the discrete image coordinate (issue #1938)
    im_centerline = im_seg.copy()
    data_centerline = im_centerline.data * 0
    # Find z-boundaries above which and below which there is no non-null slices
    min_z_index, max_z_index = int(round(min(z_centerline_voxel))), int(round(max(z_centerline_voxel)))
    # loop across slices and set centerline pixel to value=1
    for iz in range(min_z_index, max_z_index + 1):
        data_centerline[int(round(x_centerline_voxel[iz - min_z_index])),
                        int(round(y_centerline_voxel[iz - min_z_index])),
                        int(iz)] = 1
    # assign data to centerline image
    im_centerline.data = data_centerline
    # reorient centerline to native orientation
    im_centerline.change_orientation(native_orientation)
    # save nifti volume
    fname_centerline = file_out + '.nii.gz'
    im_centerline.save(fname_centerline, dtype='uint8')
    # display stuff
    # sct.display_viewer_syntax([fname_segmentation, fname_centerline], colormaps=['gray', 'green'])

    # output csv with centerline coordinates
    fname_centerline_csv = file_out + '.csv'
    f_csv = open(fname_centerline_csv, 'w')
    f_csv.write('x,y,z\n')  # csv header
    for i in range(min_z_index, max_z_index + 1):
        f_csv.write("%d,%d,%d\n" % (int(i),
                                    x_centerline_voxel[i - min_z_index],
                                    y_centerline_voxel[i - min_z_index]))
    f_csv.close()
    # TODO: display open syntax for csv

    # create a .roi file
    fname_roi_centerline = optic.centerline2roi(fname_image=fname_centerline,
                                                folder_output='./',
                                                verbose=verbose)

    # Remove temporary files
    # if remove_temp_files:
    #     sct.printv('\nRemove temporary files...', verbose)
    #     sct.rmtree(path_tmp)


def label_vert(fname_seg, fname_label, verbose=1):
    """
    Label segmentation using vertebral labeling information. No orientation expected.
    :param fname_seg: file name of segmentation.
    :param fname_label: file name for a labelled segmentation that will be used to label the input segmentation
    :param fname_out: file name of the output labeled segmentation. If empty, will add suffix "_labeled" to fname_seg
    :param verbose:
    :return:
    """
    # Open labels
    im_disc = msct_image.Image(fname_label).change_orientation("RPI")
    # retrieve all labels
    coord_label = im_disc.getNonZeroCoordinates()
    # compute list_disc_z and list_disc_value
    list_disc_z = []
    list_disc_value = []
    for i in range(len(coord_label)):
        list_disc_z.insert(0, coord_label[i].z)
        # '-1' to use the convention "disc labelvalue=3 ==> disc C2/C3"
        list_disc_value.insert(0, coord_label[i].value - 1)

    list_disc_value = [x for (y, x) in sorted(zip(list_disc_z, list_disc_value), reverse=True)]
    list_disc_z = [y for (y, x) in sorted(zip(list_disc_z, list_disc_value), reverse=True)]
    # label segmentation
    from sct_label_vertebrae import label_segmentation
    label_segmentation(fname_seg, list_disc_z, list_disc_value, verbose=verbose)


def normalize(vect):
    """
    Normalize vector by its L2 norm
    :param vect:
    :return:
    """
    norm = np.linalg.norm(vect)
    return vect / norm
