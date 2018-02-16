#!/usr/bin/env python
#
# This program takes as input an anatomic image and the centerline or segmentation of its spinal cord (that you can get
# using sct_get_centerline.py or sct_segmentation_propagation) and returns the anatomic image where the spinal
# cord was straightened.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Geoffrey Leveque, Julien Touati
# Modified: 2014-09-01
#
# License: see the LICENSE.TXT
# ======================================================================================================================
# check if needed Python libraries are already installed or not
import sys, io, os, shutil, time
from math import sqrt

import numpy as np
from nibabel import Nifti1Image, save
from scipy import ndimage

from msct_parser import Parser
from sct_apply_transfo import Transform
import sct_utils as sct
from msct_smooth import smoothing_window, evaluate_derivative_3D


def smooth_centerline(fname_centerline, algo_fitting='hanning', type_window='hanning', window_length=80, verbose=0, nurbs_pts_number=1000, all_slices=True, phys_coordinates=False, remove_outliers=False):
    """
    :param fname_centerline: centerline in RPI orientation, or an Image
    :return: x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv
    """
    # window_length = param.window_length
    # type_window = param.type_window
    # algo_fitting = param.algo_fitting
    remove_edge_points = 2  # remove points at the edge (issue #513)

    sct.printv('\nSmooth centerline/segmentation...', verbose)

    # get dimensions (again!)
    from msct_image import Image
    file_image = None
    if isinstance(fname_centerline, str):
        file_image = Image(fname_centerline)
    elif isinstance(fname_centerline, Image):
        file_image = fname_centerline
    else:
        sct.printv('ERROR: wrong input image', 1, 'error')

    nx, ny, nz, nt, px, py, pz, pt = file_image.dim

    # open centerline
    data = file_image.data

    # loop across z and associate x,y coordinate with the point having maximum intensity
    # N.B. len(z_centerline) = nz_nonz can be smaller than nz in case the centerline is smaller than the input volume
    z_centerline = [iz for iz in range(0, nz, 1) if data[:, :, iz].any()]
    nz_nonz = len(z_centerline)
    x_centerline = [0 for _ in range(0, nz_nonz, 1)]
    y_centerline = [0 for _ in range(0, nz_nonz, 1)]
    x_centerline_fit = [0 for _ in range(0, nz_nonz, 1)]
    y_centerline_fit = [0 for _ in range(0, nz_nonz, 1)]
    z_centerline_fit = [0 for _ in range(0, nz_nonz, 1)]
    x_centerline_deriv = [0 for _ in range(0, nz_nonz, 1)]
    y_centerline_deriv = [0 for _ in range(0, nz_nonz, 1)]
    z_centerline_deriv = [0 for _ in range(0, nz_nonz, 1)]
    num_features = [0 for _ in range(0, nz_nonz, 1)]
    distances = []

    if nz_nonz <= 5 and algo_fitting == 'nurbs':
        sct.printv('WARNING: switching to hanning smoothing due to low number of slices.', verbose=verbose, type='warning')
        algo_fitting = 'hanning'

    # get center of mass of the centerline/segmentation and remove outliers
    sct.printv('.. Get center of mass of the centerline/segmentation...', verbose)
    for iz in range(0, nz_nonz, 1):
        slice = np.array(data[:, :, z_centerline[iz]])
        labeled_array, num_f = ndimage.measurements.label(slice)
        num_features[iz] = num_f
        x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(slice)
        if iz != 0:
            distances.append(sqrt((x_centerline[iz] - x_centerline[iz - 1]) ** 2 + (y_centerline[iz] - y_centerline[iz - 1]) ** 2))

    if remove_outliers:
        mean_distances = np.mean(distances)
        std_distances = np.std(distances)
        indices_to_remove = []

        # ascending verification
        for iz in range(0, nz_nonz // 2, 1):
            distance = sqrt((x_centerline[iz] - x_centerline[iz + 1]) ** 2 + (y_centerline[iz] - y_centerline[iz + 1]) ** 2)
            if num_features[iz] > 1 or abs(distance - mean_distances) > 3 * std_distances:
                indices_to_remove.append(iz)

        # descending verification
        for iz in range(nz_nonz - 1, nz_nonz // 2, -1):
            distance = sqrt((x_centerline[iz] - x_centerline[iz - 1]) ** 2 + (y_centerline[iz] - y_centerline[iz - 1]) ** 2)
            if num_features[iz] > 1 or abs(distance - mean_distances) > 3 * std_distances:
                indices_to_remove.append(iz)

        x_centerline = np.delete(x_centerline, indices_to_remove)
        y_centerline = np.delete(y_centerline, indices_to_remove)
        z_centerline = np.delete(z_centerline, indices_to_remove)

    if phys_coordinates:
        sct.printv('.. Computing physical coordinates of centerline/segmentation...', verbose)
        coord_centerline = np.array(list(zip(x_centerline, y_centerline, z_centerline)))
        phys_coord_centerline = np.asarray(file_image.transfo_pix2phys(coord_centerline))
        x_centerline = phys_coord_centerline[:, 0]
        y_centerline = phys_coord_centerline[:, 1]
        z_centerline = phys_coord_centerline[:, 2]

    sct.printv('.. Smoothing algo = ' + algo_fitting, verbose)
    if algo_fitting == 'hanning':
        # 2D smoothing
        sct.printv('.. Windows length = ' + str(window_length), verbose)

        # change to array
        x_centerline = np.asarray(x_centerline)
        y_centerline = np.asarray(y_centerline)

        # Smooth the curve
        x_centerline_smooth = smoothing_window(x_centerline, window_len=window_length / pz, window=type_window,
                                               verbose=verbose, robust=0, remove_edge_points=remove_edge_points)
        y_centerline_smooth = smoothing_window(y_centerline, window_len=window_length / pz, window=type_window,
                                               verbose=verbose, robust=0, remove_edge_points=remove_edge_points)

        # convert to list final result
        x_centerline_smooth = x_centerline_smooth.tolist()
        y_centerline_smooth = y_centerline_smooth.tolist()

        # clear variable
        del data

        x_centerline_fit = x_centerline_smooth
        y_centerline_fit = y_centerline_smooth
        z_centerline_fit = z_centerline

        # get derivative
        x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = evaluate_derivative_3D(x_centerline_fit,
                                                                                            y_centerline_fit,
                                                                                            z_centerline, px, py, pz)

        x_centerline_fit = np.asarray(x_centerline_fit)
        y_centerline_fit = np.asarray(y_centerline_fit)
        z_centerline_fit = np.asarray(z_centerline_fit)

    elif algo_fitting == "nurbs":
        from msct_smooth import b_spline_nurbs

        # TODO: remove outliers that are at the edges of the spinal cord
        # simple way to do it: go from one end and remove point if the distance from mean is higher than 2 * std

        curdir = os.getcwd()

        x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv,\
            z_centerline_deriv, mse = b_spline_nurbs(x_centerline, y_centerline, z_centerline, nbControl=None, path_qc=curdir, point_number=nurbs_pts_number, verbose=verbose, all_slices=all_slices)

        # Checking accuracy of fitting. If NURBS fitting is not accurate enough, do not smooth segmentation
        if mse >= 2.0:
            x_centerline_fit = np.asarray(x_centerline)
            y_centerline_fit = np.asarray(y_centerline)
            z_centerline_fit = np.asarray(z_centerline)
            # get derivative
            x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = evaluate_derivative_3D(x_centerline_fit,
                                                                                                y_centerline_fit,
                                                                                                z_centerline_fit,
                                                                                                px, py, pz)

    else:
        sct.printv("ERROR: wrong algorithm for fitting", 1, "error")

    return x_centerline_fit, y_centerline_fit, z_centerline_fit, \
            x_centerline_deriv, y_centerline_deriv, z_centerline_deriv


class SpinalCordStraightener(object):

    def __init__(self, input_filename, centerline_filename, debug=0, deg_poly=10, gapxy=30, gapz=15,
                 leftright_width=150, interpolation_warp='spline', rm_tmp_files=1, verbose=1, algo_fitting='nurbs',
                 precision=2.0, threshold_distance=10, type_window='hanning', window_length=50, output_filename=''):
        self.input_filename = input_filename
        self.centerline_filename = centerline_filename
        self.output_filename = output_filename
        self.debug = debug
        self.deg_poly = deg_poly  # maximum degree of polynomial function for fitting centerline.
        self.gapxy = gapxy  # size of cross in x and y direction for the landmarks
        self.gapz = gapz  # gap between landmarks along z voxels
        # the FOV due to the curvature of the spinal cord
        self.leftright_width = leftright_width
        self.interpolation_warp = interpolation_warp
        self.remove_temp_files = rm_tmp_files  # remove temporary files
        self.verbose = verbose
        self.algo_fitting = algo_fitting  # 'hanning' or 'nurbs'
        self.precision = precision
        self.threshold_distance = threshold_distance
        self.type_window = type_window  # !! for more choices, edit msct_smooth. Possibilities: 'flat', 'hanning',
        # 'hamming', 'bartlett', 'blackman'
        self.window_length = window_length
        self.path_output = ""
        self.use_straight_reference = False
        self.centerline_reference_filename = ""
        self.disks_input_filename = ""
        self.disks_ref_filename = ""

        self.mse_straightening = 0.0
        self.max_distance_straightening = 0.0

        self.curved2straight = True
        self.straight2curved = True

        self.resample_factor = 0.0
        self.accuracy_results = 0

        self.elapsed_time = 0.0
        self.elapsed_time_accuracy = 0.0

        self.template_orientation = 0

    def straighten(self):
        # Initialization
        fname_anat = self.input_filename
        fname_centerline = self.centerline_filename
        fname_output = self.output_filename
        gapxy = self.gapxy
        gapz = self.gapz
        leftright_width = self.leftright_width
        remove_temp_files = self.remove_temp_files
        verbose = self.verbose
        interpolation_warp = self.interpolation_warp
        algo_fitting = self.algo_fitting
        window_length = self.window_length
        type_window = self.type_window
        qc = self.qc

        # start timer
        start_time = time.time()

        # get path of the toolbox
        path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
        sct.printv(path_sct, verbose)

        # Display arguments
        sct.printv("\nCheck input arguments:", verbose)
        sct.printv("  Input volume ...................... " + fname_anat, verbose)
        sct.printv("  Centerline ........................ " + fname_centerline, verbose)
        sct.printv("  Final interpolation ............... " + interpolation_warp, verbose)
        sct.printv("  Verbose ........................... " + str(verbose), verbose)
        sct.printv("", verbose)

        # Extract path/file/extension
        path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)
        path_centerline, file_centerline, ext_centerline = sct.extract_fname(fname_centerline)

        path_tmp = sct.tmp_create(basename="straighten_spinalcord", verbose=verbose)

        # Copying input data to tmp folder
        sct.printv('\nCopy files to tmp folder...', verbose)
        sct.run('sct_convert -i ' + fname_anat + ' -o ' + os.path.join(path_tmp, "data.nii"))
        sct.run('sct_convert -i ' + fname_centerline + ' -o ' + os.path.join(path_tmp, "centerline.nii.gz"))

        if self.use_straight_reference:
            sct.run('sct_convert -i ' + self.centerline_reference_filename + ' -o ' + os.path.join(path_tmp, "centerline_ref.nii.gz"))
        if self.disks_input_filename != '':
            sct.run('sct_convert -i ' + self.disks_input_filename + ' -o ' + os.path.join(path_tmp, "labels_input.nii.gz"))
        if self.disks_ref_filename != '':
            sct.run('sct_convert -i ' + self.disks_ref_filename + ' -o ' + os.path.join(path_tmp, "labels_ref.nii.gz"))

        # go to tmp folder
        curdir = os.getcwd()
        os.chdir(path_tmp)

        try:
            # Change orientation of the input centerline into RPI
            sct.printv("\nOrient centerline to RPI orientation...", verbose)
            sct.run('sct_image -i centerline.nii.gz -setorient RPI -o centerline_rpi.nii.gz')

            # Get dimension
            sct.printv('\nGet dimensions...', verbose)
            from msct_image import Image
            image_centerline = Image('centerline_rpi.nii.gz')
            nx, ny, nz, nt, px, py, pz, pt = image_centerline.dim
            sct.printv('.. matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)
            sct.printv('.. voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm', verbose)

            if self.resample_factor != 0.0:
                os.rename('centerline_rpi.nii.gz', 'centerline_rpi_native.nii.gz')
                pz_native = pz
                sct.run('sct_resample -i centerline_rpi_native.nii.gz -mm ' + str(self.resample_factor) + 'x' + str(self.resample_factor) + 'x' + str(self.resample_factor) + ' -o centerline_rpi.nii.gz')
                image_centerline = Image('centerline_rpi.nii.gz')
                nx, ny, nz, nt, px, py, pz, pt = image_centerline.dim

            if np.min(image_centerline.data) < 0 or np.max(image_centerline.data) > 1:
                image_centerline.data[image_centerline.data < 0] = 0
                image_centerline.data[image_centerline.data > 1] = 1
                image_centerline.save()

            """
            Steps: (everything is done in physical space)
            1. open input image and centreline image
            2. extract bspline fitting of the centreline, and its derivatives
            3. compute length of centerline
            4. compute and generate straight space
            5. compute transformations
                for each voxel of one space: (done using matrices --> improves speed by a factor x300)
                    a. determine which plane of spinal cord centreline it is included
                    b. compute the position of the voxel in the plane (X and Y distance from centreline, along the plane)
                    c. find the correspondant centreline point in the other space
                    d. find the correspondance of the voxel in the corresponding plane
            6. generate warping fields for each transformations
            7. write warping fields and apply them

            step 5.b: how to find the corresponding plane?
                The centerline plane corresponding to a voxel correspond to the nearest point of the centerline.
                However, we need to compute the distance between the voxel position and the plane to be sure it is part of the plane and not too distant.
                If it is more far than a threshold, warping value should be 0.

            step 5.d: how to make the correspondance between centerline point in both images?
                Both centerline have the same lenght. Therefore, we can map centerline point via their position along the curve.
                If we use the same number of points uniformely along the spinal cord (1000 for example), the correspondance is straight-forward.
            """

            # number of points along the spinal cord
            if algo_fitting == 'hanning':
                number_of_points = nz
            else:
                number_of_points = int(self.precision * (float(nz) / pz))
                if number_of_points < 100:
                    number_of_points *= 50
                if number_of_points == 0:
                    number_of_points = 50

            # 2. extract bspline fitting of the centreline, and its derivatives
            x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline('centerline_rpi.nii.gz', algo_fitting=algo_fitting, type_window=type_window, window_length=window_length, verbose=verbose, nurbs_pts_number=number_of_points, all_slices=False, phys_coordinates=True, remove_outliers=True)
            from msct_types import Centerline
            centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

            number_of_points = centerline.number_of_points

            # ==========================================================================================
            sct.printv("\nCreate the straight space and the safe zone...", verbose)
            # 3. compute length of centerline
            # compute the length of the spinal cord based on fitted centerline and size of centerline in z direction
            from math import sqrt, atan2, sin

            # Computation of the safe zone.
            # The safe zone is defined as the length of the spinal cord for which an axial segmentation will be complete
            # The safe length (to remove) is computed using the safe radius (given as parameter) and the angle of the
            # last centerline point with the inferior-superior direction. Formula: Ls = Rs * sin(angle)
            # Calculate Ls for both edges and remove appropriate number of centerline points
            radius_safe = 0.0  # mm

            # inferior edge
            u = np.array([x_centerline_deriv[0], y_centerline_deriv[0], z_centerline_deriv[0]])
            v = np.array([0, 0, -1])
            angle_inferior = atan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
            length_safe_inferior = radius_safe * sin(angle_inferior)

            # superior edge
            u = np.array([x_centerline_deriv[-1], y_centerline_deriv[-1], z_centerline_deriv[-1]])
            v = np.array([0, 0, 1])
            angle_superior = atan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
            length_safe_superior = radius_safe * sin(angle_superior)

            # remove points
            from bisect import bisect
            inferior_bound = bisect(centerline.progressive_length, length_safe_inferior) - 1
            superior_bound = centerline.number_of_points - bisect(centerline.progressive_length_inverse, length_safe_superior)

            length_centerline = centerline.length
            size_z_centerline = z_centerline[-1] - z_centerline[0]

            # compute the size factor between initial centerline and straight bended centerline
            factor_curved_straight = length_centerline / size_z_centerline
            middle_slice = (z_centerline[0] + z_centerline[-1]) / 2.0

            bound_curved = [z_centerline[inferior_bound], z_centerline[superior_bound]]
            bound_straight = [(z_centerline[inferior_bound] - middle_slice) * factor_curved_straight + middle_slice,
                              (z_centerline[superior_bound] - middle_slice) * factor_curved_straight + middle_slice]

            if verbose == 2:
                sct.printv("Length of spinal cord = ", str(length_centerline))
                sct.printv("Size of spinal cord in z direction = ", str(size_z_centerline))
                sct.printv("Ratio length/size = ", str(factor_curved_straight))
                sct.printv("Safe zone boundaries: ")
                sct.printv("Curved space = ", bound_curved)
                sct.printv("Straight space = ", bound_straight)

            # 4. compute and generate straight space
            # points along curved centerline are already regularly spaced.
            # calculate position of points along straight centerline

            # Create straight NIFTI volumes
            # ==========================================================================================
            if self.use_straight_reference:
                image_centerline_pad = Image('centerline_rpi.nii.gz')
                nx, ny, nz, nt, px, py, pz, pt = image_centerline_pad.dim

                sct.run('sct_image -i centerline_ref.nii.gz -setorient RPI -o centerline_ref_rpi.nii.gz')
                fname_ref = 'centerline_ref_rpi.nii.gz'
                image_centerline_straight = Image('centerline_ref_rpi.nii.gz')
                nx_s, ny_s, nz_s, nt_s, px_s, py_s, pz_s, pt_s = image_centerline_straight.dim
                x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline('centerline_ref_rpi.nii.gz', algo_fitting=algo_fitting, type_window=type_window, window_length=window_length, verbose=verbose, nurbs_pts_number=number_of_points, all_slices=False, phys_coordinates=True, remove_outliers=True)
                centerline_straight = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

                hdr_warp = image_centerline_pad.hdr.copy()
                hdr_warp_s = image_centerline_straight.hdr.copy()
                hdr_warp_s.set_data_dtype('float32')

                if self.disks_input_filename != "" and self.disks_ref_filename != "":
                    disks_input_image = Image('labels_input.nii.gz')
                    coord = disks_input_image.getNonZeroCoordinates(sorting='z', reverse_coord=True)
                    coord_physical = []
                    for c in coord:
                        c_p = disks_input_image.transfo_pix2phys([[c.x, c.y, c.z]])[0]
                        c_p.append(c.value)
                        coord_physical.append(c_p)
                    centerline.compute_vertebral_distribution(coord_physical)
                    centerline.save_centerline(image=disks_input_image, fname_output='disks_input_image.nii.gz')

                    disks_ref_image = Image('labels_ref.nii.gz')
                    coord = disks_ref_image.getNonZeroCoordinates(sorting='z', reverse_coord=True)
                    coord_physical = []
                    for c in coord:
                        c_p = disks_ref_image.transfo_pix2phys([[c.x, c.y, c.z]])[0]
                        c_p.append(c.value)
                        coord_physical.append(c_p)
                    centerline_straight.compute_vertebral_distribution(coord_physical)
                    centerline_straight.save_centerline(image=disks_ref_image, fname_output='disks_ref_image.nii.gz')

            else:
                sct.printv('\nPad input volume to account for spinal cord length...', verbose)
                from numpy import ceil
                start_point = (z_centerline[0] - middle_slice) * factor_curved_straight + middle_slice
                end_point = (z_centerline[-1] - middle_slice) * factor_curved_straight + middle_slice

                xy_space = 35  # in mm
                offset_z = 0

                # if the destination image is resampled, we still create the straight reference space with the native resolution
                if self.resample_factor != 0.0:
                    padding_z = int(ceil(1.5 * ((length_centerline - size_z_centerline) / 2.0) / pz_native))
                    sct.run('sct_image -i centerline_rpi_native.nii.gz -o tmp.centerline_pad_native.nii.gz -pad 0,0,' + str(padding_z))
                    image_centerline_pad = Image('centerline_rpi_native.nii.gz')
                    nx, ny, nz, nt, px, py, pz, pt = image_centerline_pad.dim
                    start_point_coord_native = image_centerline_pad.transfo_phys2pix([[0, 0, start_point]])[0]
                    end_point_coord_native = image_centerline_pad.transfo_phys2pix([[0, 0, end_point]])[0]
                    straight_size_x = int(xy_space / px)
                    straight_size_y = int(xy_space / py)
                    warp_space_x = [int(np.round(nx / 2)) - straight_size_x, int(np.round(nx / 2)) + straight_size_x]
                    warp_space_y = [int(np.round(ny / 2)) - straight_size_y, int(np.round(ny / 2)) + straight_size_y]
                    if warp_space_x[0] < 0:
                        warp_space_x[1] += warp_space_x[0] - 2
                        warp_space_x[0] = 0
                    if warp_space_y[0] < 0:
                        warp_space_y[1] += warp_space_y[0] - 2
                        warp_space_y[0] = 0
                    if self.resample_factor != 0.0:
                        sct.run('sct_crop_image -i tmp.centerline_pad_native.nii.gz -o tmp.centerline_pad_crop_native.nii.gz -dim 0,1,2 -start ' + str(warp_space_x[0]) + ',' + str(warp_space_y[0]) + ',0 -end ' + str(warp_space_x[1]) + ',' + str(warp_space_y[1]) + ',' + str(end_point_coord_native[2] - start_point_coord_native[2]))

                    fname_ref = 'tmp.centerline_pad_crop_native.nii.gz'
                    xy_space = 40
                    offset_z = 4
                else:
                    fname_ref = 'tmp.centerline_pad_crop.nii.gz'

                nx, ny, nz, nt, px, py, pz, pt = image_centerline.dim
                padding_z = int(ceil(1.5 * ((length_centerline - size_z_centerline) / 2.0) / pz)) + offset_z
                sct.run('sct_image -i centerline_rpi.nii.gz -o tmp.centerline_pad.nii.gz -pad 0,0,' + str(padding_z))
                image_centerline_pad = Image('centerline_rpi.nii.gz')
                nx, ny, nz, nt, px, py, pz, pt = image_centerline_pad.dim
                hdr_warp = image_centerline_pad.hdr.copy()
                start_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, start_point]])[0]
                end_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, end_point]])[0]

                straight_size_x = int(xy_space / px)
                straight_size_y = int(xy_space / py)
                warp_space_x = [int(np.round(nx / 2)) - straight_size_x, int(np.round(nx / 2)) + straight_size_x]
                warp_space_y = [int(np.round(ny / 2)) - straight_size_y, int(np.round(ny / 2)) + straight_size_y]
                if warp_space_x[0] < 0:
                    warp_space_x[1] += warp_space_x[0] - 2
                    warp_space_x[0] = 0
                if warp_space_y[0] < 0:
                    warp_space_y[1] += warp_space_y[0] - 2
                    warp_space_y[0] = 0

                sct.run('sct_crop_image -i tmp.centerline_pad.nii.gz -o tmp.centerline_pad_crop.nii.gz -dim 0,1,2 -start ' + str(warp_space_x[0]) + ',' + str(warp_space_y[0]) + ',0 -end ' + str(warp_space_x[1]) + ',' + str(warp_space_y[1]) + ',' + str(end_point_coord[2] - start_point_coord[2] + offset_z))

                image_centerline_straight = Image('tmp.centerline_pad_crop.nii.gz')
                nx_s, ny_s, nz_s, nt_s, px_s, py_s, pz_s, pt_s = image_centerline_straight.dim
                hdr_warp_s = image_centerline_straight.hdr.copy()
                hdr_warp_s.set_data_dtype('float32')
                #origin = [(nx_s * px_s)/2.0, -(ny_s * py_s)/2.0, -(nz_s * pz_s)/2.0]
                #hdr_warp_s.structarr['qoffset_x'] = origin[0]
                #hdr_warp_s.structarr['qoffset_y'] = origin[1]
                #hdr_warp_s.structarr['qoffset_z'] = origin[2]
                #hdr_warp_s.structarr['srow_x'][-1] = origin[0]
                #hdr_warp_s.structarr['srow_y'][-1] = origin[1]
                #hdr_warp_s.structarr['srow_z'][-1] = origin[2]

                if self.template_orientation == 1:
                    hdr_warp_s.structarr['quatern_b'] = 0.0
                    hdr_warp_s.structarr['quatern_c'] = 1.0
                    hdr_warp_s.structarr['quatern_d'] = 0.0
                    hdr_warp_s.structarr['srow_x'][0] = -px_s
                    hdr_warp_s.structarr['srow_x'][1] = 0.0
                    hdr_warp_s.structarr['srow_x'][2] = 0.0
                    hdr_warp_s.structarr['srow_y'][0] = 0.0
                    hdr_warp_s.structarr['srow_y'][1] = py_s
                    hdr_warp_s.structarr['srow_y'][2] = 0.0
                    hdr_warp_s.structarr['srow_z'][0] = 0.0
                    hdr_warp_s.structarr['srow_z'][1] = 0.0
                    hdr_warp_s.structarr['srow_z'][2] = pz_s

                image_centerline_straight.hdr = hdr_warp_s
                image_centerline_straight.compute_transform_matrix()
                image_centerline_straight.save()

                start_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, start_point]])[0]
                end_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, end_point]])[0]

                number_of_voxel = nx * ny * nz
                sct.printv("Number of voxel = " + str(number_of_voxel))

                time_centerlines = time.time()

                from numpy import linspace
                ix_straight = [int(np.round(nx_s / 2))] * number_of_points
                iy_straight = [int(np.round(ny_s / 2))] * number_of_points
                iz_straight = linspace(0, end_point_coord[2] - start_point_coord[2], number_of_points)
                dx_straight = [0.0] * number_of_points
                dy_straight = [0.0] * number_of_points
                dz_straight = [1.0] * number_of_points
                coord_straight = np.array(list(zip(ix_straight, iy_straight, iz_straight)))
                coord_phys_straight = np.asarray(image_centerline_straight.transfo_pix2phys(coord_straight))

                centerline_straight = Centerline(coord_phys_straight[:, 0], coord_phys_straight[:, 1], coord_phys_straight[:, 2],
                                                 dx_straight, dy_straight, dz_straight)

                time_centerlines = time.time() - time_centerlines
                sct.printv('Time to generate centerline: ' + str(np.round(time_centerlines * 1000.0)) + ' ms', verbose)

            """ 
            import matplotlib.pyplot as plt
            curved_points = centerline.progressive_length
            straight_points = centerline_straight.progressive_length
            range_points = np.linspace(0, 1, number_of_points)
            dist_curved = np.zeros(number_of_points)
            dist_straight = np.zeros(number_of_points)
            for i in range(1, number_of_points):
                dist_curved[i] = dist_curved[i - 1] + curved_points[i - 1] / centerline.length
                dist_straight[i] = dist_straight[i - 1] + straight_points[i - 1] / centerline_straight.length
            plt.plot(range_points, dist_curved)
            plt.plot(range_points, dist_straight)
            plt.grid(True)
            plt.show()
            """

            #alignment_mode = 'length'
            alignment_mode = 'levels'

            lookup_curved2straight = range(centerline.number_of_points)
            if self.disks_input_filename != "":
                # create look-up table curved to straight
                for index in range(centerline.number_of_points):
                    disk_label = centerline.l_points[index]
                    if alignment_mode == 'length':
                        relative_position = centerline.dist_points[index]
                    else:
                        relative_position = centerline.dist_points_rel[index]
                    idx_closest = centerline_straight.get_closest_to_absolute_position(disk_label, relative_position, backup_index=index, backup_centerline=centerline_straight, mode=alignment_mode)
                    if idx_closest is not None:
                        lookup_curved2straight[index] = idx_closest
                    else:
                        lookup_curved2straight[index] = 0
            for p in range(0, len(lookup_curved2straight)//2):
                if lookup_curved2straight[p] == lookup_curved2straight[p + 1]:
                    lookup_curved2straight[p] = 0
                else:
                    break
            for p in range(len(lookup_curved2straight)-1, len(lookup_curved2straight)//2, -1):
                if lookup_curved2straight[p] == lookup_curved2straight[p - 1]:
                    lookup_curved2straight[p] = 0
                else:
                    break
            lookup_curved2straight = np.array(lookup_curved2straight)

            lookup_straight2curved = range(centerline_straight.number_of_points)
            if self.disks_input_filename != "":
                for index in range(centerline_straight.number_of_points):
                    disk_label = centerline_straight.l_points[index]
                    if alignment_mode == 'length':
                        relative_position = centerline_straight.dist_points[index]
                    else:
                        relative_position = centerline_straight.dist_points_rel[index]
                    idx_closest = centerline.get_closest_to_absolute_position(disk_label, relative_position, backup_index=index, backup_centerline=centerline_straight, mode=alignment_mode)
                    if idx_closest is not None:
                        lookup_straight2curved[index] = idx_closest
            for p in range(0, len(lookup_straight2curved)//2):
                if lookup_straight2curved[p] == lookup_straight2curved[p + 1]:
                    lookup_straight2curved[p] = 0
                else:
                    break
            for p in range(len(lookup_straight2curved)-1, len(lookup_straight2curved)//2, -1):
                if lookup_straight2curved[p] == lookup_straight2curved[p - 1]:
                    lookup_straight2curved[p] = 0
                else:
                    break
            lookup_straight2curved = np.array(lookup_straight2curved)

            # Create volumes containing curved and straight warping fields
            time_generation_volumes = time.time()
            data_warp_curved2straight = np.zeros((nx_s, ny_s, nz_s, 1, 3))
            data_warp_straight2curved = np.zeros((nx, ny, nz, 1, 3))

            # 5. compute transformations
            # Curved and straight images and the same dimensions, so we compute both warping fields at the same time.
            # b. determine which plane of spinal cord centreline it is included
            # sct.printv(nx * ny * nz, nx_s * ny_s * nz_s)

            if self.curved2straight:
                timer_straightening = sct.Timer(nz_s)
                timer_straightening.start()
                for u in range(nz_s):
                    timer_straightening.add_iteration()
                    x_s, y_s, z_s = np.mgrid[0:nx_s, 0:ny_s, u:u + 1]
                    indexes_straight = np.array(list(zip(x_s.ravel(), y_s.ravel(), z_s.ravel())))
                    physical_coordinates_straight = image_centerline_straight.transfo_pix2phys(indexes_straight)
                    nearest_indexes_straight = centerline_straight.find_nearest_indexes(physical_coordinates_straight)
                    distances_straight = centerline_straight.get_distances_from_planes(physical_coordinates_straight, nearest_indexes_straight)
                    lookup = lookup_straight2curved[nearest_indexes_straight]
                    indexes_out_distance_straight = np.logical_or(np.logical_or(distances_straight > self.threshold_distance, distances_straight < -self.threshold_distance), lookup == 0)
                    projected_points_straight = centerline_straight.get_projected_coordinates_on_planes(physical_coordinates_straight, nearest_indexes_straight)
                    coord_in_planes_straight = centerline_straight.get_in_plans_coordinates(projected_points_straight, nearest_indexes_straight)

                    coord_straight2curved = centerline.get_inverse_plans_coordinates(coord_in_planes_straight, lookup)
                    displacements_straight = coord_straight2curved - physical_coordinates_straight
                    # for some reason, displacement in Z is inverted. Probably due to left/right-handed definition of referential.
                    #displacements_straight[:, 0] = -displacements_straight[:, 0]
                    displacements_straight[:, 2] = -displacements_straight[:, 2]
                    displacements_straight[indexes_out_distance_straight] = [100000.0, 100000.0, 100000.0]

                    data_warp_curved2straight[indexes_straight[:, 0], indexes_straight[:, 1], indexes_straight[:, 2], 0, :] = -displacements_straight
                timer_straightening.stop()

            if self.straight2curved:
                timer_straightening = sct.Timer(nz)
                timer_straightening.start()
                for u in range(nz):
                    timer_straightening.add_iteration()
                    x, y, z = np.mgrid[0:nx, 0:ny, u:u + 1]
                    indexes = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())))
                    physical_coordinates = image_centerline_pad.transfo_pix2phys(indexes)
                    nearest_indexes_curved = centerline.find_nearest_indexes(physical_coordinates)
                    distances_curved = centerline.get_distances_from_planes(physical_coordinates, nearest_indexes_curved)
                    lookup = lookup_curved2straight[nearest_indexes_curved]
                    indexes_out_distance_curved = np.logical_or(np.logical_or(distances_curved > self.threshold_distance, distances_curved < -self.threshold_distance), lookup == 0)
                    projected_points_curved = centerline.get_projected_coordinates_on_planes(physical_coordinates, nearest_indexes_curved)
                    coord_in_planes_curved = centerline.get_in_plans_coordinates(projected_points_curved, nearest_indexes_curved)

                    coord_curved2straight = centerline_straight.points[lookup]
                    coord_curved2straight[:, 0:2] += coord_in_planes_curved[:, 0:2]
                    coord_curved2straight[:, 2] += distances_curved

                    displacements_curved = coord_curved2straight - physical_coordinates
                    # for some reason, displacement in Z is inverted. Probably due to left/right-hended definition of referential.
                    #displacements_curved[:, 0] = -displacements_curved[:, 0]
                    displacements_curved[:, 2] = -displacements_curved[:, 2]
                    displacements_curved[indexes_out_distance_curved] = [100000.0, 100000.0, 100000.0]

                    data_warp_straight2curved[indexes[:, 0], indexes[:, 1], indexes[:, 2], 0, :] = -displacements_curved
                timer_straightening.stop()

            # Creation of the safe zone based on pre-calculated safe boundaries
            coord_bound_curved_inf, coord_bound_curved_sup = image_centerline_pad.transfo_phys2pix([[0, 0, bound_curved[0]]]), image_centerline_pad.transfo_phys2pix([[0, 0, bound_curved[1]]])
            coord_bound_straight_inf, coord_bound_straight_sup = image_centerline_straight.transfo_phys2pix([[0, 0, bound_straight[0]]]), image_centerline_straight.transfo_phys2pix([[0, 0, bound_straight[1]]])

            if radius_safe > 0:
                data_warp_curved2straight[:, :, 0:coord_bound_straight_inf[0][2], 0, :] = 100000.0
                data_warp_curved2straight[:, :, coord_bound_straight_sup[0][2]:, 0, :] = 100000.0
                data_warp_straight2curved[:, :, 0:coord_bound_curved_inf[0][2], 0, :] = 100000.0
                data_warp_straight2curved[:, :, coord_bound_curved_sup[0][2]:, 0, :] = 100000.0

            # Generate warp files as a warping fields
            hdr_warp_s.set_intent('vector', (), '')
            hdr_warp_s.set_data_dtype('float32')
            hdr_warp.set_intent('vector', (), '')
            hdr_warp.set_data_dtype('float32')
            if self.curved2straight:
                img = Nifti1Image(data_warp_curved2straight, None, hdr_warp_s)
                save(img, 'tmp.curve2straight.nii.gz')
                sct.printv('\nDONE ! Warping field generated: tmp.curve2straight.nii.gz', verbose)

            if self.straight2curved:
                img = Nifti1Image(data_warp_straight2curved, None, hdr_warp)
                save(img, 'tmp.straight2curve.nii.gz')
                sct.printv('\nDONE ! Warping field generated: tmp.straight2curve.nii.gz', verbose)

            if self.curved2straight:
                # Apply transformation to input image
                sct.printv('\nApply transformation to input image...', verbose)
                sct.run('sct_apply_transfo -i data.nii -d ' + fname_ref + ' -o tmp.anat_rigid_warp.nii.gz -w tmp.curve2straight.nii.gz -x ' + interpolation_warp, verbose)

            if self.accuracy_results:
                time_accuracy_results = time.time()
                # compute the error between the straightened centerline/segmentation and the central vertical line.
                # Ideally, the error should be zero.
                # Apply deformation to input image
                sct.printv('\nApply transformation to centerline image...', verbose)
                Transform(input_filename='centerline.nii.gz', fname_dest=fname_ref,
                          output_filename="tmp.centerline_straight.nii.gz", interp="nn",
                          warp="tmp.curve2straight.nii.gz", verbose=verbose).apply()
                from msct_image import Image
                file_centerline_straight = Image('tmp.centerline_straight.nii.gz', verbose=verbose)
                coordinates_centerline = file_centerline_straight.getNonZeroCoordinates(sorting='z')
                mean_coord = []
                for z in range(coordinates_centerline[0].z, coordinates_centerline[-1].z):
                    temp_mean = [coord.value for coord in coordinates_centerline if coord.z == z]
                    if temp_mean:
                        mean_value = np.mean(temp_mean)
                        mean_coord.append(np.mean([[coord.x * coord.value / mean_value, coord.y * coord.value / mean_value]
                                                    for coord in coordinates_centerline if coord.z == z], axis=0))

                # compute error between the straightened centerline and the straight line.
                x0 = file_centerline_straight.data.shape[0] / 2.0
                y0 = file_centerline_straight.data.shape[1] / 2.0
                count_mean = 0
                if number_of_points >= 10:
                    mean_c = mean_coord[2:-2]  # we don't include the four extrema because there are usually messy.
                else:
                    mean_c = mean_coord
                for coord_z in mean_c:
                    if not np.isnan(np.sum(coord_z)):
                        dist = ((x0 - coord_z[0]) * px)**2 + ((y0 - coord_z[1]) * py)**2
                        self.mse_straightening += dist
                        dist = sqrt(dist)
                        if dist > self.max_distance_straightening:
                            self.max_distance_straightening = dist
                        count_mean += 1
                self.mse_straightening = sqrt(self.mse_straightening / float(count_mean))

                self.elapsed_time_accuracy = time.time() - time_accuracy_results

        except Exception as e:
            sct.printv('WARNING: Exception during Straightening:', 1, 'warning')
            sct.printv('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), 1, 'warning')
            sct.printv(str(e), 1, 'warning')

        os.chdir(curdir)

        # Generate output file (in current folder)
        # TODO: do not uncompress the warping field, it is too time consuming!
        sct.printv("\nGenerate output file (in current folder)...", verbose)
        if self.curved2straight:
            sct.generate_output_file(os.path.join(path_tmp, "tmp.curve2straight.nii.gz"), os.path.join(self.path_output, "warp_curve2straight.nii.gz"), verbose)
        if self.straight2curved:
            sct.generate_output_file(os.path.join(path_tmp, "tmp.straight2curve.nii.gz"), os.path.join(self.path_output, "warp_straight2curve.nii.gz"), verbose)

        # create ref_straight.nii.gz file that can be used by other SCT functions that need a straight reference space
        if self.curved2straight:
            sct.copy(os.path.join(path_tmp, "tmp.anat_rigid_warp.nii.gz"), os.path.join(self.path_output, "straight_ref.nii.gz"))
            # move straightened input file
            if fname_output == '':
                fname_straight = sct.generate_output_file(os.path.join(path_tmp, "tmp.anat_rigid_warp.nii.gz"),
                                                          os.path.join(self.path_output, file_anat + "_straight" + ext_anat), verbose)
            else:
                fname_straight = sct.generate_output_file(os.path.join(path_tmp, "tmp.anat_rigid_warp.nii.gz"),
                                                          os.path.join(self.path_output, fname_output), verbose)  # straightened anatomic

        # Remove temporary files
        if remove_temp_files:
            sct.printv("\nRemove temporary files...", verbose)
            shutil.rmtree(path_tmp)

        sct.printv('\nDone!\n', verbose)

        if self.accuracy_results:
            sct.printv("Maximum x-y error = " + str(np.round(self.max_distance_straightening, 2)) + " mm", verbose, "bold")
            sct.printv("Accuracy of straightening (MSE) = " + str(np.round(self.mse_straightening, 2)) +
                       " mm", verbose, "bold")

        # display elapsed time
        self.elapsed_time = time.time() - start_time
        sct.printv("\nFinished! Elapsed time: " + str(int(np.round(self.elapsed_time))) + " s", verbose)
        if self.accuracy_results:
            sct.printv('    including ' + str(int(np.round(self.elapsed_time_accuracy))) + ' s spent computing '
                                                                                      'accuracy results', verbose)
        if self.curved2straight:
            sct.display_viewer_syntax([fname_straight], verbose=verbose)

        # output QC image
        if qc and self.curved2straight:
            from msct_image import Image
            Image(fname_straight).save_quality_control(plane='sagittal', n_slices=1, path_output=self.path_output)


def get_parser():
    # Initialize parser
    parser = Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description("This program takes as input an anatomic image and the centerline or segmentation of "
                                 "its spinal cord (that you can get using sct_get_centerline.py or "
                                 "sct_segmentation_propagation) and returns the anatomic image where the spinal cord "
                                 "was straightened.")
    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="input image.",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name="-s",
                      type_value="image_nifti",
                      description="centerline or segmentation.",
                      mandatory=True,
                      example="centerline.nii.gz")
    parser.add_option(name="-c",
                      type_value=None,
                      description="centerline or segmentation.",
                      mandatory=False,
                      deprecated_by='-s')
    parser.add_option(name="-ref",
                      type_value="image_nifti",
                      description="reference centerline (or segmentation) on which to register the input image, using the same philosophy as straightening procedure..",
                      mandatory=False,
                      example="centerline.nii.gz")
    parser.add_option(name="-disks-input",
                      type_value="image_nifti",
                      description="",
                      mandatory=False,
                      example="disks.nii.gz")
    parser.add_option(name="-disks-ref",
                      type_value="image_nifti",
                      description="",
                      mandatory=False,
                      example="disks_ref.nii.gz")
    parser.add_option(name="-p",
                      type_value=None,
                      description="amount of padding for generating labels.",
                      mandatory=False,
                      deprecated_by='-pad')
    parser.add_option(name="-disable-straight2curved",
                      type_value=None,
                      description="Disable straight to curved transformation computation.",
                      mandatory=False)
    parser.add_option(name="-disable-curved2straight",
                      type_value=None,
                      description="Disable curved to straight transformation computation.",
                      mandatory=False)
    parser.add_option(name="-resample",
                      type_value='float',
                      description='Isotropic resolution of the straightening output, in millimeters.\n'
                                  'Resampling to lower resolution decreases computational time while decreasing straightening accuracy.\n'
                                  'To keep native resolution, set this option to 0.\n',
                      mandatory=False,
                      default_value=0)
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="straightened file",
                      mandatory=False,
                      default_value='',
                      example="data_straight.nii.gz")
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder (all outputs will go there).",
                      mandatory=False,
                      default_value='')
    parser.add_option(name="-x",
                      type_value="multiple_choice",
                      description="Final interpolation.",
                      mandatory=False,
                      example=["nn", "linear", "spline"],
                      default_value="spline")
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="remove temporary files.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="Verbose. 0: nothing, 1: basic, 2: extended.",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')

    parser.add_option(name="-param",
                      type_value=[[','], 'str'],
                      description="Parameters for spinal cord straightening. Separate arguments with ','."
                                  "\nalgo_fitting: {hanning,nurbs} algorithm for curve fitting. Default=nurbs"
                                  "\nprecision: [1.0,inf[. Precision factor of straightening, related to the number of slices. Increasing this parameter increases the precision along with increased computational time. Not taken into account with hanning fitting method. Default=2"
                                  "\nthreshold_distance: [0.0,inf[. Threshold at which voxels are not considered into displacement. Increase this threshold if the image is blackout around the spinal cord too much. Default=10"
                                  "\naccuracy_results: {0, 1} Disable/Enable computation of accuracy results after straightening. Default=0"
                                  "\ntemplate_orientation: {0, 1} Disable/Enable orientation of the straight image to be the same as the template. Default=0",
                      mandatory=False,
                      example="algo_fitting=nurbs")
    parser.add_option(name="-params",
                      type_value=None,
                      description="Parameters for spinal cord straightening. Separate arguments with ','."
                                  "\nalgo_fitting: {hanning,nurbs} algorithm for curve fitting. Default=nurbs"
                                  "\nprecision: [1.0,inf[. Precision factor of straightening, related to the number of slices. Increasing this parameter increases the precision along with a loss of time. Is not taken into account with hanning fitting method. Default=2.0"
                                  "\nthreshold_distance: [0.0,inf[. Threshold for which voxels are not considered into displacement. Default=1.0",
                      mandatory=False,
                      deprecated_by='-param')

    parser.add_option(name='-qc',
                      type_value='multiple_choice',
                      description='Output images for quality control.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value='0')

    return parser


# MAIN
# ==========================================================================================
def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = get_parser()
    arguments = parser.parse(args)

    # assigning variables to arguments
    input_filename = arguments["-i"]
    centerline_file = arguments["-s"]

    sc_straight = SpinalCordStraightener(input_filename, centerline_file)

    if "-ref" in arguments:
        sc_straight.use_straight_reference = True
        sc_straight.centerline_reference_filename = str(arguments["-ref"])

    if "-disks-input" in arguments:
        if not sc_straight.use_straight_reference:
            sct.printv('Warning: disks position are not yet taken into account if reference is not provided.')
        else:
            sc_straight.disks_input_filename = str(arguments["-disks-input"])
            sc_straight.precision = 4.0
    if "-disks-ref" in arguments:
        if not sc_straight.use_straight_reference:
            sct.printv('Warning: disks position are not yet taken into account if reference is not provided.')
        else:
            sc_straight.disks_ref_filename = str(arguments["-disks-ref"])
            sc_straight.precision = 4.0

    # Handling optional arguments
    if "-r" in arguments:
        sc_straight.remove_temp_files = int(arguments["-r"])
    if "-x" in arguments:
        sc_straight.interpolation_warp = str(arguments["-x"])
    if "-o" in arguments:
        sc_straight.output_filename = str(arguments["-o"])
    if '-ofolder' in arguments:
        sc_straight.path_output = arguments['-ofolder']
    else:
        sc_straight.path_output = './'
    if "-v" in arguments:
        sc_straight.verbose = int(arguments["-v"])
    # if "-cpu-nb" in arguments:
    #     sc_straight.cpu_number = int(arguments["-cpu-nb"])
    if '-qc' in arguments:
        sc_straight.qc = int(arguments['-qc'])

    if '-disable-straight2curved' in arguments:
        sc_straight.straight2curved = False
    if '-disable-curved2straight' in arguments:
        sc_straight.curved2straight = False

    if '-resample' in arguments:
        sc_straight.resample_factor = arguments['-resample']

    if "-param" in arguments:
        params_user = arguments['-param']
        # update registration parameters
        for param in params_user:
            param_split = param.split('=')
            if param_split[0] == 'algo_fitting':
                sc_straight.algo_fitting = param_split[1]
                if sc_straight.algo_fitting == 'hanning':
                    sct.printv("WARNING: hanning has been disabled in this function. The fitting algorithm has been changed to NURBS.", type='warning')
                    sc_straight.algo_fitting = 'nurbs'
            if param_split[0] == 'precision':
                sc_straight.precision = float(param_split[1])
            if param_split[0] == 'threshold_distance':
                sc_straight.threshold_distance = float(param_split[1])
            if param_split[0] == 'accuracy_results':
                sc_straight.accuracy_results = int(param_split[1])
            if param_split[0] == 'template_orientation':
                sc_straight.template_orientation = int(param_split[1])

    sc_straight.straighten()


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # call main function
    main()
