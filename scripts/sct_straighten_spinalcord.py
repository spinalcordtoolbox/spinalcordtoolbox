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

from __future__ import division, absolute_import

import sys, os, time, bisect

import numpy as np
from nibabel import Nifti1Image, save
import tqdm

import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import get_centerline
from msct_parser import Parser
from msct_types import Centerline
from sct_apply_transfo import Transform
import sct_utils as sct


def generate_qc(fn_input, fn_centerline, fn_output, args, path_qc):
    """
    Generate a QC entry allowing to quickly review the straightening process.
    """

    import spinalcordtoolbox.reports.qc as qc
    import spinalcordtoolbox.reports.slice as qcslice

    # Just display the straightened spinal cord
    img_out = Image(fn_output)
    foreground = qcslice.Sagittal([img_out]).single()[0]

    qc.add_entry(
     src=fn_input,
     process="sct_straighten_spinalcord",
     args=args,
     path_qc=path_qc,
     plane="Sagittal",
     foreground=foreground,
    )


class SpinalCordStraightener(object):

    def __init__(self, input_filename, centerline_filename, debug=0, deg_poly=10,
                 interpolation_warp='spline', rm_tmp_files=1, verbose=1, algo_fitting='bspline',
                 precision=2.0, threshold_distance=10, output_filename=''):
        self.input_filename = input_filename
        self.centerline_filename = centerline_filename
        self.output_filename = output_filename
        self.debug = debug
        self.deg_poly = deg_poly  # maximum degree of polynomial function for fitting centerline.
        # the FOV due to the curvature of the spinal cord
        self.interpolation_warp = interpolation_warp
        self.remove_temp_files = rm_tmp_files  # remove temporary files
        self.verbose = verbose
        self.algo_fitting = algo_fitting  # 'bspline' or 'nurbs'
        self.precision = precision
        self.threshold_distance = threshold_distance
        self.path_output = ""
        self.use_straight_reference = False
        self.centerline_reference_filename = ""
        self.discs_input_filename = ""
        self.discs_ref_filename = ""

        self.mse_straightening = 0.0
        self.max_distance_straightening = 0.0

        self.curved2straight = True
        self.straight2curved = True

        self.speed_factor = 1.0
        self.accuracy_results = 0

        self.elapsed_time = 0.0
        self.elapsed_time_accuracy = 0.0

        self.template_orientation = 0
        self.xy_size = 35  # in mm

        self.path_qc = None

    def straighten(self):
        # Initialization
        fname_anat = self.input_filename
        fname_centerline = self.centerline_filename
        fname_output = self.output_filename
        remove_temp_files = self.remove_temp_files
        verbose = self.verbose
        interpolation_warp = self.interpolation_warp
        algo_fitting = self.algo_fitting

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
        img_src = Image(fname_anat).save(os.path.join(path_tmp, "data.nii"))
        img_centerline = Image(fname_centerline).save(os.path.join(path_tmp, "centerline.nii.gz"))

        if self.use_straight_reference:
            Image(self.centerline_reference_filename).save(os.path.join(path_tmp, "centerline_ref.nii.gz"))
        if self.discs_input_filename != '':
            Image(self.discs_input_filename).save(os.path.join(path_tmp, "labels_input.nii.gz"))
        if self.discs_ref_filename != '':
            Image(self.discs_ref_filename).save(os.path.join(path_tmp, "labels_ref.nii.gz"))

        # go to tmp folder
        curdir = os.getcwd()
        os.chdir(path_tmp)

        try:
            # Change orientation of the input centerline into RPI
            sct.printv("\nOrient centerline to RPI orientation...", verbose)
            image_centerline = Image("centerline.nii.gz").change_orientation("RPI").save("centerline_rpi.nii.gz", mutable=True)

            # Get dimension
            sct.printv('\nGet dimensions...', verbose)
            nx, ny, nz, nt, px, py, pz, pt = image_centerline.dim
            sct.printv('.. matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)
            sct.printv('.. voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm', verbose)
            if self.speed_factor != 1.0:
                intermediate_resampling = True
                px_r, py_r, pz_r = px * self.speed_factor, py * self.speed_factor, pz * self.speed_factor
            else:
                intermediate_resampling = False

            if intermediate_resampling:
                sct.mv('centerline_rpi.nii.gz', 'centerline_rpi_native.nii.gz')
                pz_native = pz

                sct.run(['sct_resample', '-i', 'centerline_rpi_native.nii.gz', '-mm', str(px_r) + 'x' + str(py_r) + 'x' + str(pz_r), '-o', 'centerline_rpi.nii.gz'])
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
            if algo_fitting == 'nurbs':
                number_of_points = int(self.precision * (float(nz) / pz))
                if number_of_points < 100:
                    number_of_points *= 50
                if number_of_points == 0:
                    number_of_points = 50
            else:
                number_of_points = nz

            # 2. extract bspline fitting of the centerline, and its derivatives
            img_ctl = Image('centerline_rpi.nii.gz')
            _, arr_ctl, arr_ctl_der = get_centerline(img_ctl, algo_fitting=algo_fitting, verbose=verbose)
            # Transform centerline and derivatives to physical coordinate system
            arr_ctl_phys = img_ctl.transfo_pix2phys(
                [[arr_ctl[0][i], arr_ctl[1][i], arr_ctl[2][i]] for i in range(len(arr_ctl[0]))])
            x_centerline, y_centerline, z_centerline = arr_ctl_phys[:, 0], arr_ctl_phys[:, 1], arr_ctl_phys[:, 2]
            # arr_ctl_der_phys = img_ctl.transfo_pix2phys(
            #     [[arr_ctl_der[0][i], arr_ctl_der[1][i], 1] for i in range(len(arr_ctl_der[0]))])
            # x_centerline_deriv, y_centerline_deriv = arr_ctl_der_phys[:, 0], arr_ctl_der_phys[:, 1]
            x_centerline_deriv, y_centerline_deriv = arr_ctl_der[0][:] * px, arr_ctl_der[1][:] * py
            # Construct centerline object
            centerline = Centerline(x_centerline.tolist(), y_centerline.tolist(), z_centerline.tolist(),
                                    x_centerline_deriv.tolist(), y_centerline_deriv.tolist(),
                                    np.ones_like(x_centerline_deriv).tolist())

            number_of_points = centerline.number_of_points

            # ==========================================================================================
            sct.printv("\nCreate the straight space and the safe zone...", verbose)
            # 3. compute length of centerline
            # compute the length of the spinal cord based on fitted centerline and size of centerline in z direction

            # Computation of the safe zone.
            # The safe zone is defined as the length of the spinal cord for which an axial segmentation will be complete
            # The safe length (to remove) is computed using the safe radius (given as parameter) and the angle of the
            # last centerline point with the inferior-superior direction. Formula: Ls = Rs * sin(angle)
            # Calculate Ls for both edges and remove appropriate number of centerline points
            radius_safe = 0.0  # mm

            # inferior edge
            u = centerline.derivatives[0]
            v = np.array([0, 0, -1])

            angle_inferior = np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
            length_safe_inferior = radius_safe * np.sin(angle_inferior)

            # superior edge
            u = centerline.derivatives[-1]
            v = np.array([0, 0, 1])
            angle_superior = np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
            length_safe_superior = radius_safe * np.sin(angle_superior)

            # remove points
            inferior_bound = bisect.bisect(centerline.progressive_length, length_safe_inferior) - 1
            superior_bound = centerline.number_of_points - bisect.bisect(centerline.progressive_length_inverse, length_safe_superior)

            z_centerline = centerline.points[:, 2]
            length_centerline = centerline.length
            size_z_centerline = z_centerline[-1] - z_centerline[0]

            # compute the size factor between initial centerline and straight bended centerline
            factor_curved_straight = length_centerline / size_z_centerline
            middle_slice = (z_centerline[0] + z_centerline[-1]) / 2.0

            bound_curved = [z_centerline[inferior_bound], z_centerline[superior_bound]]
            bound_straight = [(z_centerline[inferior_bound] - middle_slice) * factor_curved_straight + middle_slice,
                              (z_centerline[superior_bound] - middle_slice) * factor_curved_straight + middle_slice]

            if verbose == 2:
                sct.printv("Length of spinal cord = " + str(length_centerline))
                sct.printv("Size of spinal cord in z direction = " + str(size_z_centerline))
                sct.printv("Ratio length/size = " + str(factor_curved_straight))
                sct.printv("Safe zone boundaries: ")
                sct.printv("Curved space = " + str(bound_curved))
                sct.printv("Straight space = " + str(bound_straight))

            # 4. compute and generate straight space
            # points along curved centerline are already regularly spaced.
            # calculate position of points along straight centerline

            # Create straight NIFTI volumes. TODO: maybe this if case is not needed?
            # ==========================================================================================
            if self.use_straight_reference:
                image_centerline_pad = Image('centerline_rpi.nii.gz')
                nx, ny, nz, nt, px, py, pz, pt = image_centerline_pad.dim

                fname_ref = 'centerline_ref_rpi.nii.gz'
                image_centerline_straight = Image('centerline_ref.nii.gz')\
                    .change_orientation("RPI")\
                    .save(fname_ref, mutable=True)
                nx_s, ny_s, nz_s, nt_s, px_s, py_s, pz_s, pt_s = image_centerline_straight.dim
                # TODO: update this chunk below to work with physical coordinates
                _, arr_ctl, arr_ctl_der = get_centerline(image_centerline_straight, algo_fitting=algo_fitting,
                                                         verbose=verbose)
                x_centerline, y_centerline, z_centerline = arr_ctl
                x_centerline_deriv, y_centerline_deriv = arr_ctl_der
                centerline_straight = Centerline(x_centerline.tolist(), y_centerline.tolist(), z_centerline.tolist(),
                                                 x_centerline_deriv.tolist(), y_centerline_deriv.tolist(),
                                                 np.ones_like(x_centerline_deriv).tolist())

                # Prepare warping fields headers
                hdr_warp = image_centerline_pad.hdr.copy()
                hdr_warp.set_data_dtype('float32')
                hdr_warp_s = image_centerline_straight.hdr.copy()
                hdr_warp_s.set_data_dtype('float32')

                if self.discs_input_filename != "" and self.discs_ref_filename != "":
                    discs_input_image = Image('labels_input.nii.gz')
                    coord = discs_input_image.getNonZeroCoordinates(sorting='z', reverse_coord=True)
                    coord_physical = []
                    for c in coord:
                        c_p = discs_input_image.transfo_pix2phys([[c.x, c.y, c.z]]).tolist()[0]
                        c_p.append(c.value)
                        coord_physical.append(c_p)
                    centerline.compute_vertebral_distribution(coord_physical)
                    centerline.save_centerline(image=discs_input_image, fname_output='discs_input_image.nii.gz')

                    discs_ref_image = Image('labels_ref.nii.gz')
                    coord = discs_ref_image.getNonZeroCoordinates(sorting='z', reverse_coord=True)
                    coord_physical = []
                    for c in coord:
                        c_p = discs_ref_image.transfo_pix2phys([[c.x, c.y, c.z]]).tolist()[0]
                        c_p.append(c.value)
                        coord_physical.append(c_p)
                    centerline_straight.compute_vertebral_distribution(coord_physical)
                    centerline_straight.save_centerline(image=discs_ref_image, fname_output='discs_ref_image.nii.gz')

            else:
                sct.printv('\nPad input volume to account for spinal cord length...', verbose)

                start_point = (z_centerline[0] - middle_slice) * factor_curved_straight + middle_slice
                end_point = (z_centerline[-1] - middle_slice) * factor_curved_straight + middle_slice

                offset_z = 0

                # if the destination image is resampled, we still create the straight reference space with the native resolution. # TODO: Maybe this if case is not needed?
                if intermediate_resampling:
                    padding_z = int(np.ceil(1.5 * ((length_centerline - size_z_centerline) / 2.0) / pz_native))
                    sct.run(['sct_image', '-i', 'centerline_rpi_native.nii.gz', '-o', 'tmp.centerline_pad_native.nii.gz', '-pad', '0,0,' + str(padding_z)])
                    image_centerline_pad = Image('centerline_rpi_native.nii.gz')
                    nx, ny, nz, nt, px, py, pz, pt = image_centerline_pad.dim
                    start_point_coord_native = image_centerline_pad.transfo_phys2pix([[0, 0, start_point]])[0]
                    end_point_coord_native = image_centerline_pad.transfo_phys2pix([[0, 0, end_point]])[0]
                    straight_size_x = int(self.xy_size / px)
                    straight_size_y = int(self.xy_size / py)
                    warp_space_x = [int(np.round(nx / 2)) - straight_size_x, int(np.round(nx / 2)) + straight_size_x]
                    warp_space_y = [int(np.round(ny / 2)) - straight_size_y, int(np.round(ny / 2)) + straight_size_y]
                    if warp_space_x[0] < 0:
                        warp_space_x[1] += warp_space_x[0] - 2
                        warp_space_x[0] = 0
                    if warp_space_y[0] < 0:
                        warp_space_y[1] += warp_space_y[0] - 2
                        warp_space_y[0] = 0

                    spec = dict((
                     (0, warp_space_x),
                     (1, warp_space_y),
                     (2, (0, end_point_coord_native[2] - start_point_coord_native[2])),
                    ))
                    msct_image.spatial_crop(Image("tmp.centerline_pad_native.nii.gz"), spec).save("tmp.centerline_pad_crop_native.nii.gz")

                    fname_ref = 'tmp.centerline_pad_crop_native.nii.gz'
                    offset_z = 4
                else:
                    fname_ref = 'tmp.centerline_pad_crop.nii.gz'

                nx, ny, nz, nt, px, py, pz, pt = image_centerline.dim
                padding_z = int(np.ceil(1.5 * ((length_centerline - size_z_centerline) / 2.0) / pz)) + offset_z
                sct.run(['sct_image', '-i', 'centerline_rpi.nii.gz', '-o', 'tmp.centerline_pad.nii.gz', '-pad', '0,0,' + str(padding_z)])
                image_centerline_pad = Image('centerline_rpi.nii.gz')
                nx, ny, nz, nt, px, py, pz, pt = image_centerline_pad.dim
                hdr_warp = image_centerline_pad.hdr.copy()
                hdr_warp.set_data_dtype('float32')
                start_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, start_point]])[0]
                end_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, end_point]])[0]

                straight_size_x = int(self.xy_size / px)
                straight_size_y = int(self.xy_size / py)
                warp_space_x = [int(np.round(nx / 2)) - straight_size_x, int(np.round(nx / 2)) + straight_size_x]
                warp_space_y = [int(np.round(ny / 2)) - straight_size_y, int(np.round(ny / 2)) + straight_size_y]

                if warp_space_x[0] < 0:
                    warp_space_x[1] += warp_space_x[0] - 2
                    warp_space_x[0] = 0
                if warp_space_x[1] >= nx:
                    warp_space_x[1] = nx - 1
                if warp_space_y[0] < 0:
                    warp_space_y[1] += warp_space_y[0] - 2
                    warp_space_y[0] = 0
                if warp_space_y[1] >= ny:
                    warp_space_y[1] = ny - 1

                spec = dict((
                 (0, warp_space_x),
                 (1, warp_space_y),
                 (2, (0, end_point_coord[2] - start_point_coord[2] + offset_z)),
                ))
                msct_image.spatial_crop(Image("tmp.centerline_pad.nii.gz"), spec).save("tmp.centerline_pad_crop.nii.gz")

                image_centerline_straight = Image('tmp.centerline_pad_crop.nii.gz')
                nx_s, ny_s, nz_s, nt_s, px_s, py_s, pz_s, pt_s = image_centerline_straight.dim
                hdr_warp_s = image_centerline_straight.hdr.copy()
                hdr_warp_s.set_data_dtype('float32')

                if self.template_orientation == 1:
                    raise NotImplementedError()


                start_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, start_point]])[0]
                end_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, end_point]])[0]

                number_of_voxel = nx * ny * nz
                sct.printv("Number of voxel = " + str(number_of_voxel))

                time_centerlines = time.time()

                coord_straight = np.empty((number_of_points,3))
                coord_straight[...,0] = int(np.round(nx_s / 2))
                coord_straight[...,1] = int(np.round(ny_s / 2))
                coord_straight[...,2] = np.linspace(0, end_point_coord[2] - start_point_coord[2], number_of_points)
                coord_phys_straight = image_centerline_straight.transfo_pix2phys(coord_straight)
                derivs_straight = np.empty((number_of_points,3))
                derivs_straight[...,0] = derivs_straight[...,1] = 0
                derivs_straight[...,2] = 1
                dx_straight, dy_straight, dz_straight = derivs_straight.T
                centerline_straight = Centerline(coord_phys_straight[:, 0], coord_phys_straight[:, 1], coord_phys_straight[:, 2],
                                                 dx_straight, dy_straight, dz_straight)

                time_centerlines = time.time() - time_centerlines
                sct.printv('Time to generate centerline: ' + str(np.round(time_centerlines * 1000.0)) + ' ms', verbose)


            if verbose == 2:
                import matplotlib.pyplot as plt
                from datetime import datetime
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
                plt.savefig('fig_straighten_' + datetime.now().strftime("%y%m%d%H%M%S%f") + '.png')
                plt.close()

            #alignment_mode = 'length'
            alignment_mode = 'levels'

            lookup_curved2straight = list(range(centerline.number_of_points))
            if self.discs_input_filename != "":
                # create look-up table curved to straight
                for index in range(centerline.number_of_points):
                    disc_label = centerline.l_points[index]
                    if alignment_mode == 'length':
                        relative_position = centerline.dist_points[index]
                    else:
                        relative_position = centerline.dist_points_rel[index]
                    idx_closest = centerline_straight.get_closest_to_absolute_position(disc_label, relative_position, backup_index=index, backup_centerline=centerline_straight, mode=alignment_mode)
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

            lookup_straight2curved = list(range(centerline_straight.number_of_points))
            if self.discs_input_filename != "":
                for index in range(centerline_straight.number_of_points):
                    disc_label = centerline_straight.l_points[index]
                    if alignment_mode == 'length':
                        relative_position = centerline_straight.dist_points[index]
                    else:
                        relative_position = centerline_straight.dist_points_rel[index]
                    idx_closest = centerline.get_closest_to_absolute_position(disc_label, relative_position, backup_index=index, backup_centerline=centerline_straight, mode=alignment_mode)
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
                for u in tqdm.tqdm(range(nz_s)):
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
                    # Invert Z coordinate as ITK & ANTs physical coordinate system is LPS- (RAI+)
                    # while ours is LPI-
                    # Refs: https://sourceforge.net/p/advants/discussion/840261/thread/2a1e9307/#fb5a
                    #  https://www.slicer.org/wiki/Coordinate_systems
                    displacements_straight[:, 2] = -displacements_straight[:, 2]
                    displacements_straight[indexes_out_distance_straight] = [100000.0, 100000.0, 100000.0]

                    data_warp_curved2straight[indexes_straight[:, 0], indexes_straight[:, 1], indexes_straight[:, 2], 0, :] = -displacements_straight

            if self.straight2curved:
                for u in tqdm.tqdm(range(nz)):
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

                    displacements_curved[:, 2] = -displacements_curved[:, 2]
                    displacements_curved[indexes_out_distance_curved] = [100000.0, 100000.0, 100000.0]

                    data_warp_straight2curved[indexes[:, 0], indexes[:, 1], indexes[:, 2], 0, :] = -displacements_curved

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
                sct.printv('\nApply transformation to input image...', verbose)
                s, o = sct.run(['sct_apply_transfo', '-i', 'data.nii', '-d', fname_ref, '-o', 'tmp.anat_rigid_warp.nii.gz', '-w', 'tmp.curve2straight.nii.gz', '-x', interpolation_warp], verbose)
                for line in o.splitlines():
                    sct.printv("> %s" % line, verbose=verbose)

            if self.accuracy_results:
                time_accuracy_results = time.time()
                # compute the error between the straightened centerline/segmentation and the central vertical line.
                # Ideally, the error should be zero.
                # Apply deformation to input image
                sct.printv('\nApply transformation to centerline image...', verbose)
                Transform(input_filename='centerline.nii.gz', fname_dest=fname_ref,
                          output_filename="tmp.centerline_straight.nii.gz", interp="nn",
                          warp="tmp.curve2straight.nii.gz", verbose=verbose).apply()
                file_centerline_straight = Image('tmp.centerline_straight.nii.gz', verbose=verbose)
                nx, ny, nz, nt, px, py, pz, pt = file_centerline_straight.dim
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
                        dist = np.sqrt(dist)
                        if dist > self.max_distance_straightening:
                            self.max_distance_straightening = dist
                        count_mean += 1
                self.mse_straightening = np.sqrt(self.mse_straightening / float(count_mean))

                self.elapsed_time_accuracy = time.time() - time_accuracy_results

        except Exception as e:
            sct.printv('WARNING: Exception during Straightening:', 1, 'warning')
            sct.printv('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), 1, 'warning')
            sct.printv(str(e), 1, 'warning')
            raise
        finally:
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
            sct.rmtree(path_tmp)

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

        return fname_straight


def get_parser():
    # Initialize parser
    parser = Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description("This program takes as input an anatomic image and the spinal cord centerline (or "
                                 "segmentation), and returns the an image of a straightened spinal cord. Reference: "
                                 "De Leener B, Mangeat G, Dupont S, Martin AR, Callot V, Stikov N, Fehlings MG, "
                                 "Cohen-Adad J. Topologically-preserving straightening of spinal cord MRI. J Magn "
                                 "Reson Imaging. 2017 Oct;46(4):1209-1219")
    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="Input image with curved spinal cord.",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name="-s",
                      type_value="image_nifti",
                      description="Spinal cord centerline (or segmentation) of the input image. To obtain the centerline"
                                  "you can use sct_get_centerline. To obtain the segmentation you can use sct_propseg"
                                  "or sct_deepseg_sc.",
                      mandatory=True,
                      example="centerline.nii.gz")
    parser.add_option(name="-c",
                      type_value=None,
                      description="centerline or segmentation.",
                      mandatory=False,
                      deprecated_by='-s')
    parser.add_option(name="-dest",
                      type_value="image_nifti",
                      description="Spinal cord centerline (or segmentation) of a destination image (which could be "
                                  "straight or curved). An "
                                  "algorithm scales the length of the input centerline to match that of the "
                                  "destination centerline. If using -ldisc_input and -ldisc_dest with this parameter, "
                                  "instead of linear scaling, the source centerline will be non-linearly matched so "
                                  "that the inter-vertebral discs of the input image will match that of the "
                                  "destination image. This feature is particularly useful for registering to a "
                                  "template while accounting for disc alignment.",
                      mandatory=False,
                      example="centerline.nii.gz")
    parser.add_option(name="-ldisc_input",
                      type_value="image_nifti",
                      description="Labels located at the posterior edge of the intervertebral discs, for the input "
                                  "image (-i). All disc covering the region of interest should be provided. E.g., if "
                                  "you are interested in levels C2 to C7, then you should provide disc labels 2,3,4,5,"
                                  "6,7). More details about label creation at "
                                  "http://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels/.\n"  # TODO (Julien) update this link
                                  "This option must be used with the -ldisc_dest parameter.",
                      mandatory=False,
                      example="ldisc_input.nii.gz")
    parser.add_option(name="-ldisc_dest",
                      type_value="image_nifti",
                      description="Labels located at the posterior edge of the intervertebral discs, for the "
                                  "destination file (-dest). The same comments as in -ldisc_input apply.\n"
                                  "This option must be used with the -ldisc_input parameter.",
                      mandatory=False,
                      example="ldisc_dest.nii.gz")
    parser.add_option(name="-disable-straight2curved",
                      type_value=None,
                      description="Disable straight to curved transformation computation, in case you do not need the "
                                  "output warping field straight-->curve (faster).",
                      mandatory=False)
    parser.add_option(name="-disable-curved2straight",
                      type_value=None,
                      description="Disable curved to straight transformation computation, in case you do not need the "
                                  "output warping field curve-->straight (faster).",
                      mandatory=False)
    parser.add_option(name="-speed_factor",
                      type_value='float',
                      description='Acceleration factor for the calculation of the straightening warping field.'
                                  ' This speed factor enables an intermediate resampling to a lower resolution, which '
                                  'decreases the computational time at the cost of lower accuracy.'
                                  ' A speed factor of 2 means that the input image will be downsampled by a factor 2 '
                                  'before calculating the straightening warping field. For example, a 1x1x1 mm^3 image '
                                  'will be downsampled to 2x2x2 mm3, providing a speed factor of approximately 8.'
                                  ' Note that accelerating the straightening process reduces the precision of the '
                                  'algorithm, and induces undesirable edges effects. Default=1 (no downsampling).',
                      mandatory=False,
                      default_value=1)
    parser.add_option(name="-xy_size",
                      type_value='float',
                      description='Change the size of the XY FOV, in mm. The resolution of the destination image is '
                                  'the same as that of the source image (-i).\n',
                      mandatory=False,
                      default_value=35.0)
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

    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=None)

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

    if "-dest" in arguments:
        sc_straight.use_straight_reference = True
        sc_straight.centerline_reference_filename = str(arguments["-dest"])

    if "-ldisc_input" in arguments:
        if not sc_straight.use_straight_reference:
            sct.printv('Warning: discs position are not taken into account if reference is not provided.')
        else:
            sc_straight.discs_input_filename = str(arguments["-ldisc_input"])
            sc_straight.precision = 4.0
    if "-ldisc_dest" in arguments:
        if not sc_straight.use_straight_reference:
            sct.printv('Warning: discs position are not taken into account if reference is not provided.')
        else:
            sc_straight.discs_ref_filename = str(arguments["-ldisc_dest"])
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

    verbose = int(arguments.get("-v", 0))
    sc_straight.verbose = verbose

    # if "-cpu-nb" in arguments:
    #     sc_straight.cpu_number = int(arguments["-cpu-nb"])

    path_qc = arguments.get("-qc", None)

    if '-disable-straight2curved' in arguments:
        sc_straight.straight2curved = False
    if '-disable-curved2straight' in arguments:
        sc_straight.curved2straight = False

    if '-speed_factor' in arguments:
        sc_straight.speed_factor = arguments['-speed_factor']

    if '-xy_size' in arguments:
        sc_straight.xy_size = arguments['-xy_size']

    if "-param" in arguments:
        params_user = arguments['-param']
        # update registration parameters
        for param in params_user:
            param_split = param.split('=')
            if param_split[0] == 'algo_fitting':
                sc_straight.algo_fitting = param_split[1]
                # if sc_straight.algo_fitting == 'hanning':
                #     sct.printv("WARNING: hanning has been disabled in this function. The fitting algorithm has been changed to NURBS.", type='warning')
                #     sc_straight.algo_fitting = 'nurbs'
            if param_split[0] == 'precision':
                sc_straight.precision = float(param_split[1])
            if param_split[0] == 'threshold_distance':
                sc_straight.threshold_distance = float(param_split[1])
            if param_split[0] == 'accuracy_results':
                sc_straight.accuracy_results = int(param_split[1])
            if param_split[0] == 'template_orientation':
                sc_straight.template_orientation = int(param_split[1])

    fname_straight = sc_straight.straighten()

    if sc_straight.curved2straight:

        if path_qc is not None:
           generate_qc(input_filename, centerline_file,
            fname_straight, args, os.path.abspath(path_qc))

        sct.display_viewer_syntax([fname_straight], verbose=verbose)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
