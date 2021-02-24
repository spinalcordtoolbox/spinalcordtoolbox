#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with spinal cord straightening

# TODO: move _get_centerline inside Class
# TODO: only input Image instead of file names

import os
import time
import logging
import inspect
import bisect

import numpy as np
from nibabel import Nifti1Image, save

from spinalcordtoolbox.types import Centerline
from spinalcordtoolbox.image import Image, spatial_crop, generate_output_file, pad_image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.utils.sys import sct_progress_bar, run_proc
from spinalcordtoolbox.utils.fs import tmp_create, rmtree, copy, mv, extract_fname

logger = logging.getLogger(__name__)


class SpinalCordStraightener(object):
    def __init__(self, input_filename, centerline_filename, debug=0, param_centerline=ParamCenterline(),
                 interpolation_warp='spline', rm_tmp_files=1, verbose=1, precision=2.0, threshold_distance=10,
                 output_filename='', path_output=''):
        self.input_filename = input_filename
        self.centerline_filename = centerline_filename
        self.output_filename = output_filename
        self.debug = debug
        self.interpolation_warp = interpolation_warp
        self.remove_temp_files = rm_tmp_files  # remove temporary files
        self.verbose = verbose
        self.precision = precision
        self.threshold_distance = threshold_distance
        self.path_output = path_output
        self.use_straight_reference = False
        self.centerline_reference_filename = ""
        self.discs_input_filename = ""
        self.discs_ref_filename = ""
        self.speed_factor = 1.0  # Speed parameter
        self.xy_size = 70  # in mm
        self.param_centerline = param_centerline

        # QC metrics
        self.accuracy_results = 0
        self.mse_straightening = 0.0
        self.max_distance_straightening = 0.0
        self.elapsed_time = 0.0
        self.elapsed_time_accuracy = 0.0

        # Outputs
        self.curved2straight = True
        self.straight2curved = True
        self.path_qc = None

        self.template_orientation = 0

    def straighten(self):
        """
        Straighten spinal cord. Steps: (everything is done in physical space)
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

        :return:
        """
        # Initialization
        fname_anat = self.input_filename
        fname_centerline = self.centerline_filename
        fname_output = self.output_filename
        remove_temp_files = self.remove_temp_files
        verbose = self.verbose
        interpolation_warp = self.interpolation_warp  # TODO: remove this

        # start timer
        start_time = time.time()

        # Extract path/file/extension
        path_anat, file_anat, ext_anat = extract_fname(fname_anat)

        path_tmp = tmp_create(basename="straighten_spinalcord")

        # Copying input data to tmp folder
        logger.info('Copy files to tmp folder...')
        Image(fname_anat, check_sform=True).save(os.path.join(path_tmp, "data.nii"))
        Image(fname_centerline, check_sform=True).save(os.path.join(path_tmp, "centerline.nii.gz"))

        if self.use_straight_reference:
            Image(self.centerline_reference_filename, check_sform=True).save(os.path.join(path_tmp, "centerline_ref.nii.gz"))
        if self.discs_input_filename != '':
            Image(self.discs_input_filename, check_sform=True).save(os.path.join(path_tmp, "labels_input.nii.gz"))
        if self.discs_ref_filename != '':
            Image(self.discs_ref_filename, check_sform=True).save(os.path.join(path_tmp, "labels_ref.nii.gz"))

        # go to tmp folder
        curdir = os.getcwd()
        os.chdir(path_tmp)

        # Change orientation of the input centerline into RPI
        image_centerline = Image("centerline.nii.gz").change_orientation("RPI").save("centerline_rpi.nii.gz",
                                                                                     mutable=True)

        # Get dimension
        nx, ny, nz, nt, px, py, pz, pt = image_centerline.dim
        if self.speed_factor != 1.0:
            intermediate_resampling = True
            px_r, py_r, pz_r = px * self.speed_factor, py * self.speed_factor, pz * self.speed_factor
        else:
            intermediate_resampling = False

        if intermediate_resampling:
            mv('centerline_rpi.nii.gz', 'centerline_rpi_native.nii.gz')
            pz_native = pz
            # TODO: remove system call
            run_proc(['sct_resample', '-i', 'centerline_rpi_native.nii.gz', '-mm',
                      str(px_r) + 'x' + str(py_r) + 'x' + str(pz_r), '-o', 'centerline_rpi.nii.gz'])
            image_centerline = Image('centerline_rpi.nii.gz')
            nx, ny, nz, nt, px, py, pz, pt = image_centerline.dim

        if np.min(image_centerline.data) < 0 or np.max(image_centerline.data) > 1:
            image_centerline.data[image_centerline.data < 0] = 0
            image_centerline.data[image_centerline.data > 1] = 1
            image_centerline.save()

        # 2. extract bspline fitting of the centerline, and its derivatives
        img_ctl = Image('centerline_rpi.nii.gz')
        centerline = _get_centerline(img_ctl, self.param_centerline, verbose)
        number_of_points = centerline.number_of_points

        # ==========================================================================================
        logger.info('Create the straight space and the safe zone')
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
        superior_bound = centerline.number_of_points - bisect.bisect(centerline.progressive_length_inverse,
                                                                     length_safe_superior)

        z_centerline = centerline.points[:, 2]
        length_centerline = centerline.length
        size_z_centerline = z_centerline[-1] - z_centerline[0]

        # compute the size factor between initial centerline and straight bended centerline
        factor_curved_straight = length_centerline / size_z_centerline
        middle_slice = (z_centerline[0] + z_centerline[-1]) / 2.0

        bound_curved = [z_centerline[inferior_bound], z_centerline[superior_bound]]
        bound_straight = [(z_centerline[inferior_bound] - middle_slice) * factor_curved_straight + middle_slice,
                          (z_centerline[superior_bound] - middle_slice) * factor_curved_straight + middle_slice]

        logger.info('Length of spinal cord: {}'.format(length_centerline))
        logger.info('Size of spinal cord in z direction: {}'.format(size_z_centerline))
        logger.info('Ratio length/size: {}'.format(factor_curved_straight))
        logger.info('Safe zone boundaries (curved space): {}'.format(bound_curved))
        logger.info('Safe zone boundaries (straight space): {}'.format(bound_straight))

        # 4. compute and generate straight space
        # points along curved centerline are already regularly spaced.
        # calculate position of points along straight centerline

        # Create straight NIFTI volumes.
        # ==========================================================================================
        # TODO: maybe this if case is not needed?
        if self.use_straight_reference:
            image_centerline_pad = Image('centerline_rpi.nii.gz')
            nx, ny, nz, nt, px, py, pz, pt = image_centerline_pad.dim

            fname_ref = 'centerline_ref_rpi.nii.gz'
            image_centerline_straight = Image('centerline_ref.nii.gz') \
                .change_orientation("RPI") \
                .save(fname_ref, mutable=True)
            centerline_straight = _get_centerline(image_centerline_straight, self.param_centerline, verbose)
            nx_s, ny_s, nz_s, nt_s, px_s, py_s, pz_s, pt_s = image_centerline_straight.dim

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
            logger.info('Pad input volume to account for spinal cord length...')

            start_point, end_point = bound_straight[0], bound_straight[1]
            offset_z = 0

            # if the destination image is resampled, we still create the straight reference space with the native
            # resolution.
            # TODO: Maybe this if case is not needed?
            if intermediate_resampling:
                padding_z = int(np.ceil(1.5 * ((length_centerline - size_z_centerline) / 2.0) / pz_native))
                run_proc(
                    ['sct_image', '-i', 'centerline_rpi_native.nii.gz', '-o', 'tmp.centerline_pad_native.nii.gz',
                     '-pad', '0,0,' + str(padding_z)])
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
                spatial_crop(Image("tmp.centerline_pad_native.nii.gz"), spec).save(
                    "tmp.centerline_pad_crop_native.nii.gz")

                fname_ref = 'tmp.centerline_pad_crop_native.nii.gz'
                offset_z = 4
            else:
                fname_ref = 'tmp.centerline_pad_crop.nii.gz'

            nx, ny, nz, nt, px, py, pz, pt = image_centerline.dim
            padding_z = int(np.ceil(1.5 * ((length_centerline - size_z_centerline) / 2.0) / pz)) + offset_z
            image_centerline_pad = pad_image(image_centerline, pad_z_i=padding_z, pad_z_f=padding_z)
            nx, ny, nz = image_centerline_pad.data.shape
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
            image_centerline_straight = spatial_crop(image_centerline_pad, spec)

            nx_s, ny_s, nz_s, nt_s, px_s, py_s, pz_s, pt_s = image_centerline_straight.dim
            hdr_warp_s = image_centerline_straight.hdr.copy()
            hdr_warp_s.set_data_dtype('float32')

            if self.template_orientation == 1:
                raise NotImplementedError()

            start_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, start_point]])[0]
            end_point_coord = image_centerline_pad.transfo_phys2pix([[0, 0, end_point]])[0]

            number_of_voxel = nx * ny * nz
            logger.debug('Number of voxels: {}'.format(number_of_voxel))

            time_centerlines = time.time()

            coord_straight = np.empty((number_of_points, 3))
            coord_straight[..., 0] = int(np.round(nx_s / 2))
            coord_straight[..., 1] = int(np.round(ny_s / 2))
            coord_straight[..., 2] = np.linspace(0, end_point_coord[2] - start_point_coord[2], number_of_points)
            coord_phys_straight = image_centerline_straight.transfo_pix2phys(coord_straight)
            derivs_straight = np.empty((number_of_points, 3))
            derivs_straight[..., 0] = derivs_straight[..., 1] = 0
            derivs_straight[..., 2] = 1
            dx_straight, dy_straight, dz_straight = derivs_straight.T
            centerline_straight = Centerline(coord_phys_straight[:, 0], coord_phys_straight[:, 1],
                                             coord_phys_straight[:, 2],
                                             dx_straight, dy_straight, dz_straight)

            time_centerlines = time.time() - time_centerlines
            logger.info('Time to generate centerline: {} ms'.format(np.round(time_centerlines * 1000.0)))

        if verbose == 2:
            # TODO: use OO
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

        # alignment_mode = 'length'
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
                idx_closest = centerline_straight.get_closest_to_absolute_position(disc_label, relative_position,
                                                                                   backup_index=index,
                                                                                   backup_centerline=centerline_straight,
                                                                                   mode=alignment_mode)
                if idx_closest is not None:
                    lookup_curved2straight[index] = idx_closest
                else:
                    lookup_curved2straight[index] = 0
        for p in range(0, len(lookup_curved2straight) // 2):
            if lookup_curved2straight[p] == lookup_curved2straight[p + 1]:
                lookup_curved2straight[p] = 0
            else:
                break
        for p in range(len(lookup_curved2straight) - 1, len(lookup_curved2straight) // 2, -1):
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
                idx_closest = centerline.get_closest_to_absolute_position(disc_label, relative_position,
                                                                          backup_index=index,
                                                                          backup_centerline=centerline_straight,
                                                                          mode=alignment_mode)
                if idx_closest is not None:
                    lookup_straight2curved[index] = idx_closest
        for p in range(0, len(lookup_straight2curved) // 2):
            if lookup_straight2curved[p] == lookup_straight2curved[p + 1]:
                lookup_straight2curved[p] = 0
            else:
                break
        for p in range(len(lookup_straight2curved) - 1, len(lookup_straight2curved) // 2, -1):
            if lookup_straight2curved[p] == lookup_straight2curved[p - 1]:
                lookup_straight2curved[p] = 0
            else:
                break
        lookup_straight2curved = np.array(lookup_straight2curved)

        # Create volumes containing curved and straight warping fields
        data_warp_curved2straight = np.zeros((nx_s, ny_s, nz_s, 1, 3))
        data_warp_straight2curved = np.zeros((nx, ny, nz, 1, 3))

        # 5. compute transformations
        # Curved and straight images and the same dimensions, so we compute both warping fields at the same time.
        # b. determine which plane of spinal cord centreline it is included

        if self.curved2straight:
            for u in sct_progress_bar(range(nz_s)):
                x_s, y_s, z_s = np.mgrid[0:nx_s, 0:ny_s, u:u + 1]
                indexes_straight = np.array(list(zip(x_s.ravel(), y_s.ravel(), z_s.ravel())))
                physical_coordinates_straight = image_centerline_straight.transfo_pix2phys(indexes_straight)
                nearest_indexes_straight = centerline_straight.find_nearest_indexes(physical_coordinates_straight)
                distances_straight = centerline_straight.get_distances_from_planes(physical_coordinates_straight,
                                                                                   nearest_indexes_straight)
                lookup = lookup_straight2curved[nearest_indexes_straight]
                indexes_out_distance_straight = np.logical_or(
                    np.logical_or(distances_straight > self.threshold_distance,
                                  distances_straight < -self.threshold_distance), lookup == 0)
                projected_points_straight = centerline_straight.get_projected_coordinates_on_planes(
                    physical_coordinates_straight, nearest_indexes_straight)
                coord_in_planes_straight = centerline_straight.get_in_plans_coordinates(projected_points_straight,
                                                                                        nearest_indexes_straight)

                coord_straight2curved = centerline.get_inverse_plans_coordinates(coord_in_planes_straight, lookup)
                displacements_straight = coord_straight2curved - physical_coordinates_straight
                # Invert Z coordinate as ITK & ANTs physical coordinate system is LPS- (RAI+)
                # while ours is LPI-
                # Refs: https://sourceforge.net/p/advants/discussion/840261/thread/2a1e9307/#fb5a
                #  https://www.slicer.org/wiki/Coordinate_systems
                displacements_straight[:, 2] = -displacements_straight[:, 2]
                displacements_straight[indexes_out_distance_straight] = [100000.0, 100000.0, 100000.0]

                data_warp_curved2straight[indexes_straight[:, 0], indexes_straight[:, 1], indexes_straight[:, 2], 0, :]\
                    = -displacements_straight

        if self.straight2curved:
            for u in sct_progress_bar(range(nz)):
                x, y, z = np.mgrid[0:nx, 0:ny, u:u + 1]
                indexes = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())))
                physical_coordinates = image_centerline_pad.transfo_pix2phys(indexes)
                nearest_indexes_curved = centerline.find_nearest_indexes(physical_coordinates)
                distances_curved = centerline.get_distances_from_planes(physical_coordinates,
                                                                        nearest_indexes_curved)
                lookup = lookup_curved2straight[nearest_indexes_curved]
                indexes_out_distance_curved = np.logical_or(
                    np.logical_or(distances_curved > self.threshold_distance,
                                  distances_curved < -self.threshold_distance), lookup == 0)
                projected_points_curved = centerline.get_projected_coordinates_on_planes(physical_coordinates,
                                                                                         nearest_indexes_curved)
                coord_in_planes_curved = centerline.get_in_plans_coordinates(projected_points_curved,
                                                                             nearest_indexes_curved)

                coord_curved2straight = centerline_straight.points[lookup]
                coord_curved2straight[:, 0:2] += coord_in_planes_curved[:, 0:2]
                coord_curved2straight[:, 2] += distances_curved

                displacements_curved = coord_curved2straight - physical_coordinates

                displacements_curved[:, 2] = -displacements_curved[:, 2]
                displacements_curved[indexes_out_distance_curved] = [100000.0, 100000.0, 100000.0]

                data_warp_straight2curved[indexes[:, 0], indexes[:, 1], indexes[:, 2], 0, :] = -displacements_curved

        # Creation of the safe zone based on pre-calculated safe boundaries
        coord_bound_curved_inf, coord_bound_curved_sup = image_centerline_pad.transfo_phys2pix(
            [[0, 0, bound_curved[0]]]), image_centerline_pad.transfo_phys2pix([[0, 0, bound_curved[1]]])
        coord_bound_straight_inf, coord_bound_straight_sup = image_centerline_straight.transfo_phys2pix(
            [[0, 0, bound_straight[0]]]), image_centerline_straight.transfo_phys2pix([[0, 0, bound_straight[1]]])

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
            logger.info('Warping field generated: tmp.curve2straight.nii.gz')

        if self.straight2curved:
            img = Nifti1Image(data_warp_straight2curved, None, hdr_warp)
            save(img, 'tmp.straight2curve.nii.gz')
            logger.info('Warping field generated: tmp.straight2curve.nii.gz')

        image_centerline_straight.save(fname_ref)
        if self.curved2straight:
            logger.info('Apply transformation to input image...')
            run_proc(['isct_antsApplyTransforms',
                      '-d', '3',
                      '-r', fname_ref,
                      '-i', 'data.nii',
                      '-o', 'tmp.anat_rigid_warp.nii.gz',
                      '-t', 'tmp.curve2straight.nii.gz',
                      '-n', 'BSpline[3]'],
                     is_sct_binary=True,
                     verbose=verbose)

        if self.accuracy_results:
            time_accuracy_results = time.time()
            # compute the error between the straightened centerline/segmentation and the central vertical line.
            # Ideally, the error should be zero.
            # Apply deformation to input image
            logger.info('Apply transformation to centerline image...')
            run_proc(['isct_antsApplyTransforms',
                      '-d', '3',
                      '-r', fname_ref,
                      '-i', 'centerline.nii.gz',
                      '-o', 'tmp.centerline_straight.nii.gz',
                      '-t', 'tmp.curve2straight.nii.gz',
                      '-n', 'NearestNeighbor'],
                     is_sct_binary=True,
                     verbose=verbose)
            file_centerline_straight = Image('tmp.centerline_straight.nii.gz', verbose=verbose)
            nx, ny, nz, nt, px, py, pz, pt = file_centerline_straight.dim
            coordinates_centerline = file_centerline_straight.getNonZeroCoordinates(sorting='z')
            mean_coord = []
            for z in range(coordinates_centerline[0].z, coordinates_centerline[-1].z):
                temp_mean = [coord.value for coord in coordinates_centerline if coord.z == z]
                if temp_mean:
                    mean_value = np.mean(temp_mean)
                    mean_coord.append(
                        np.mean([[coord.x * coord.value / mean_value, coord.y * coord.value / mean_value]
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
                    dist = ((x0 - coord_z[0]) * px) ** 2 + ((y0 - coord_z[1]) * py) ** 2
                    self.mse_straightening += dist
                    dist = np.sqrt(dist)
                    if dist > self.max_distance_straightening:
                        self.max_distance_straightening = dist
                    count_mean += 1
            self.mse_straightening = np.sqrt(self.mse_straightening / float(count_mean))

            self.elapsed_time_accuracy = time.time() - time_accuracy_results

        os.chdir(curdir)

        # Generate output file (in current folder)
        # TODO: do not uncompress the warping field, it is too time consuming!
        logger.info('Generate output files...')
        if self.curved2straight:
            generate_output_file(os.path.join(path_tmp, "tmp.curve2straight.nii.gz"),
                                     os.path.join(self.path_output, "warp_curve2straight.nii.gz"), verbose)
        if self.straight2curved:
            generate_output_file(os.path.join(path_tmp, "tmp.straight2curve.nii.gz"),
                                     os.path.join(self.path_output, "warp_straight2curve.nii.gz"), verbose)

        # create ref_straight.nii.gz file that can be used by other SCT functions that need a straight reference space
        if self.curved2straight:
            copy(os.path.join(path_tmp, "tmp.anat_rigid_warp.nii.gz"),
                     os.path.join(self.path_output, "straight_ref.nii.gz"))
            # move straightened input file
            if fname_output == '':
                fname_straight = generate_output_file(os.path.join(path_tmp, "tmp.anat_rigid_warp.nii.gz"),
                                                          os.path.join(self.path_output,
                                                                       file_anat + "_straight" + ext_anat), verbose)
            else:
                fname_straight = generate_output_file(os.path.join(path_tmp, "tmp.anat_rigid_warp.nii.gz"),
                                                          os.path.join(self.path_output, fname_output),
                                                          verbose)  # straightened anatomic

        # Remove temporary files
        if remove_temp_files:
            logger.info('Remove temporary files...')
            rmtree(path_tmp)

        if self.accuracy_results:
            logger.info('Maximum x-y error: {} mm'.format(self.max_distance_straightening))
            logger.info('Accuracy of straightening (MSE): {} mm'.format(self.mse_straightening))

        # display elapsed time
        self.elapsed_time = int(np.round(time.time() - start_time))

        return fname_straight


def _get_centerline(img, param_centerline, verbose):
    nx, ny, nz, nt, px, py, pz, pt = img.dim
    _, arr_ctl, arr_ctl_der, _ = get_centerline(img, param_centerline, verbose=verbose)
    # Transform centerline to physical coordinate system
    arr_ctl_phys = img.transfo_pix2phys(
        [[arr_ctl[0][i], arr_ctl[1][i], arr_ctl[2][i]] for i in range(len(arr_ctl[0]))])
    x_centerline, y_centerline, z_centerline = arr_ctl_phys[:, 0], arr_ctl_phys[:, 1], arr_ctl_phys[:, 2]
    # Adjust derivatives with pixel size
    x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = arr_ctl_der[0][:] * px, \
        arr_ctl_der[1][:] * py, \
        arr_ctl_der[2][:] * pz
    # Construct centerline object
    return Centerline(x_centerline.tolist(), y_centerline.tolist(), z_centerline.tolist(),
                      x_centerline_deriv.tolist(), y_centerline_deriv.tolist(), z_centerline_deriv.tolist())
