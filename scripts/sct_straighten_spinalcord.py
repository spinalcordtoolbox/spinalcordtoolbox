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
import os
import time
import commands
import sys
from msct_parser import Parser
from nibabel import Nifti1Image, save
from numpy import array, asarray, sum, isnan, round, mgrid, zeros, mean, std, delete
from scipy import ndimage
from sct_apply_transfo import Transform
import sct_utils as sct
from msct_smooth import smoothing_window, evaluate_derivative_3D
from math import sqrt


def smooth_centerline(fname_centerline, algo_fitting='hanning', type_window='hanning', window_length=80, verbose=0, nurbs_pts_number=1000, all_slices=True, phys_coordinates=False):
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

    # get center of mass of the centerline/segmentation and remove outliers
    sct.printv('.. Get center of mass of the centerline/segmentation...', verbose)
    for iz in range(0, nz_nonz, 1):
        slice = array(data[:, :, z_centerline[iz]])
        labeled_array, num_f = ndimage.measurements.label(slice)
        num_features[iz] = num_f
        x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(slice)
        if iz != 0:
            distances.append(sqrt((x_centerline[iz]-x_centerline[iz-1]) ** 2 + (y_centerline[iz]-y_centerline[iz-1]) ** 2))

    mean_distances = mean(distances)
    std_distances = std(distances)
    indices_to_remove = []

    # ascending verification
    for iz in range(0, nz_nonz/2, 1):
        distance = sqrt((x_centerline[iz]-x_centerline[iz+1]) ** 2 + (y_centerline[iz]-y_centerline[iz+1]) ** 2)
        if num_features[iz] > 1 or abs(distance - mean_distances) > 3 * std_distances:
            indices_to_remove.append(iz)

    # descending verification
    for iz in range(nz_nonz-1, nz_nonz/2, -1):
        distance = sqrt((x_centerline[iz]-x_centerline[iz-1]) ** 2 + (y_centerline[iz]-y_centerline[iz-1]) ** 2)
        if num_features[iz] > 1 or abs(distance - mean_distances) > 3 * std_distances:
            indices_to_remove.append(iz)

    x_centerline = delete(x_centerline, indices_to_remove)
    y_centerline = delete(y_centerline, indices_to_remove)
    z_centerline = delete(z_centerline, indices_to_remove)

    if phys_coordinates:
        sct.printv('.. Computing physical coordinates of centerline/segmentation...', verbose)
        coord_centerline = array(zip(x_centerline, y_centerline, z_centerline))
        phys_coord_centerline = file_image.transfo_pix2phys(coord_centerline)
        x_centerline = phys_coord_centerline[:, 0]
        y_centerline = phys_coord_centerline[:, 1]
        z_centerline = phys_coord_centerline[:, 2]

    sct.printv('.. Smoothing algo = '+algo_fitting, verbose)
    if algo_fitting == 'hanning':
        # 2D smoothing
        sct.printv('.. Windows length = '+str(window_length), verbose)

        # change to array
        x_centerline = asarray(x_centerline)
        y_centerline = asarray(y_centerline)

        # Smooth the curve
        x_centerline_smooth = smoothing_window(x_centerline, window_len=window_length/pz, window=type_window,
                                               verbose=verbose, robust=0, remove_edge_points=remove_edge_points)
        y_centerline_smooth = smoothing_window(y_centerline, window_len=window_length/pz, window=type_window,
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

        x_centerline_fit = asarray(x_centerline_fit)
        y_centerline_fit = asarray(y_centerline_fit)
        z_centerline_fit = asarray(z_centerline_fit)

    elif algo_fitting == "nurbs":
        from msct_smooth import b_spline_nurbs

        # TODO: remove outliers that are at the edges of the spinal cord
        # simple way to do it: go from one end and remove point if the distance from mean is higher than 2 * std

        x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv,\
            z_centerline_deriv = b_spline_nurbs(x_centerline, y_centerline, z_centerline, nbControl=None,
                                                point_number=nurbs_pts_number, verbose=verbose, all_slices=all_slices)

    else:
        sct.printv("ERROR: wrong algorithm for fitting", 1, "error")

    return x_centerline_fit, y_centerline_fit, z_centerline_fit, \
            x_centerline_deriv, y_centerline_deriv, z_centerline_deriv


class SpinalCordStraightener(object):

    def __init__(self, input_filename, centerline_filename, debug=0, deg_poly=10, gapxy=30, gapz=15, padding=30,
                 leftright_width=150, interpolation_warp='spline', rm_tmp_files=1, verbose=1, algo_fitting='nurbs',
                 precision=2.0, type_window='hanning', window_length=50, crop=1, output_filename=''):
        self.input_filename = input_filename
        self.centerline_filename = centerline_filename
        self.output_filename = output_filename
        self.debug = debug
        self.deg_poly = deg_poly  # maximum degree of polynomial function for fitting centerline.
        self.gapxy = gapxy  # size of cross in x and y direction for the landmarks
        self.gapz = gapz  # gap between landmarks along z voxels
        self.padding = padding  # pad input volume in order to deal with the fact that some landmarks might be outside
        # the FOV due to the curvature of the spinal cord
        self.leftright_width = leftright_width
        self.interpolation_warp = interpolation_warp
        self.remove_temp_files = rm_tmp_files  # remove temporary files
        self.verbose = verbose
        self.algo_fitting = algo_fitting  # 'hanning' or 'nurbs'
        self.precision = precision
        self.type_window = type_window  # !! for more choices, edit msct_smooth. Possibilities: 'flat', 'hanning',
        # 'hamming', 'bartlett', 'blackman'
        self.window_length = window_length
        self.crop = crop
        self.path_output = ""

        self.mse_straightening = 0.0
        self.max_distance_straightening = 0.0

    def straighten(self):
        # Initialization
        fname_anat = self.input_filename
        fname_centerline = self.centerline_filename
        fname_output = self.output_filename
        gapxy = self.gapxy
        gapz = self.gapz
        padding = self.padding
        leftright_width = self.leftright_width
        remove_temp_files = self.remove_temp_files
        verbose = self.verbose
        interpolation_warp = self.interpolation_warp
        algo_fitting = self.algo_fitting
        window_length = self.window_length
        type_window = self.type_window
        crop = self.crop
        qc = self.qc

        # start timer
        start_time = time.time()

        # get path of the toolbox
        status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
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

        # create temporary folder
        path_tmp = sct.tmp_create(verbose=verbose)

        # Copying input data to tmp folder
        sct.printv('\nCopy files to tmp folder...', verbose)
        sct.run('sct_convert -i '+fname_anat+' -o '+path_tmp+'data.nii')
        sct.run('sct_convert -i '+fname_centerline+' -o '+path_tmp+'centerline.nii.gz')

        # go to tmp folder
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
            sct.printv('.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)
            sct.printv('.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', verbose)

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
                number_of_points = int(self.precision * nz)

            # 2. extract bspline fitting of the centreline, and its derivatives
            x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline('centerline_rpi.nii.gz', algo_fitting=algo_fitting, type_window=type_window, window_length=window_length, verbose=verbose, nurbs_pts_number=number_of_points, all_slices=False, phys_coordinates=True)
            from msct_types import Centerline
            centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

            # Get coordinates of landmarks along curved centerline
            # ==========================================================================================
            sct.printv("\nGet coordinates of landmarks along curved centerline...", verbose)
            # 3. compute length of centerline
            # compute the length of the spinal cord based on fitted centerline and size of centerline in z direction
            from math import sqrt

            length_centerline = centerline.length
            size_z_centerline = nz * pz

            # compute the size factor between initial centerline and straight bended centerline
            factor_curved_straight = length_centerline / size_z_centerline
            middle_slice = (z_centerline[0] + z_centerline[-1]) / 2.0
            if verbose == 2:
                print "Length of spinal cord = ", str(length_centerline)
                print "Size of spinal cord in z direction = ", str(nz * pz)
                print "Ratio length/size = ", str(factor_curved_straight)

            # 4. compute and generate straight space
            # points along curved centerline are already regularly spaced.
            # calculate position of points along straight centerline

            # Create straight NIFTI volumes
            # ==========================================================================================
            sct.printv('\nPad input volume to account for landmarks that fall outside the FOV...', verbose)
            from numpy import ceil
            start_point = (z_centerline[0] - middle_slice) * factor_curved_straight + middle_slice
            padding_z = int(ceil(abs(start_point/float(nz))))
            sct.run('sct_image -i centerline_rpi.nii.gz -o tmp.centerline_pad.nii.gz -pad 0,0,'+str(padding_z))
            image_centerline_pad = Image('tmp.centerline_pad.nii.gz')
            nx, ny, nz, nt, px, py, pz, pt = image_centerline_pad.dim
            hdr_warp = image_centerline_pad.hdr.copy()

            number_of_voxel = nx * ny * nz
            sct.printv("Number of voxel = " + str(number_of_voxel))

            time_centerlines = time.time()
            z_centerline = [item + padding_z for item in z_centerline]
            centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline,
                                    x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

            from numpy import linspace
            ix_straight = [int(round(nx / 2))] * number_of_points
            iy_straight = [int(round(ny / 2))] * number_of_points
            iz_straight = linspace(0, nz, number_of_points)
            dx_straight = [0.0] * number_of_points
            dy_straight = [0.0] * number_of_points
            dz_straight = [1.0] * number_of_points
            coord_straight = array(zip(ix_straight, iy_straight, iz_straight))
            coord_phys_straight = image_centerline_pad.transfo_pix2phys(coord_straight)

            centerline_straight = Centerline(coord_phys_straight[:, 0], coord_phys_straight[:, 1], coord_phys_straight[:, 2],
                                             dx_straight, dy_straight, dz_straight)

            time_centerlines = time.time() - time_centerlines
            print 'Time to generate centerline: ' + str(round(time_centerlines * 1000.0)) + ' ms'

            # Create volumes containing curved and straight warping fields
            time_generation_volumes = time.time()
            data_warp_curved2straight = zeros((nx, ny, nz, 1, 3))
            data_warp_straight2curved = zeros((nx, ny, nz, 1, 3))

            # 5. compute transformations
            # Curved and straight images and the same dimensions, so we compute both warping fields at the same time.
            # b. determine which plane of spinal cord centreline it is includedv
            x, y, z = mgrid[0:nx, 0:ny, 0:nz]
            indexes = array(zip(x.ravel(), y.ravel(), z.ravel()))
            time_generation_volumes = time.time() - time_generation_volumes
            print 'Time to generate volumes and indices: ' + str(round(time_generation_volumes * 1000.0)) + ' ms'

            time_find_nearest_indexes = time.time()
            physical_coordinates = image_centerline_pad.transfo_pix2phys(indexes)
            nearest_indexes_curved = centerline.find_nearest_indexes(physical_coordinates)
            nearest_indexes_straight = centerline_straight.find_nearest_indexes(physical_coordinates)
            time_find_nearest_indexes = time.time() - time_find_nearest_indexes
            print 'Time to find nearest centerline points: ' + str(round(time_find_nearest_indexes * 1000.0)) + ' ms'

            # compute the distance from voxels to corresponding plans.
            # This distance is used to blackout voxels that are not in the modified image.
            time_get_distances_from_planes = time.time()
            distances_curved = centerline.get_distances_from_planes(physical_coordinates, nearest_indexes_curved)
            distances_straight = centerline_straight.get_distances_from_planes(physical_coordinates, nearest_indexes_straight)
            threshold_distance = 1.0
            indexes_out_distance_curved = distances_curved > threshold_distance
            indexes_out_distance_straight = distances_straight > threshold_distance
            time_get_distances_from_planes = time.time() - time_get_distances_from_planes
            print 'Time to compute distance between voxels and nearest plans: ' + str(round(time_get_distances_from_planes * 1000.0)) + ' ms'

            # c. compute the position of the voxel in the plane coordinate system
            # (X and Y distance from centreline, along the plane)
            time_get_projected_coordinates_on_planes = time.time()
            projected_points_curved = centerline.get_projected_coordinates_on_planes(physical_coordinates, nearest_indexes_curved)
            projected_points_straight = centerline_straight.get_projected_coordinates_on_planes(physical_coordinates, nearest_indexes_straight)
            time_get_projected_coordinates_on_planes = time.time() - time_get_projected_coordinates_on_planes
            print 'Time to get projected voxels on plans: ' + str(round(time_get_projected_coordinates_on_planes * 1000.0)) + ' ms'

            # e. find the correspondance of the voxel in the corresponding plane
            time_get_in_plans_coordinates = time.time()
            coord_in_planes_curved = centerline.get_in_plans_coordinates(projected_points_curved, nearest_indexes_curved)
            coord_in_planes_straight = centerline_straight.get_in_plans_coordinates(projected_points_straight, nearest_indexes_straight)
            time_get_in_plans_coordinates = time.time() - time_get_in_plans_coordinates
            print 'Time to get in-plane coordinates: ' + str(round(time_get_in_plans_coordinates * 1000.0)) + ' ms'

            # 6. generate warping fields for each transformations
            # compute coordinate in straight space based on position on plane
            time_displacements = time.time()
            coord_curved2straight = centerline_straight.points[nearest_indexes_curved]
            coord_curved2straight[:, 0:2] += coord_in_planes_curved[:, 0:2]
            coord_curved2straight[:, 2] += distances_curved

            displacements_curved = coord_curved2straight - physical_coordinates
            # for some reason, displacement in Z is inverted. Probably due to left/right-hended definition of referential.
            displacements_curved[:, 2] = -displacements_curved[:, 2]
            displacements_curved[indexes_out_distance_curved] = [100000.0, 100000.0, 100000.0]

            coord_straight2curved = centerline.get_inverse_plans_coordinates(coord_in_planes_straight, nearest_indexes_straight)
            displacements_straight = coord_straight2curved - physical_coordinates
            # for some reason, displacement in Z is inverted. Probably due to left/right-hended definition of referential.
            displacements_straight[:, 2] = -displacements_straight[:, 2]
            displacements_straight[indexes_out_distance_straight] = [100000.0, 100000.0, 100000.0]

            # For error-free interpolation purpose, warping fields are inverted in the definition of ITK.
            data_warp_curved2straight[indexes[:, 0], indexes[:, 1], indexes[:, 2], 0, :] = -displacements_straight
            data_warp_straight2curved[indexes[:, 0], indexes[:, 1], indexes[:, 2], 0, :] = -displacements_curved

            time_displacements = time.time() - time_displacements
            print 'Time to compute physical displacements: ' + str(round(time_displacements * 1000.0)) + ' ms'

            # Generate warp files as a warping fields
            hdr_warp.set_intent('vector', (), '')
            hdr_warp.set_data_dtype('float32')
            img = Nifti1Image(data_warp_curved2straight, None, hdr_warp)
            save(img, 'tmp.curve2straight.nii.gz')
            sct.printv('\nDONE ! Warping field generated: tmp.curve2straight.nii.gz', verbose)

            img = Nifti1Image(data_warp_straight2curved, None, hdr_warp)
            save(img, 'tmp.straight2curve.nii.gz')
            sct.printv('\nDONE ! Warping field generated: tmp.straight2curve.nii.gz', verbose)

            # Apply transformation to input image
            sct.printv('\nApply transformation to input image...', verbose)
            sct.run('sct_apply_transfo -i data.nii -d tmp.centerline_pad.nii.gz -o tmp.anat_rigid_warp.nii.gz -w tmp.curve2straight.nii.gz -x '+interpolation_warp, verbose)

            # compute the error between the straightened centerline/segmentation and the central vertical line.
            # Ideally, the error should be zero.
            # Apply deformation to input image
            sct.printv('\nApply transformation to centerline image...', verbose)
            Transform(input_filename='centerline.nii.gz', fname_dest="tmp.centerline_pad.nii.gz",
                      output_filename="tmp.centerline_straight.nii.gz", interp="nn",
                      warp="tmp.curve2straight.nii.gz", verbose=verbose).apply()
            from msct_image import Image
            file_centerline_straight = Image('tmp.centerline_straight.nii.gz', verbose=verbose)
            coordinates_centerline = file_centerline_straight.getNonZeroCoordinates(sorting='z')
            mean_coord = []
            from numpy import mean
            for z in range(coordinates_centerline[0].z, coordinates_centerline[-1].z):
                temp_mean = [coord.value for coord in coordinates_centerline if coord.z == z]
                if temp_mean:
                    mean_value = mean(temp_mean)
                    mean_coord.append(mean([[coord.x * coord.value / mean_value, coord.y * coord.value / mean_value]
                                            for coord in coordinates_centerline if coord.z == z], axis=0))

            # compute error between the straightened centerline and the straight line.
            from math import sqrt
            x0 = file_centerline_straight.data.shape[0]/2.0
            y0 = file_centerline_straight.data.shape[1]/2.0
            count_mean = 0
            for coord_z in mean_coord[2:-2]:  # we don't include the four extrema because there are usually messy.
                if not isnan(sum(coord_z)):
                    dist = ((x0-coord_z[0])*px)**2 + ((y0-coord_z[1])*py)**2
                    self.mse_straightening += dist
                    dist = sqrt(dist)
                    if dist > self.max_distance_straightening:
                        self.max_distance_straightening = dist
                    count_mean += 1
            self.mse_straightening = sqrt(self.mse_straightening/float(count_mean))

        except Exception as e:
            sct.printv('WARNING: Exception during Straightening:', 1, 'warning')
            sct.printv('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), 1, 'warning')
            sct.printv(str(e), 1, 'warning')

        os.chdir('..')

        # Generate output file (in current folder)
        # TODO: do not uncompress the warping field, it is too time consuming!
        sct.printv("\nGenerate output file (in current folder)...", verbose)
        sct.generate_output_file(path_tmp + "/tmp.curve2straight.nii.gz", self.path_output + "warp_curve2straight.nii.gz", verbose)
        sct.generate_output_file(path_tmp + "/tmp.straight2curve.nii.gz", self.path_output + "warp_straight2curve.nii.gz", verbose)
        if fname_output == '':
            fname_straight = sct.generate_output_file(path_tmp + "/tmp.anat_rigid_warp.nii.gz",
                                                      self.path_output + file_anat + "_straight" + ext_anat, verbose)
        else:
            fname_straight = sct.generate_output_file(path_tmp+'/tmp.anat_rigid_warp.nii.gz',
                                                      self.path_output + fname_output, verbose)  # straightened anatomic

        # Remove temporary files
        if remove_temp_files:
            sct.printv("\nRemove temporary files...", verbose)
            sct.run("rm -rf " + path_tmp, verbose)

        sct.printv('\nDone!\n', verbose)

        sct.printv("Maximum x-y error = " + str(round(self.max_distance_straightening, 2)) + " mm", verbose, "bold")
        sct.printv("Accuracy of straightening (MSE) = " + str(round(self.mse_straightening, 2)) +
                   " mm", verbose, "bold")

        # display elapsed time
        elapsed_time = time.time() - start_time
        sct.printv("\nFinished! Elapsed time: " + str(int(round(elapsed_time))) + "s", verbose)
        sct.printv("\nTo view results, type:", verbose)
        sct.printv("fslview " + fname_straight + " &\n", verbose, 'info')

        # output QC image
        if qc:
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
    parser.add_option(name="-pad",
                      type_value="int",
                      description="amount of padding for generating labels.",
                      mandatory=False,
                      example="30",
                      default_value=30)
    parser.add_option(name="-p",
                      type_value=None,
                      description="amount of padding for generating labels.",
                      mandatory=False,
                      deprecated_by='-pad')
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
    parser.add_option(name="-f",
                      type_value="multiple_choice",
                      description="Crop option. 0: no crop, 1: crop around landmarks.",
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
                                  "\nalgo_fitting: {hanning,nurbs} algorithm for curve fitting. Default=hanning"
                                  "\nprecision: [1.0,inf] Precision factor of straightening, related to the number of slices. Increasing this parameter increases the precision along with a loss of time. Is not taken into account with hanning fitting method. Default=2.0",
                      mandatory=False,
                      example="algo_fitting=nurbs")
    parser.add_option(name="-params",
                      type_value=None,
                      description="Parameters for spinal cord straightening. Separate arguments with ','."
                                  "\nalgo_fitting: {hanning,nurbs} algorithm for curve fitting. Default=nurbs"
                                  "\nprecision: [1.0,inf] Precision factor of straightening, related to the number of slices. Increasing this parameter increases the precision along with a loss of time. Is not taken into account with hanning fitting method. Default=2.0",
                      mandatory=False,
                      deprecated_by='-param')

    parser.add_option(name='-qc',
                      type_value='multiple_choice',
                      description='Output images for quality control.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value='0')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    # assigning variables to arguments
    input_filename = arguments["-i"]
    centerline_file = arguments["-s"]

    sc_straight = SpinalCordStraightener(input_filename, centerline_file)

    # Handling optional arguments
    if "-r" in arguments:
        sc_straight.remove_temp_files = int(arguments["-r"])
    if "-pad" in arguments:
        sc_straight.padding = int(arguments["-pad"])
    if "-x" in arguments:
        sc_straight.interpolation_warp = str(arguments["-x"])
    if "-o" in arguments:
        sc_straight.output_filename = str(arguments["-o"])
    if '-ofolder' in arguments:
        sc_straight.path_output = arguments['-ofolder']
    else:
        sc_straight.path_output = ''
    if "-f" in arguments:
        sc_straight.crop = int(arguments["-f"])
    if "-v" in arguments:
        sc_straight.verbose = int(arguments["-v"])
    # if "-cpu-nb" in arguments:
    #     sc_straight.cpu_number = int(arguments["-cpu-nb"])
    if '-qc' in arguments:
        sc_straight.qc = int(arguments['-qc'])

    if "-param" in arguments:
        params_user = arguments['-param']
        # update registration parameters
        for param in params_user:
            param_split = param.split('=')
            if param_split[0] == 'algo_fitting':
                sc_straight.algo_fitting = param_split[1]
            if param_split[0] == 'precision':
                sc_straight.precision = float(param_split[1])

    sc_straight.straighten()
