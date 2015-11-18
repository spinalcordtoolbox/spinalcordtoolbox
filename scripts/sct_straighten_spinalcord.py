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
from copy import deepcopy
from msct_parser import Parser
from sct_label_utils import ProcessLabels
from sct_crop_image import ImageCropper
from nibabel import load, Nifti1Image, save
from numpy import array, asarray, sum, isnan
from sympy.solvers import solve
from sympy import Symbol
from scipy import ndimage
from sct_apply_transfo import Transform
import sct_utils as sct
from msct_smooth import smoothing_window, evaluate_derivative_3D
from sct_image import set_orientation
from msct_types import Coordinate

import copy_reg
import types


def _pickle_method(method):
    """
    Author: Steven Bethard (author of argparse)
    http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    """
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    cls_name = ''
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
    if cls_name:
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    """
    Author: Steven Bethard
    http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    """
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


def is_number(s):
    """Check if input is float."""
    try:
        float(s)
        return True
    except TypeError:
        return False


def smooth_centerline(fname_centerline, algo_fitting='hanning', type_window='hanning', window_length=80, verbose=0):
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

    # get center of mass of the centerline/segmentation
    sct.printv('.. Get center of mass of the centerline/segmentation...', verbose)
    for iz in range(0, nz_nonz, 1):
        x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(array(data[:, :, z_centerline[iz]]))

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
        x_centerline_fit, y_centerline_fit, z_centerline_fit, x_centerline_deriv, y_centerline_deriv,\
            z_centerline_deriv = b_spline_nurbs(x_centerline, y_centerline, z_centerline, nbControl=None,
                                                verbose=verbose)

    else:
        sct.printv("ERROR: wrong algorithm for fitting", 1, "error")

    return x_centerline_fit, y_centerline_fit, z_centerline_fit, \
            x_centerline_deriv, y_centerline_deriv, z_centerline_deriv

def compute_cross(coordinate, gapxy=15):
    cross_coordinates = []

    x0, y0, z0 = coordinate.x, coordinate.y, coordinate.z
    initial_landmark_value = coordinate.value

    # set x, y and z coordinates for landmarks +x
    cross_coordinates.append(Coordinate([x0 + gapxy, y0, z0, initial_landmark_value + 1]))
    # set x, y and z coordinates for landmarks -x
    cross_coordinates.append(Coordinate([x0 - gapxy, y0, z0, initial_landmark_value + 2]))
    # set x, y and z coordinates for landmarks +y
    cross_coordinates.append(Coordinate([x0, y0 + gapxy, z0, initial_landmark_value + 3]))
    # set x, y and z coordinates for landmarks -y
    cross_coordinates.append(Coordinate([x0, y0 - gapxy, z0, initial_landmark_value + 4]))
    # set x, y and z coordinates for landmarks +x+y
    cross_coordinates.append(Coordinate([x0 + gapxy, y0 + gapxy, z0, initial_landmark_value + 5]))
    # set x, y and z coordinates for landmarks -x+y
    cross_coordinates.append(Coordinate([x0 - gapxy, y0 + gapxy, z0, initial_landmark_value + 6]))
    # set x, y and z coordinates for landmarks +x-y
    cross_coordinates.append(Coordinate([x0 + gapxy, y0 - gapxy, z0, initial_landmark_value + 7]))
    # set x, y and z coordinates for landmarks -x-y
    cross_coordinates.append(Coordinate([x0 - gapxy, y0 - gapxy, z0, initial_landmark_value + 8]))

    # internal crosses
    gapxy_internal = gapxy / 2
    # set x, y and z coordinates for landmarks +x
    cross_coordinates.append(Coordinate([x0 + gapxy_internal, y0, z0, initial_landmark_value + 9]))
    # set x, y and z coordinates for landmarks -x
    cross_coordinates.append(Coordinate([x0 - gapxy_internal, y0, z0, initial_landmark_value + 10]))
    # set x, y and z coordinates for landmarks +y
    cross_coordinates.append(Coordinate([x0, y0 + gapxy_internal, z0, initial_landmark_value + 11]))
    # set x, y and z coordinates for landmarks -y
    cross_coordinates.append(Coordinate([x0, y0 - gapxy_internal, z0, initial_landmark_value + 12]))
    # set x, y and z coordinates for landmarks +x+y
    cross_coordinates.append(Coordinate([x0 + gapxy_internal, y0 + gapxy_internal, z0, initial_landmark_value + 13]))
    # set x, y and z coordinates for landmarks -x+y
    cross_coordinates.append(Coordinate([x0 - gapxy_internal, y0 + gapxy_internal, z0, initial_landmark_value + 14]))
    # set x, y and z coordinates for landmarks +x-y
    cross_coordinates.append(Coordinate([x0 + gapxy_internal, y0 - gapxy_internal, z0, initial_landmark_value + 15]))
    # set x, y and z coordinates for landmarks -x-y
    cross_coordinates.append(Coordinate([x0 - gapxy_internal, y0 - gapxy_internal, z0, initial_landmark_value + 16]))

    return cross_coordinates

def compute_cross_centerline(coordinate, derivative, gapxy=15):
    # calculate d using formula: ax + by + cz + d = 0
    a = derivative.x
    b = derivative.y
    c = derivative.z
    x = coordinate.x
    y = coordinate.y
    z = coordinate.z
    d = -(a * x + b * y + c * z)

    # set y coordinate to y_centerline_fit[iz] for elements 1 and 2 of the cross
    cross_coordinates = [Coordinate(), Coordinate(), Coordinate(), Coordinate(),
                         Coordinate(), Coordinate(), Coordinate(), Coordinate(),
                         Coordinate(), Coordinate(), Coordinate(), Coordinate(),
                         Coordinate(), Coordinate(), Coordinate(), Coordinate()]

    cross_coordinates[0].y = coordinate.y
    cross_coordinates[1].y = coordinate.y

    # set x and z coordinates for landmarks +x and -x, forcing de landmark to be in the orthogonal plan
    # and the distance landmark/curve to be gapxy
    x_n = Symbol('x_n')
    cross_coordinates[1].x, cross_coordinates[0].x = solve((x_n - x) ** 2 +
                                                           ((-1 / c) * (a * x_n + b * y + d) - z) ** 2 -
                                                           gapxy ** 2, x_n)  # x for -x and +x
    cross_coordinates[0].z = (-1 / c) * (a * cross_coordinates[0].x + b * y + d)  # z for +x
    cross_coordinates[1].z = (-1 / c) * (a * cross_coordinates[1].x + b * y + d)  # z for -x

    # set x coordinate to x_centerline_fit[iz] for elements 3 and 4 of the cross
    cross_coordinates[2].x = coordinate.x
    cross_coordinates[3].x = coordinate.x

    # set coordinates for landmarks +y and -y. Here, x coordinate is 0 (already initialized).
    y_n = Symbol('y_n')
    cross_coordinates[3].y, cross_coordinates[2].y = solve((y_n - y) ** 2 +
                                                           ((-1 / c) * (a * x + b * y_n + d) - z) ** 2 -
                                                           gapxy ** 2, y_n)  # y for -y and +y
    cross_coordinates[2].z = (-1 / c) * (a * x + b * cross_coordinates[2].y + d)  # z for +y
    cross_coordinates[3].z = (-1 / c) * (a * x + b * cross_coordinates[3].y + d)  # z for -y

    # set the first corner
    cross_coordinates[4].y = coordinate.y + gapxy
    cross_coordinates[5].y = coordinate.y + gapxy

    # set x and z coordinates for landmarks +x and -x, forcing de landmark to be in the orthogonal plan
    # and the distance landmark/curve to be gapxy
    cross_coordinates[5].x, cross_coordinates[4].x = solve((x_n - x) ** 2 +
                                                           ((-1 / c) * (a * x_n + b * (y + gapxy) + d)
                                                            - z) ** 2 - gapxy ** 2, x_n)
    cross_coordinates[4].z = (-1 / c) * (a * cross_coordinates[4].x + b * (y + gapxy) + d)  # z for +x
    cross_coordinates[5].z = (-1 / c) * (a * cross_coordinates[5].x + b * (y + gapxy) + d)  # z for -x

    # set the other corner
    cross_coordinates[6].y = coordinate.y - gapxy
    cross_coordinates[7].y = coordinate.y - gapxy

    # set x and z coordinates for landmarks +x and -x, forcing de landmark to be in the orthogonal plan
    # and the distance landmark/curve to be gapxy
    cross_coordinates[7].x, cross_coordinates[6].x = solve((x_n - x) ** 2 +
                                                           ((-1 / c) * (a * x_n + b * (y - gapxy) + d)
                                                            - z) ** 2 - gapxy ** 2, x_n)
    cross_coordinates[6].z = (-1 / c) * (a * cross_coordinates[6].x + b * (y - gapxy) + d)  # z for +x
    cross_coordinates[7].z = (-1 / c) * (a * cross_coordinates[7].x + b * (y - gapxy) + d)  # z for -x

    gapxy /= 2
    cross_coordinates[8].y = coordinate.y
    cross_coordinates[9].y = coordinate.y

    # set x and z coordinates for landmarks +x and -x, forcing de landmark to be in the orthogonal plan
    # and the distance landmark/curve to be gapxy
    x_n = Symbol('x_n')
    cross_coordinates[9].x, cross_coordinates[8].x = solve((x_n - x) ** 2 +
                                                           ((-1 / c) * (a * x_n + b * y + d) - z) ** 2 -
                                                           gapxy ** 2, x_n)  # x for -x and +x
    cross_coordinates[8].z = (-1 / c) * (a * cross_coordinates[8].x + b * y + d)  # z for +x
    cross_coordinates[9].z = (-1 / c) * (a * cross_coordinates[9].x + b * y + d)  # z for -x

    # set x coordinate to x_centerline_fit[iz] for elements 3 and 4 of the cross
    cross_coordinates[10].x = coordinate.x
    cross_coordinates[11].x = coordinate.x

    # set coordinates for landmarks +y and -y. Here, x coordinate is 0 (already initialized).
    y_n = Symbol('y_n')
    cross_coordinates[11].y, cross_coordinates[10].y = solve((y_n - y) ** 2 +
                                                             ((-1 / c) * (a * x + b * y_n + d) - z) ** 2 -
                                                             gapxy ** 2, y_n)  # y for -y and +y
    cross_coordinates[10].z = (-1 / c) * (a * x + b * cross_coordinates[10].y + d)  # z for +y
    cross_coordinates[11].z = (-1 / c) * (a * x + b * cross_coordinates[11].y + d)  # z for -y

    # set the first corner
    cross_coordinates[12].y = coordinate.y + gapxy
    cross_coordinates[13].y = coordinate.y + gapxy

    # set x and z coordinates for landmarks +x and -x, forcing de landmark to be in the orthogonal plan
    # and the distance landmark/curve to be gapxy
    cross_coordinates[13].x, cross_coordinates[12].x = solve((x_n - x) ** 2 +
                                                             ((-1 / c) * (a * x_n + b * (y + gapxy) + d)
                                                              - z) ** 2 - gapxy ** 2, x_n)
    cross_coordinates[12].z = (-1 / c) * (a * cross_coordinates[12].x + b * (y + gapxy) + d)  # z for +x
    cross_coordinates[13].z = (-1 / c) * (a * cross_coordinates[13].x + b * (y + gapxy) + d)  # z for -x

    # set the other corner
    cross_coordinates[14].y = coordinate.y - gapxy
    cross_coordinates[15].y = coordinate.y - gapxy

    # set x and z coordinates for landmarks +x and -x, forcing de landmark to be in the orthogonal plan
    # and the distance landmark/curve to be gapxy
    cross_coordinates[15].x, cross_coordinates[14].x = solve((x_n - x) ** 2 +
                                                             ((-1 / c) * (a * x_n + b * (y - gapxy) + d)
                                                              - z) ** 2 - gapxy ** 2, x_n)
    cross_coordinates[14].z = (-1 / c) * (a * cross_coordinates[14].x + b * (y - gapxy) + d)  # z for +x
    cross_coordinates[15].z = (-1 / c) * (a * cross_coordinates[15].x + b * (y - gapxy) + d)  # z for -x

    for i, coord in enumerate(cross_coordinates):
        coord.value = coordinate.value + i + 1

    return cross_coordinates


class SpinalCordStraightener(object):

    def __init__(self, input_filename, centerline_filename, debug=0, deg_poly=10, gapxy=30, gapz=15, padding=30,
                 leftright_width=150, interpolation_warp='spline', rm_tmp_files=1, verbose=1, algo_fitting='hanning',
                 type_window='hanning', window_length=50, crop=1, output_filename=''):
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
        self.type_window = type_window  # !! for more choices, edit msct_smooth. Possibilities: 'flat', 'hanning',
        # 'hamming', 'bartlett', 'blackman'
        self.window_length = window_length
        self.crop = crop

        self.cpu_number = None
        self.results_landmarks_curved = []

        self.bspline_meshsize = '5x5x10'  # JULIEN
        self.bspline_numberOfLevels = '3'
        self.bspline_order = '3'
        self.all_labels = 1
        self.use_continuous_labels = 1

        self.mse_straightening = 0.0
        self.max_distance_straightening = 0.0

    def worker_landmarks_curved(self, arguments_worker):
        """Define landmarks along the centerline. Here, landmarks are densely defined along the centerline, and every
        gapxy, a cross of 16 landmarks is created (small square and big square = 8+8).
        """
        try:
            iz = arguments_worker[0]
            iz_curved, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv, x_centerline_fit, y_centerline_fit, \
                z_centerline = arguments_worker[1]

            temp_results = []

            if iz in iz_curved:
                # at a junction (gapxy): set coordinates for landmarks at the center of the cross
                coord = Coordinate([0, 0, 0, 0])
                coord.x, coord.y, coord.z = x_centerline_fit[iz], y_centerline_fit[iz], z_centerline[iz]
                deriv = Coordinate([0, 0, 0, 0])
                deriv.x, deriv.y, deriv.z = x_centerline_deriv[iz], y_centerline_deriv[iz], z_centerline_deriv[iz]
                temp_results.append(coord)

                cross_coordinates = compute_cross_centerline(coord, deriv, self.gapxy)

                for coord in cross_coordinates:
                    # check if all coordinates are real (jcohenadad, see issue #584)
                    if not (is_number(coord.x) and is_number(coord.y) and is_number(coord.z)):
                        sct.printv('WARNING: Coordinates have complex values at iz='+str(iz)+': '+str(coord), 1, 'warning')
                        # set wrong coordinates that will be removed later on
                        coord.x, coord.y, coord.z = 99999, 99999, 99999
                    temp_results.append(coord)
            else:
                # not a junction: do not create the cross.
                if self.all_labels >= 1:
                    temp_results.append(Coordinate([x_centerline_fit[iz], y_centerline_fit[iz],
                                                    z_centerline[iz], 0], mode='continuous'))

            return iz, temp_results

        except KeyboardInterrupt:
            return

        except Exception as e:
            raise e

    def worker_landmarks_curved_results(self, results):
        sorted(results, key=lambda l: l[0])
        self.results_landmarks_curved = []
        landmark_curved_value = 1
        for iz, l_curved in results:
            for landmark in l_curved:
                landmark.value = landmark_curved_value
                self.results_landmarks_curved.append(landmark)
                landmark_curved_value += 1

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

        if self.debug == 1:
            print "\n*** WARNING: DEBUG MODE ON ***\n"
            fname_anat = path_sct + "/testing/sct_testing_data/data/t2/t2.nii.gz"
            fname_centerline = path_sct + "/testing/sct_testing_data/data/t2/t2_seg.nii.gz"
            remove_temp_files = 0
            type_window = "hanning"
            verbose = 2

        # check existence of input files
        # sct.check_file_exist(fname_anat, verbose)
        # sct.check_file_exist(fname_centerline, verbose)

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
            # JULIEN
            sct.run('cp data.nii data_1mm.nii')
            sct.run('cp centerline.nii.gz centerline_1mm.nii.gz')
            # # resample data to 1mm isotropic
            # sct.printv('\nResample data to 1mm isotropic...', verbose)
            # # fname_anat_resampled = file_anat + "_resampled.nii.gz"
            # sct.run('sct_resample -i data.nii -mm 1.0x1.0x1.0 -x linear -o data_1mm.nii')
            # # fname_centerline_resampled = file_centerline + "_resampled.nii.gz"
            # sct.run('sct_resample -i centerline.nii.gz -mm 1.0x1.0x1.0 -x linear -o centerline_1mm.nii.gz')

            # Change orientation of the input centerline into RPI
            sct.printv("\nOrient centerline to RPI orientation...", verbose)
            sct.run('sct_image -i centerline_1mm.nii.gz -setorient RPI -o centerline_1mm_rpi.nii.gz')
            # fname_centerline_orient = file_centerline + "_rpi.nii.gz"
            # fname_centerline_orient = set_orientation(file_centerline+ext_centerline, "RPI", filename=True)

            # Get dimension
            sct.printv('\nGet dimensions...', verbose)
            from msct_image import Image
            image_centerline = Image('centerline_1mm_rpi.nii.gz')
            nx, ny, nz, nt, px, py, pz, pt = image_centerline.dim
            sct.printv('.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)
            sct.printv('.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', verbose)
            
            # smooth centerline
            x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline('centerline_1mm_rpi.nii.gz', algo_fitting=algo_fitting, type_window=type_window, window_length=window_length, verbose=verbose)

            # Get coordinates of landmarks along curved centerline
            # ==========================================================================================
            sct.printv("\nGet coordinates of landmarks along curved centerline...", verbose)
            # landmarks are created along the curved centerline every z=gapz. They consist of a "cross" of size gapx
            # and gapy. In voxel space!!!

            # compute the length of the spinal cord based on fitted centerline and size of centerline in z direction
            length_centerline, size_z_centerline = 0.0, 0.0
            from math import sqrt

            for iz in range(0, len(z_centerline) - 1):
                length_centerline += sqrt(((x_centerline_fit[iz] - x_centerline_fit[iz + 1]) * px) ** 2 +
                                          ((y_centerline_fit[iz] - y_centerline_fit[iz + 1]) * py) ** 2 +
                                          ((z_centerline[iz] - z_centerline[iz + 1]) * pz) ** 2)
                size_z_centerline += abs((z_centerline[iz] - z_centerline[iz + 1]) * pz)

            # compute the size factor between initial centerline and straight bended centerline
            factor_curved_straight = length_centerline / size_z_centerline
            middle_slice = (z_centerline[0] + z_centerline[-1]) / 2.0
            if verbose == 2:
                print "Length of spinal cord = ", str(length_centerline)
                print "Size of spinal cord in z direction = ", str(size_z_centerline)
                print "Ratio length/size = ", str(factor_curved_straight)

            # find z indices along centerline given a specific gap: iz_curved
            nz_nonz = len(z_centerline)
            nb_landmark = int(round(length_centerline/gapz))
            if nb_landmark == 0:
                nb_landmark = 1
            if nb_landmark == 1:
                iz_curved = [0]
                iz_straight = [0]
            else:
                iz_curved = [0]
                iz_straight = [(z_centerline[0] - middle_slice) * factor_curved_straight + middle_slice]
                temp_length_centerline = iz_straight[0]
                temp_previous_length = iz_straight[0]
                for iz in range(1, len(z_centerline) - 1):
                    temp_length_centerline += sqrt(((x_centerline_fit[iz] - x_centerline_fit[iz + 1]) * px) ** 2 +
                                                   ((y_centerline_fit[iz] - y_centerline_fit[iz + 1]) * py) ** 2 +
                                                   ((z_centerline[iz] - z_centerline[iz + 1]) * pz) ** 2)
                    if temp_length_centerline >= temp_previous_length + gapz:
                        iz_curved.append(iz)
                        iz_straight.append(temp_length_centerline)
                        temp_previous_length = temp_length_centerline
            iz_curved.append(nz_nonz - 1)
            iz_straight.append((z_centerline[-1] - middle_slice) * factor_curved_straight + middle_slice)

            # computing curved landmarks
            landmark_curved = []
            worker_arguments = (iz_curved, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv,
                                x_centerline_fit, y_centerline_fit, z_centerline)
            if self.cpu_number != 0:
                from multiprocessing import Pool
                arguments_landmarks = [(iz, worker_arguments) for iz in range(min(iz_curved), max(iz_curved) + 1, 1)]

                pool = Pool(processes=self.cpu_number)
                pool.map_async(self.worker_landmarks_curved, arguments_landmarks,
                               callback=self.worker_landmarks_curved_results)

                pool.close()
                try:
                    pool.join()  # waiting for all the jobs to be done
                    if self.results_landmarks_curved:
                        landmark_curved = self.results_landmarks_curved
                    else:
                        raise ValueError("ERROR: no curved landmarks constructed...")
                except KeyboardInterrupt:
                    print "\nWarning: Caught KeyboardInterrupt, terminating workers"
                    pool.terminate()
                    sys.exit(2)
                except Exception as e:
                    print "Error during straightening on line {}".format(sys.exc_info()[-1].tb_lineno)
                    print e
                    sys.exit(2)
            else:
                landmark_curved_temp = [self.worker_landmarks_curved((iz, worker_arguments))
                                        for iz in range(min(iz_curved), max(iz_curved) + 1, 1)]
                landmark_curved_value = 1
                for iz, l_curved in landmark_curved_temp:
                    for landmark in l_curved:
                        landmark.value = landmark_curved_value
                        landmark_curved.append(landmark)
                        landmark_curved_value += 1

            # Get coordinates of landmarks along straight centerline
            # ==========================================================================================
            sct.printv("\nGet coordinates of landmarks along straight centerline...", verbose)
            landmark_straight = []

            # calculate the z indices corresponding to the Euclidean distance between two consecutive points on the
            # curved centerline (approximation curve --> line)
            # TODO: DO NOT APPROXIMATE CURVE --> LINE

            # initialize x0 and y0 to be at the center of the FOV
            x0 = int(round(nx/2))
            y0 = int(round(ny/2))
            landmark_curved_value = 1
            for iz in range(min(iz_curved), max(iz_curved)+1, 1):
                # compute new z-coordinate based on iz, middle slice and factor_curved_straight
                if iz in iz_curved:
                    z0 = iz_straight[iz_curved.index(iz)]
                    # set coordinates for landmark at the center of the cross
                    central_coordinate = Coordinate([x0, y0, z0, landmark_curved_value])
                    landmark_straight.append(central_coordinate)
                    landmark_straight.extend(compute_cross(central_coordinate, gapxy=gapxy))
                    landmark_curved_value += 17
                elif self.all_labels >= 1:
                    z0 = (z_centerline[iz] - middle_slice) * factor_curved_straight + middle_slice
                    landmark_straight.append(Coordinate([x0, y0, z0, landmark_curved_value]))
                    landmark_curved_value += 1

            # Discard corrupted landmarks (jcohenadad, issue #584)
            landmark_curved_tmp = []
            for i_landmark_curved in landmark_curved:
                if not (i_landmark_curved.x == 99999 and i_landmark_curved.y == 99999 and i_landmark_curved.z == 99999):
                    landmark_curved_tmp.append(i_landmark_curved)
            landmark_curved = landmark_curved_tmp


            # Create NIFTI volumes with landmarks
            # ==========================================================================================
            # Pad input volume to deal with the fact that some landmarks on the curved centerline might be
            # outside the FOV
            # N.B. IT IS VERY IMPORTANT TO PAD ALSO ALONG X and Y, OTHERWISE SOME LANDMARKS MIGHT GET OUT OF THE FOV!!!
            sct.printv('\nPad input volume to account for landmarks that fall outside the FOV...', verbose)
            padding_x, padding_y, padding_z = padding, padding, padding
            if nx + padding <= leftright_width:
                padding_x = leftright_width - padding - nx
            sct.run('sct_image -i centerline_1mm_rpi.nii.gz -o tmp.centerline_pad.nii.gz -pad '+str(padding_x)+','+str(padding_y)+','+str(padding_z))

            # Open padded centerline for reading
            sct.printv('\nOpen padded centerline for reading...', verbose)
            file_image = load('tmp.centerline_pad.nii.gz')
            data = file_image.get_data()
            hdr = file_image.get_header()
            hdr_straight_landmarks = hdr.copy()
            hdr_straight_landmarks.structarr['quatern_b'] = -0.0
            hdr_straight_landmarks.structarr['quatern_c'] = 1.0
            hdr_straight_landmarks.structarr['quatern_d'] = 0.0
            hdr_straight_landmarks.structarr['srow_x'][1] = hdr_straight_landmarks.structarr['srow_x'][2] = 0.0
            hdr_straight_landmarks.structarr['srow_y'][0] = hdr_straight_landmarks.structarr['srow_y'][2] = 0.0
            hdr_straight_landmarks.structarr['srow_z'][0] = hdr_straight_landmarks.structarr['srow_z'][1] = 0.0
            hdr_straight_landmarks.structarr['srow_x'][0] = -1.0
            hdr_straight_landmarks.structarr['srow_y'][1] = hdr_straight_landmarks.structarr['srow_z'][2] = 1.0
            landmark_curved_rigid = []

            # Create volumes containing curved and straight landmarks
            data_curved_landmarks = data * 0
            data_straight_landmarks = data * 0

            # Loop across cross index
            for index in range(0, len(landmark_curved)):
                x, y, z = int(round(landmark_curved[index].x)), int(round(landmark_curved[index].y)), \
                          int(round(landmark_curved[index].z))

                # attribute landmark_value to the voxel
                # JULIEN
                # data_curved_landmarks[x + padding_x, y + padding_y, z + padding_z] = landmark_curved[index].value

                # JULIEN
                # attribute landmark_value to the voxel and its neighbours
                data_curved_landmarks[x + padding_x - 1:x + padding_x + 2, y + padding_y - 1:y + padding_y + 2,
                z + padding_z - 1:z + padding_z + 2] = landmark_curved[index].value

                # get x, y and z coordinates of straight landmark (rounded to closest integer)
                x, y, z = int(round(landmark_straight[index].x)), int(round(landmark_straight[index].y)), \
                          int(round(landmark_straight[index].z))

                # JULIEN
                # data_straight_landmarks[x + padding_x, y + padding_y, z + padding_z] = landmark_straight[index].value

                # JULIEN
                # attribute landmark_value to the voxel and its neighbours
                data_straight_landmarks[x + padding_x - 1:x + padding_x + 2, y + padding_y - 1:y + padding_y + 2,
                z + padding_z - 1:z + padding_z + 2] = landmark_straight[index].value

            # Write NIFTI volumes
            sct.printv('\nWrite NIFTI volumes...', verbose)
            hdr.set_data_dtype('uint32')  # set imagetype to uint8 #TODO: maybe use int32
            img = Nifti1Image(data_curved_landmarks, None, hdr)
            save(img, 'tmp.landmarks_curved.nii.gz')
            sct.printv('.. File created: tmp.landmarks_curved.nii.gz', verbose)
            # JULIEN
            # hdr_straight_landmarks.set_data_dtype('uint32')
            img = Nifti1Image(data_straight_landmarks, None, hdr_straight_landmarks)
            save(img, 'tmp.landmarks_straight.nii.gz')
            sct.printv('.. File created: tmp.landmarks_straight.nii.gz', verbose)

            # JULIEN
            crop_landmarks = 0
            safety_pad = 1
            if crop_landmarks == 1:
                # Crop landmarks (for faster computation)
                sct.printv("\nCrop around landmarks (for faster computation)...", verbose)
                sct.run('sct_crop_image -i tmp.landmarks_curved.nii.gz -bmax -o tmp.landmarks_curved_crop.nii.gz', verbose)
                sct.run('sct_crop_image -i tmp.landmarks_straight.nii.gz -bmax -o tmp.landmarks_straight_crop.nii.gz', verbose)

                # Pad landmarks by one voxel (to avoid issue #609)
                if safety_pad:
                    sct.printv("\nPad landmark volume to avoid having landmarks outside...", verbose)
                    sct.run('sct_image -i tmp.landmarks_curved_crop.nii.gz -pad '+str(safety_pad)+','+str(safety_pad)+','+str(safety_pad)+' -o tmp.landmarks_curved_crop.nii.gz', verbose)
                    sct.run('sct_image -i tmp.landmarks_straight_crop.nii.gz -pad '+str(safety_pad)+','+str(safety_pad)+','+str(safety_pad)+' -o tmp.landmarks_straight_crop.nii.gz', verbose)

                # Adjust LandmarksReal values after cropping
                # for curved
                x_adjust = min([int(round(i.x)) for i in landmark_curved]) + padding_x
                y_adjust = min([int(round(i.y)) for i in landmark_curved]) + padding_y
                z_adjust = min([int(round(i.z)) for i in landmark_curved]) + padding_z
                landmark_curved_adjust = deepcopy(landmark_curved)  # here we use deepcopy to copy list with object
                for index in range(0, len(landmark_curved)):
                    landmark_curved_adjust[index].x = landmark_curved[index].x - x_adjust + safety_pad
                    landmark_curved_adjust[index].y = landmark_curved[index].y - y_adjust + safety_pad
                    landmark_curved_adjust[index].z = landmark_curved[index].z - z_adjust + safety_pad
                # for straight
                x_adjust = min([int(round(i.x)) for i in landmark_straight]) + padding_x
                y_adjust = min([int(round(i.y)) for i in landmark_straight]) + padding_y
                z_adjust = min([int(round(i.z)) for i in landmark_straight]) + padding_z
                landmark_straight_adjust = deepcopy(landmark_straight)
                for index in range(0, len(landmark_straight)):
                    landmark_straight_adjust[index].x = landmark_straight[index].x - x_adjust + safety_pad
                    landmark_straight_adjust[index].y = landmark_straight[index].y - y_adjust + safety_pad
                    landmark_straight_adjust[index].z = landmark_straight[index].z - z_adjust + safety_pad
                # copy to keep same variable name throughout the code
                landmark_curved = deepcopy(landmark_curved_adjust)
                landmark_straight = deepcopy(landmark_straight_adjust)
            else:
                sct.run('cp tmp.landmarks_curved.nii.gz tmp.landmarks_curved_crop.nii.gz', verbose)
                sct.run('cp tmp.landmarks_straight.nii.gz tmp.landmarks_straight_crop.nii.gz', verbose)

            # display
            if verbose == 2:
                from mpl_toolkits.mplot3d import Axes3D
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = Axes3D(fig)
                plt_landmarks_curved, = ax.plot([coord.x for coord in landmark_curved],
                                                [coord.y for coord in landmark_curved],
                                                [coord.z for coord in landmark_curved],
                                                'b.', markersize=3)
                plt_landmarks_straight, = ax.plot([coord.x for coord in landmark_straight],
                                                  [coord.y for coord in landmark_straight],
                                                  [coord.z for coord in landmark_straight],
                                                  'r.', markersize=3)
                plt.legend([plt_landmarks_curved, plt_landmarks_straight],
                               ['Landmarks curved', 'Landmarks straight'])
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.set_aspect('equal')
                plt.show()

            # Remove non-matching landmarks
            landmark_curved, landmark_straight = ProcessLabels.remove_label_coord(landmark_curved, landmark_straight, symmetry=True)

            # Writing landmark curve in text file
            landmark_curved_file = open("LandmarksRealCurve.txt", "w+")
            for i in landmark_curved:
                landmark_curved_file.write(
                    str(i.x + padding_x) + "," +
                    str(i.y + padding_y) + "," +
                    str(i.z + padding_z) + "\n")
            landmark_curved_file.close()

            # Writing landmark curve in text file
            landmark_straight_file = open("LandmarksRealStraight.txt", "w+")
            for i in landmark_straight:
                landmark_straight_file.write(
                    str(i.x + padding_x) + "," +
                    str(i.y + padding_y) + "," +
                    str(i.z + padding_z) + "\n")
            landmark_straight_file.close()

            # Estimate b-spline transformation curve --> straight
            sct.printv("\nEstimate b-spline transformation: curve --> straight...", verbose)
            status, output = sct.run('isct_ANTSLandmarksBSplineTransform '
                                     'tmp.landmarks_straight.nii.gz '
                                     'tmp.landmarks_curved.nii.gz '
                                     'tmp.curve2straight_rigid.txt '
                                     'tmp.warp_curve2straight.nii.gz ' +
                                     self.bspline_meshsize + ' ' +
                                     self.bspline_numberOfLevels + ' '
                                     'LandmarksRealCurve.txt '
                                     'LandmarksRealStraight.txt ' +
                                     self.bspline_order +
                                     ' 0',
                                     verbose=verbose)

            # JULIEN
            # remove padding for straight labels
            crop = 1
            if crop == 1:
                ImageCropper(input_file="tmp.landmarks_straight.nii.gz",
                             output_file="tmp.landmarks_straight_crop.nii.gz", dim=[0, 1, 2], bmax=True,
                             verbose=verbose).crop()
                pass
            else:
                sct.run("cp tmp.landmarks_straight.nii.gz tmp.landmarks_straight_crop.nii.gz", verbose)

            # Concatenate rigid and non-linear transformations...
            sct.printv("\nConcatenate rigid and non-linear transformations...", verbose)
            sct.run('sct_concat_transfo -w tmp.curve2straight_rigid.txt,tmp.warp_curve2straight.nii.gz -d tmp.landmarks_straight_crop.nii.gz -o tmp.curve2straight.nii.gz')

            # Estimate b-spline transformation straight --> curve
            sct.printv("\nEstimate b-spline transformation: straight --> curve...", verbose)
            status, output = sct.run('isct_ANTSLandmarksBSplineTransform '
                                     'tmp.landmarks_curved.nii.gz '
                                     'tmp.landmarks_straight.nii.gz '
                                     'tmp.straight2curve_rigid.txt '
                                     'tmp.warp_straight2curve.nii.gz ' +
                                     self.bspline_meshsize + ' ' +
                                     self.bspline_numberOfLevels + ' ' +
                                     'LandmarksRealStraight.txt '
                                     'LandmarksRealCurve.txt ' +
                                     self.bspline_order +
                                     ' 0',
                                     verbose=verbose)

            # Concatenate rigid and non-linear transformations...
            sct.printv("\nConcatenate rigid and non-linear transformations...", verbose)
            sct.run('sct_concat_transfo -w tmp.straight2curve_rigid.txt,tmp.warp_straight2curve.nii.gz -d data.nii -o tmp.straight2curve.nii.gz')

            # Apply transformation to input image
            sct.printv('\nApply transformation to input image...', verbose)
            sct.run('sct_apply_transfo -i data.nii -d tmp.landmarks_straight_crop.nii.gz -o tmp.anat_rigid_warp.nii.gz -w tmp.curve2straight.nii.gz -x '+interpolation_warp, verbose)
            # Transform(input_filename='data.nii', fname_dest="tmp.landmarks_straight_crop.nii.gz",
            #           output_filename="tmp.anat_rigid_warp.nii.gz", interp=interpolation_warp,
            #           warp="tmp.curve2straight.nii.gz", verbose=verbose).apply()

            # compute the error between the straightened centerline/segmentation and the central vertical line.
            # Ideally, the error should be zero.
            # Apply deformation to input image
            sct.printv('\nApply transformation to centerline image...', verbose)
            Transform(input_filename='centerline.nii.gz', fname_dest="tmp.landmarks_straight_crop.nii.gz",
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
            print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
            print e

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
    parser.add_option(name="-c",
                      type_value="image_nifti",
                      description="centerline or segmentation.",
                      mandatory=True,
                      example="centerline.nii.gz")
    parser.add_option(name="-p",
                      type_value="int",
                      description="amount of padding for generating labels.",
                      mandatory=False,
                      example="30",
                      default_value=30)
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
    parser.add_option(name="-a",
                      type_value="multiple_choice",
                      description="Algorithm for curve fitting.",
                      mandatory=False,
                      example=["hanning", "nurbs"],
                      default_value="hanning")
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

    parser.add_option(name="-params",
                      type_value=[[','], 'str'],
                      description="Parameters for spinal cord straightening. Separate arguments with ','."
                                  "\nall_labels : 0,1. Default = 1"
                                  "\nalgo_fitting: {hanning,nurbs} algorithm for curve fitting. Default=hanning"
                                  "\nbspline_meshsize: <int>x<int>x<int> size of mesh for B-Spline registration. "
                                  "Default=3x3x5"
                                  "\nbspline_numberOfLevels: <int> number of levels for BSpline interpolation. "
                                  "Default=3"
                                  "\nbspline_order: <int> Order of BSpline for interpolation. Default=2"
                                  "\nalgo_landmark_rigid {rigid,xy,translation,translation-xy,rotation,rotation-xy} "
                                  "constraints on landmark-based rigid pre-registration"
                                  "\nleftright_width: <int> Width of padded image in left-right direction. Can be set "
                                  "improve warping field of straightening. Default=150.",
                      mandatory=False,
                      example="algo_fitting=nurbs,bspline_meshsize=5x5x12,algo_landmark_rigid=xy")

    parser.add_option(name='-qc',
                      type_value='multiple_choice',
                      description='Output images for quality control.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')

    parser.add_option(name="-cpu-nb",
                      type_value="int",
                      description="Number of CPU used for straightening. 0: no multiprocessing. If not provided, "
                                  "it uses all the available cores.",
                      mandatory=False,
                      example="8")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    # assigning variables to arguments
    input_filename = arguments["-i"]
    centerline_file = arguments["-c"]

    sc_straight = SpinalCordStraightener(input_filename, centerline_file)

    # Handling optional arguments
    if "-r" in arguments:
        sc_straight.remove_temp_files = int(arguments["-r"])
    if "-p" in arguments:
        sc_straight.padding = int(arguments["-p"])
    if "-x" in arguments:
        sc_straight.interpolation_warp = str(arguments["-x"])
    if "-o" in arguments:
        sc_straight.output_filename = str(arguments["-o"])
    if '-ofolder' in arguments:
        sc_straight.path_output = arguments['-ofolder']
    else:
        sc_straight.path_output = ''
    if "-a" in arguments:
        sc_straight.algo_fitting = str(arguments["-a"])
    if "-f" in arguments:
        sc_straight.crop = int(arguments["-f"])
    if "-v" in arguments:
        sc_straight.verbose = int(arguments["-v"])
    if "-cpu-nb" in arguments:
        sc_straight.cpu_number = int(arguments["-cpu-nb"])
    if '-qc' in arguments:
        sc_straight.qc = int(arguments['-qc'])

    if "-params" in arguments:
        params_user = arguments['-params']
        # update registration parameters
        for param in params_user:
            param_split = param.split('=')
            if param_split[0] == 'algo_fitting':
                sc_straight.algo_fitting = param_split[1]
            elif param_split[0] == 'bspline_meshsize':
                sc_straight.bspline_meshsize = param_split[1]
            elif param_split[0] == 'bspline_numberOfLevels':
                sc_straight.bspline_numberOfLevels = param_split[1]
            elif param_split[0] == 'bspline_order':
                sc_straight.bspline_order = param_split[1]
            elif param_split[0] == 'algo_landmark_rigid':
                sct.printv('ERROR: This feature (use_continuous_labels) is deprecated.', 1, 'error')
            elif param_split[0] == 'all_labels':
                sc_straight.all_labels = int(param_split[1])
            elif param_split[0] == 'use_continuous_labels':
                sct.printv('ERROR: This feature (use_continuous_labels) is deprecated.', 1, 'error')
            elif param_split[0] == 'gapz':
                sc_straight.gapz = int(param_split[1])
            elif param_split[0] == 'leftright_width':
                sc_straight.leftright_width = int(param_split[1])

    sc_straight.straighten()
