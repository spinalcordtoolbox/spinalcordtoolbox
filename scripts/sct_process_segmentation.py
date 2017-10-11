#!/usr/bin/env python
#########################################################################################
#
# Perform various types of processing from the spinal cord segmentation (e.g. extract centerline, compute CSA, etc.).
# (extract_centerline) extract the spinal cord centerline from the segmentation. Output file is an image in the same
# space as the segmentation.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Touati, Gabriel Mangeat
# Modified: 2014-07-20 by jcohenadad
#
# About the license: see the file LICENSE.TXT
#########################################################################################
# TODO: the import of scipy.misc imsave was moved to the specific cases (orth and ellipse) in order to avoid issue #62. This has to be cleaned in the future.

import sys
import os
import shutil
from random import randint
import time
import numpy as np
import scipy
import sct_utils as sct
from msct_nurbs import NURBS
from sct_image import set_orientation
from sct_straighten_spinalcord import smooth_centerline
from msct_image import Image
from msct_parser import Parser
import msct_shape
import pandas as pd
from msct_types import Centerline
from spinalcordtoolbox.centerline import optic


class Param:
    def __init__(self):
        self.debug = 0
        self.verbose = 1  # verbose
        self.step = 1  # step of discretized plane in mm default is min(x_scale,py)
        self.remove_temp_files = 1
        self.smoothing_param = 0  # window size (in mm) for smoothing CSA along z. 0 for no smoothing.
        self.slices = ''
        self.type_window = 'hanning'  # for smooth_centerline @sct_straighten_spinalcord
        self.window_length = 50  # for smooth_centerline @sct_straighten_spinalcord
        self.algo_fitting = 'hanning'  # nurbs, hanning


def get_parser():
    """
    :return: Returns the parser with the command line documentation contained in it.
    """
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description("""This program is used to get the centerline of the spinal cord of a subject by using one of the three methods describe in the -method flag .""")
    parser.add_option(name='-i',
                      type_value='image_nifti',
                      description='Spinal Cord segmentation',
                      mandatory=True,
                      example='seg.nii.gz')
    parser.add_option(name='-p',
                      type_value='multiple_choice',
                      description='type of process to be performed:\n'
                                  '- centerline: extract centerline as binary file.\n'
                                  '- label-vert: Transform segmentation into vertebral level using a file that contains labels with disc value (flag: -discfile)\n'
                                  '- length: compute length of the segmentation.\n'
                                  '- csa: computes cross-sectional area by counting pixels in each'
                                  '  slice and then geometrically adjusting using centerline orientation. Outputs:\n'
                                  '  - angle_image.nii.gz: the cord segmentation (nifti file) where each slice\'s value is equal to the CSA (mm^2),\n'
                                  '  - csa_image.nii.gz: the cord segmentation (nifti file) where each slice\'s value is equal to the angle (in degrees) between the spinal cord centerline and the inferior-superior direction,\n'
                                  '  - csa_per_slice.txt: a CSV text file with z (1st column), CSA in mm^2 (2nd column) and angle with respect to the I-S direction in degrees (3rd column),\n'
                                  '  - csa_per_slice.pickle: a pickle file with the same results as \"csa_per_slice.txt\" recorded in a DataFrame (panda structure) that can be reloaded afterwrds,\n'
                                  '  - and if you select the options -z or -vert, csa_mean and csa_volume: mean CSA and volume across the selected slices or vertebral levels is ouptut in CSV text files, an MS Excel files and a pickle files.\n'
                                  '- shape: compute spinal shape properties, using scikit-image region measures, including:\n'
                                  '  - AP and RL diameters\n'
                                  '  - ratio between AP and RL diameters\n'
                                  '  - spinal cord area\n'
                                  '  - eccentricity: Eccentricity of the ellipse that has the same second-moments as the spinal cord. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.\n'
                                  '  - equivalent diameter: The diameter of a circle with the same area as the spinal cord.\n'
                                  '  - orientation: angle (in degrees) between the AP axis of the spinal cord and the AP axis of the image\n'
                                  '  - solidity: ratio of positive (spinal cord) over null (background) pixels that are contained in the convex hull region. The convex hull region is the smallest convex polygon that surround all positive pixels in the image.',
                      mandatory=True,
                      example=['centerline', 'label-vert', 'length', 'csa', 'shape'])
    parser.usage.addSection('Optional Arguments')
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="In case you choose the option \"-p csa\", this option allows you to specify the output folder for the result files. If this folder does not exist, it will be created, otherwise the result files will be output in the pre-existing folder.",
                      mandatory=False,
                      example="My_Output_Folder/",
                      default_value="")
    parser.add_option(name='-overwrite',
                      type_value='int',
                      description="""In the case you specified, in flag \"-ofolder\", a pre-existing folder that already includes a .xls result file (see flags \"-p csa\" and \"-z\" or \"-vert\"), this option will allow you to overwrite the .xls file (\"-overwrite 1\") or to add the results to it (\"-overwrite 0\").""",
                      mandatory=False,
                      default_value=0)
    parser.add_option(name='-s',
                      type_value=None,
                      description='Window size (in mm) for smoothing CSA. 0 for no smoothing.',
                      mandatory=False,
                      deprecated_by='-size')
    parser.add_option(name='-z',
                      type_value='str',
                      description= 'Slice range to compute the CSA across (requires \"-p csa\").',
                      mandatory=False,
                      example='5:23')
    parser.add_option(name='-l',
                      type_value='str',
                      description= 'Vertebral levels to compute the CSA across (requires \"-p csa\"). Example: 2:9 for C2 to T2.',
                      mandatory=False,
                      deprecated_by='-vert',
                      example='2:9')
    parser.add_option(name='-vert',
                      type_value='str',
                      description= 'Vertebral levels to compute the CSA across (requires \"-p csa\"). Example: 2:9 for C2 to T2.',
                      mandatory=False,
                      example='2:9')
    parser.add_option(name='-t',
                      type_value='image_nifti',
                      description='Vertebral labeling file. Only use with flag -vert',
                      mandatory=False,
                      deprecated_by='-vertfile',
                      default_value='label/template/PAM50_levels.nii.gz')
    parser.add_option(name='-vertfile',
                      type_value='image_nifti',
                      description='Vertebral labeling file. Only use with flag -vert',
                      mandatory=False,
                      default_value='./label/template/PAM50_levels.nii.gz')
    parser.add_option(name='-discfile',
                      type_value='image_nifti',
                      description='Disc labeling. Only use with -p label-vert',
                      mandatory=False)
    parser.add_option(name='-r',
                      type_value='multiple_choice',
                      description= 'Removes the temporary folder and debug folder used for the algorithm at the end of execution',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name='-size',
                      type_value='int',
                      description='Window size (in mm) for smoothing CSA. 0 for no smoothing.',
                      mandatory=False,
                      default_value=0)
    parser.add_option(name='-a',
                      type_value='multiple_choice',
                      description= 'Algorithm for curve fitting.',
                      mandatory=False,
                      default_value='nurbs',
                      example=['hanning', 'nurbs'])
    parser.add_option(name='-no-angle',
                      type_value='multiple_choice',
                      description='0: angle correction for csa computation. 1: no angle correction.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value='0')
    parser.add_option(name='-use-image-coord',
                      type_value='multiple_choice',
                      description='0: physical coordinates are used to compute CSA. 1: image coordinates are used to compute CSA.\n'
                                  'Physical coordinates are less prone to instability in CSA computation and should be preferred.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value='0')
    parser.add_option(name='-v',
                      type_value='multiple_choice',
                      description='1: display on, 0: display off (default)',
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')
    parser.add_option(name='-h',
                      type_value=None,
                      description='display this help',
                      mandatory=False)

    return parser


def main(args):

    parser = get_parser()
    arguments = parser.parse(args)
    param = Param()

    # Initialization
    path_script = os.path.dirname(__file__)
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI
    processes = ['centerline', 'csa', 'length', 'shape']
    start_time = time.time()
    # spline_smoothing = param.spline_smoothing
    step = param.step
    smoothing_param = param.smoothing_param
    slices = param.slices
    angle_correction = True
    use_phys_coord = True

    fname_segmentation = arguments['-i']
    name_process = arguments['-p']
    overwrite = 0
    if "-ofolder" in arguments:
        output_folder = sct.slash_at_the_end(arguments["-ofolder"], slash=1)
    else:
        output_folder = os.getcwd() + '/'
    if '-overwrite' in arguments:
        overwrite = arguments['-overwrite']
    if '-vert' in arguments:
        vert_lev = arguments['-vert']
    else:
        vert_lev = ''
    if '-r' in arguments:
        remove_temp_files = arguments['-r']
    if '-size' in arguments:
        smoothing_param = arguments['-size']
    if '-vertfile' in arguments:
        fname_vertebral_labeling = arguments['-vertfile']
    if '-v' in arguments:
        verbose = int(arguments['-v'])
    if '-z' in arguments:
        slices = arguments['-z']
    if '-a' in arguments:
        param.algo_fitting = arguments['-a']
    if '-no-angle' in arguments:
        if arguments['-no-angle'] == '1':
            angle_correction = False
        elif arguments['-no-angle'] == '0':
            angle_correction = True
    if '-use-image-coord' in arguments:
        if arguments['-use-image-coord'] == '1':
            use_phys_coord = False
        if arguments['-use-image-coord'] == '0':
            use_phys_coord = True

    # update fields
    param.verbose = verbose

    # sct.printv(arguments)
    sct.printv('\nCheck parameters:')
    sct.printv('.. segmentation file:             ' + fname_segmentation)

    if name_process == 'centerline':
        fname_output = extract_centerline(fname_segmentation, remove_temp_files, verbose=param.verbose, algo_fitting=param.algo_fitting, use_phys_coord=use_phys_coord)
        if os.path.abspath(fname_output) != output_folder + fname_output:
            shutil.copy(fname_output, output_folder)
        # to view results
        sct.printv('\nDone! To view results, type:', param.verbose)
        sct.printv('fslview ' + fname_segmentation + ' ' + output_folder + fname_output + ' -l Red &\n', param.verbose, 'info')

    if name_process == 'csa':
        compute_csa(fname_segmentation, output_folder, overwrite, verbose, remove_temp_files, step, smoothing_param, slices, vert_lev, fname_vertebral_labeling, algo_fitting=param.algo_fitting, type_window=param.type_window, window_length=param.window_length, angle_correction=angle_correction, use_phys_coord=use_phys_coord)

    if name_process == 'label-vert':
        if '-discfile' in arguments:
            fname_disc_label = arguments['-discfile']
        else:
            sct.printv('\nERROR: Disc label file is mandatory (flag: -discfile).\n', 1, 'error')
        label_vert(fname_segmentation, fname_disc_label)

    if name_process == 'length':
        result_length = compute_length(fname_segmentation, remove_temp_files, output_folder, overwrite, slices, vert_lev, fname_vertebral_labeling, verbose=verbose)

    if name_process == 'shape':
        fname_disks = None
        if '-discfile' in arguments:
            fname_disks = arguments['-discfile']
        compute_shape(fname_segmentation, remove_temp_files, output_folder, overwrite, slices, vert_lev, fname_disks=fname_disks, verbose=verbose)

    # End of Main


def compute_shape(fname_segmentation, remove_temp_files, output_folder, overwrite, slices, vert_levels, fname_disks=None, verbose=1):
    """
    This function characterizes the shape of the spinal cord, based on the segmentation
    Shape properties are computed along the spinal cord and averaged per z-slices.
    Option is to provide intervertebral disks to average shape properties over vertebral levels (fname_disks).
    """
    # List of properties to compute on spinal cord
    property_list = ['area',
                     'diameters',
                     'equivalent_diameter',
                     'ratio_minor_major',
                     'eccentricity',
                     'solidity',
                     'symmetry']

    property_list, shape_properties = msct_shape.compute_properties_along_centerline(fname_seg_image=fname_segmentation,
                                                                                     property_list=property_list,
                                                                                     fname_disks_image=fname_disks,
                                                                                     smooth_factor=0.0,
                                                                                     interpolation_mode=0,
                                                                                     remove_temp_files=remove_temp_files,
                                                                                     verbose=verbose)

    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)
    fname_output_csv = output_folder + file_data + '_shape.csv'

    # choose sorting mode: z-slice or vertebral levels, depending on input (fname_disks)
    rejected_values = []  # some values are not vertebral levels
    if fname_disks is not None:
        # average over spinal cord levels
        sorting_mode = 'vertebral_level'
        rejected_values = [0, '0']
        # if vert_levels != '':

    else:
        # averaging over slices
        sorting_mode = 'z_slice'

    # extract all values for shape properties to be averaged on (z-slices or vertebral levels)
    sorting_values = []
    for label in shape_properties[sorting_mode]:
        if label not in sorting_values and label not in rejected_values:
            sorting_values.append(label)

    # average spinal cord shape properties
    averaged_shape = dict()
    for property_name in property_list:
        averaged_shape[property_name] = []
        for label in sorting_values:
            averaged_shape[property_name].append(np.mean([item for i, item in enumerate(shape_properties[property_name]) if shape_properties[sorting_mode][i] == label]))

    # save spinal cord shape properties
    df_shape_properties = pd.DataFrame(averaged_shape, index=sorting_values)
    df_shape_properties.sort_index(inplace=True)
    pd.set_option('expand_frame_repr', True)
    df_shape_properties.to_csv(fname_output_csv, sep=',')

    if verbose == 1:
        sct.printv(df_shape_properties)

    # display info
    sct.printv('\nDone! Results are save in the file: ' + fname_output_csv, verbose, 'info')


# compute the length of the spinal cord
# ==========================================================================================
def compute_length(fname_segmentation, remove_temp_files, output_folder, overwrite, slices, vert_levels, fname_vertebral_labeling='', verbose = 0):
    from math import sqrt

    # Extract path, file and extension
    fname_segmentation = os.path.abspath(fname_segmentation)
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = sct.slash_at_the_end('tmp.' + time.strftime("%y%m%d%H%M%S") + '_' + str(randint(1, 1000000)), 1)
    sct.run('mkdir ' + path_tmp, verbose)

    # copy files into tmp folder
    sct.printv('cp ' + fname_segmentation + ' ' + path_tmp)
    shutil.copy(fname_segmentation, path_tmp)

    if slices or vert_levels:
        # check if vertebral labeling file exists
        sct.check_file_exist(fname_vertebral_labeling)
        path_vert, file_vert, ext_vert = sct.extract_fname(fname_vertebral_labeling)
        sct.printv('cp ' + fname_vertebral_labeling + ' ' + path_tmp)
        shutil.copy(fname_vertebral_labeling, path_tmp)
        fname_vertebral_labeling = file_vert + ext_vert

    # go to tmp folder
    os.chdir(path_tmp)

    # Change orientation of the input centerline into RPI
    sct.printv('\nOrient centerline to RPI orientation...', param.verbose)
    im_seg = Image(file_data + ext_data)
    fname_segmentation_orient = 'segmentation_rpi' + ext_data
    im_seg_orient = set_orientation(im_seg, 'RPI')
    im_seg_orient.setFileName(fname_segmentation_orient)
    im_seg_orient.save()

    # Get dimension
    sct.printv('\nGet dimensions...', param.verbose)
    nx, ny, nz, nt, px, py, pz, pt = im_seg_orient.dim
    sct.printv('.. matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), param.verbose)
    sct.printv('.. voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm', param.verbose)

    # smooth segmentation/centerline
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
        fname_segmentation_orient, nurbs_pts_number=3000, phys_coordinates=False, all_slices=True, algo_fitting='nurbs', verbose=verbose)

    # average csa across vertebral levels or slices if asked (flag -z or -l)
    if slices or vert_levels:
        warning = ''
        if vert_levels and not fname_vertebral_labeling:
            sct.printv(
                '\nERROR: You asked for specific vertebral levels (option -vert) but you did not provide any vertebral labeling file (see option -vertfile). The path to the vertebral labeling file is usually \"./label/template/PAM50_levels.nii.gz\". See usage.\n',
                1, 'error')

        elif vert_levels and fname_vertebral_labeling:

            # from sct_extract_metric import get_slices_matching_with_vertebral_levels
            sct.printv('Selected vertebral levels... ' + vert_levels)

            # convert the vertebral labeling file to RPI orientation
            im_vertebral_labeling = Image(fname_vertebral_labeling)
            im_vertebral_labeling.change_orientation(orientation='RPI')

            # get the slices corresponding to the vertebral levels
            # slices, vert_levels_list, warning = get_slices_matching_with_vertebral_levels(data_seg, vert_levels, im_vertebral_labeling.data, 1)
            slices, vert_levels_list, warning = get_slices_matching_with_vertebral_levels_based_centerline(vert_levels, im_vertebral_labeling.data, z_centerline)

        elif not vert_levels:
            vert_levels_list = []

        if slices is None:
            length = np.nan
            slices = '0'
            vert_levels_list = []

        else:
            # parse the selected slices
            slices_lim = slices.strip().split(':')
            slices_list = range(int(slices_lim[0]), int(slices_lim[-1]) + 1)
            sct.printv('Spinal cord length slices ' + str(slices_lim[0]) + ' to ' + str(slices_lim[-1]) + '...',
                       type='info')

            length = 0.0
            for i in range(len(x_centerline_fit) - 1):
                if z_centerline[i] in slices_list:
                    length += sqrt(((x_centerline_fit[i + 1] - x_centerline_fit[i]) * px)**2 + ((y_centerline_fit[i + 1] - y_centerline_fit[i]) * py)**2 + ((z_centerline[i + 1] - z_centerline[i]) * pz)**2)

        sct.printv('\nLength of the segmentation = ' + str(round(length, 2)) + ' mm\n', verbose, 'info')

        # write result into output file
        save_results(output_folder + 'length', overwrite, fname_segmentation, 'length',
                     '(in mm)', length, np.nan, slices, actual_vert=vert_levels_list,
                     warning_vert_levels=warning)

    elif (not (slices or vert_levels)) and (overwrite == 1):
        sct.printv('WARNING: Flag \"-overwrite\" is only available if you select (a) slice(s) or (a) vertebral level(s) (flag -z or -vert) ==> CSA estimation per slice will be output in .txt and .pickle files only.', type='warning')
        length = np.nan

    else:
        # compute length of full centerline
        length = 0.0
        for i in range(len(x_centerline_fit) - 1):
            length += sqrt(((x_centerline_fit[i + 1] - x_centerline_fit[i]) * px)**2 + ((y_centerline_fit[i + 1] - y_centerline_fit[i]) * py)**2 + ((z_centerline[i + 1] - z_centerline[i]) * pz)**2)

        sct.printv('\nLength of the segmentation = ' + str(round(length, 2)) + ' mm\n', verbose, 'info')
        # write result into output file
        save_results(output_folder + 'length', overwrite, fname_segmentation, 'length', '(in mm)', length, np.nan,
                     slices, actual_vert=[], warning_vert_levels='')

    # Remove temporary files
    if remove_temp_files:
        sct.printv('\nRemove temporary files...', verbose)
        shutil.rmtree(path_tmp, ignore_errors=True)

    return length


# extract_centerline
# ==========================================================================================
def extract_centerline(fname_segmentation, remove_temp_files, verbose = 0, algo_fitting = 'hanning', type_window = 'hanning', window_length = 80, use_phys_coord=True):

    # Extract path, file and extension
    fname_segmentation = os.path.abspath(fname_segmentation)
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = sct.slash_at_the_end('tmp.' + time.strftime("%y%m%d%H%M%S") + '_' + str(randint(1, 1000000)), 1)
    sct.run('mkdir ' + path_tmp, verbose)

    # Copying input data to tmp folder
    sct.printv('\nCopying data to tmp folder...', verbose)
    sct.run('sct_convert -i ' + fname_segmentation + ' -o ' + path_tmp + 'segmentation.nii.gz', verbose)

    # go to tmp folder
    os.chdir(path_tmp)

    # Change orientation of the input centerline into RPI
    sct.printv('\nOrient centerline to RPI orientation...', verbose)
    # fname_segmentation_orient = 'segmentation_RPI.nii.gz'
    # BELOW DOES NOT WORK (JULIEN, 2015-10-17)
    # im_seg = Image(file_data+ext_data)
    # set_orientation(im_seg, 'RPI')
    # im_seg.setFileName(fname_segmentation_orient)
    # im_seg.save()
    sct.run('sct_image -i segmentation.nii.gz -setorient RPI -o segmentation_RPI.nii.gz', verbose)

    # Open segmentation volume
    sct.printv('\nOpen segmentation volume...', verbose)
    im_seg = Image('segmentation_RPI.nii.gz')
    data = im_seg.data

    # Get size of data
    sct.printv('\nGet data dimensions...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    sct.printv('.. matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)
    sct.printv('.. voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm', verbose)

    # # Get dimension
    # sct.printv('\nGet dimensions...', verbose)
    # nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    #
    # # Extract orientation of the input segmentation
    # orientation = get_orientation(im_seg)
    # sct.printv('\nOrientation of segmentation image: ' + orientation, verbose)
    #
    # sct.printv('\nOpen segmentation volume...', verbose)
    # data = im_seg.data
    # hdr = im_seg.hdr

    # Extract min and max index in Z direction
    X, Y, Z = (data > 0).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)
    x_centerline = [0 for i in range(0, max_z_index - min_z_index + 1)]
    y_centerline = [0 for i in range(0, max_z_index - min_z_index + 1)]
    z_centerline = [iz for iz in range(min_z_index, max_z_index + 1)]
    # Extract segmentation points and average per slice
    for iz in range(min_z_index, max_z_index + 1):
        x_seg, y_seg = (data[:, :, iz] > 0).nonzero()
        x_centerline[iz - min_z_index] = np.mean(x_seg)
        y_centerline[iz - min_z_index] = np.mean(y_seg)
    for k in range(len(X)):
        data[X[k], Y[k], Z[k]] = 0

    # extract centerline and smooth it
    if use_phys_coord:
        # fit centerline, smooth it and return the first derivative (in physical space)
        x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline('segmentation_RPI.nii.gz', algo_fitting=algo_fitting, type_window=type_window, window_length=window_length, nurbs_pts_number=3000, phys_coordinates=True, verbose=verbose, all_slices=False)
        centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

        # average centerline coordinates over slices of the image
        x_centerline_fit_rescorr, y_centerline_fit_rescorr, z_centerline_rescorr, x_centerline_deriv_rescorr, y_centerline_deriv_rescorr, z_centerline_deriv_rescorr = centerline.average_coordinates_over_slices(im_seg)

        # compute z_centerline in image coordinates for usage in vertebrae mapping
        voxel_coordinates = im_seg.transfo_phys2pix([[x_centerline_fit_rescorr[i], y_centerline_fit_rescorr[i], z_centerline_rescorr[i]] for i in range(len(z_centerline_rescorr))])
        x_centerline_voxel = [coord[0] for coord in voxel_coordinates]
        y_centerline_voxel = [coord[1] for coord in voxel_coordinates]
        z_centerline_voxel = [coord[2] for coord in voxel_coordinates]

    else:
        # fit centerline, smooth it and return the first derivative (in voxel space but FITTED coordinates)
        x_centerline_voxel, y_centerline_voxel, z_centerline_voxel, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline('segmentation_RPI.nii.gz', algo_fitting=algo_fitting, type_window=type_window, window_length=window_length, nurbs_pts_number=3000, phys_coordinates=False, verbose=verbose, all_slices=True)

    if verbose == 2:
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
    min_z_index, max_z_index = int(round(min(z_centerline_voxel))), int(round(max(z_centerline_voxel)))
    for iz in range(min_z_index, max_z_index + 1):
        data[int(round(x_centerline_voxel[iz - min_z_index])), int(round(y_centerline_voxel[iz - min_z_index])), int(iz)] = 1  # if index is out of bounds here for hanning: either the segmentation has holes or labels have been added to the file
    # Write the centerline image in RPI orientation
    # hdr.set_data_dtype('uint8') # set imagetype to uint8
    sct.printv('\nWrite NIFTI volumes...', verbose)
    im_seg.data = data
    im_seg.setFileName('centerline_RPI.nii.gz')
    im_seg.changeType('uint8')
    im_seg.save()

    sct.printv('\nSet to original orientation...', verbose)
    # get orientation of the input data
    im_seg_original = Image('segmentation.nii.gz')
    orientation = im_seg_original.orientation
    sct.run('sct_image -i centerline_RPI.nii.gz -setorient ' + orientation + ' -o centerline.nii.gz')

    # create a txt file with the centerline
    name_output_txt = 'centerline.txt'
    sct.printv('\nWrite text file...', verbose)
    file_results = open(name_output_txt, 'w')
    for i in range(min_z_index, max_z_index + 1):
        file_results.write(str(int(i)) + ' ' + str(x_centerline_voxel[i - min_z_index]) + ' ' + str(y_centerline_voxel[i - min_z_index]) + '\n')
    file_results.close()

    # create a .roi file
    fname_roi_centerline = optic.centerline2roi(fname_image='centerline_RPI.nii.gz',
                                                folder_output='./',
                                                verbose=verbose)

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(path_tmp + 'centerline.nii.gz', file_data + '_centerline.nii.gz')
    sct.generate_output_file(path_tmp + 'centerline.txt', file_data + '_centerline.txt')
    sct.generate_output_file(path_tmp + fname_roi_centerline, file_data + '_centerline.roi')

    # Remove temporary files
    if remove_temp_files:
        sct.printv('\nRemove temporary files...', verbose)
        sct.run('rm -rf ' + path_tmp, verbose)

    return file_data + '_centerline.nii.gz'


# compute_csa
# ==========================================================================================
def compute_csa(fname_segmentation, output_folder, overwrite, verbose, remove_temp_files, step, smoothing_param, slices, vert_levels, fname_vertebral_labeling='', algo_fitting='hanning', type_window='hanning', window_length=80, angle_correction=True, use_phys_coord=True):

    from math import degrees
    import pandas as pd
    import pickle

    # Extract path, file and extension
    fname_segmentation = os.path.abspath(fname_segmentation)
    # path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = sct.slash_at_the_end('tmp.' + time.strftime("%y%m%d%H%M%S") + '_' + str(randint(1, 1000000)), 1)
    sct.run('mkdir ' + path_tmp, verbose)

    # Copying input data to tmp folder
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    sct.run('sct_convert -i ' + fname_segmentation + ' -o ' + path_tmp + 'segmentation.nii.gz', verbose)
    # go to tmp folder
    os.chdir(path_tmp)
    # Change orientation of the input segmentation into RPI
    sct.printv('\nChange orientation to RPI...', verbose)
    sct.run('sct_image -i segmentation.nii.gz -setorient RPI -o segmentation_RPI.nii.gz', verbose)

    # Open segmentation volume
    sct.printv('\nOpen segmentation volume...', verbose)
    im_seg = Image('segmentation_RPI.nii.gz')
    data_seg = im_seg.data
    # hdr_seg = im_seg.hdr

    # Get size of data
    sct.printv('\nGet data dimensions...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim
    sct.printv('  ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)

    # # Extract min and max index in Z direction
    X, Y, Z = (data_seg > 0).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # if angle correction is required, get segmentation centerline
    if angle_correction:
        if use_phys_coord:
            # fit centerline, smooth it and return the first derivative (in physical space)
            x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline('segmentation_RPI.nii.gz', algo_fitting=algo_fitting, type_window=type_window, window_length=window_length, nurbs_pts_number=3000, phys_coordinates=True, verbose=verbose, all_slices=False)
            centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

            # average centerline coordinates over slices of the image
            x_centerline_fit_rescorr, y_centerline_fit_rescorr, z_centerline_rescorr, x_centerline_deriv_rescorr, y_centerline_deriv_rescorr, z_centerline_deriv_rescorr = centerline.average_coordinates_over_slices(im_seg)

            # compute Z axis of the image, in physical coordinate
            axis_X, axis_Y, axis_Z = im_seg.get_directions()

            # compute z_centerline in image coordinates for usage in vertebrae mapping
            z_centerline_voxel = [coord[2] for coord in im_seg.transfo_phys2pix([[x_centerline_fit_rescorr[i], y_centerline_fit_rescorr[i], z_centerline_rescorr[i]] for i in range(len(z_centerline_rescorr))])]

        else:
            # fit centerline, smooth it and return the first derivative (in voxel space but FITTED coordinates)
            x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline('segmentation_RPI.nii.gz', algo_fitting=algo_fitting, type_window=type_window, window_length=window_length, nurbs_pts_number=3000, phys_coordinates=False, verbose=verbose, all_slices=True)

            # correct centerline fitted coordinates according to the data resolution
            x_centerline_fit_rescorr, y_centerline_fit_rescorr, z_centerline_rescorr, x_centerline_deriv_rescorr, y_centerline_deriv_rescorr, z_centerline_deriv_rescorr = x_centerline_fit * px, y_centerline_fit * py, z_centerline * pz, x_centerline_deriv * px, y_centerline_deriv * py, z_centerline_deriv * pz

            axis_Z = [0.0, 0.0, 1.0]

            # compute z_centerline in image coordinates for usage in vertebrae mapping
            z_centerline_voxel = z_centerline

    # Compute CSA
    sct.printv('\nCompute CSA...', verbose)

    # Empty arrays in which CSA for each z slice will be stored
    csa = np.zeros(max_z_index - min_z_index + 1)
    angles = np.zeros(max_z_index - min_z_index + 1)

    for iz in xrange(min_z_index, max_z_index + 1):
        if angle_correction:
            # in the case of problematic segmentation (e.g., non continuous segmentation often at the extremities), display a warning but do not crash
            try:
                # normalize the tangent vector to the centerline (i.e. its derivative)
                tangent_vect = normalize(np.array([x_centerline_deriv_rescorr[iz - min_z_index], y_centerline_deriv_rescorr[iz - min_z_index], z_centerline_deriv_rescorr[iz - min_z_index]]))

            except IndexError:
                sct.printv('WARNING: Your segmentation does not seem continuous, which could cause wrong estimations at the problematic slices. Please check it, especially at the extremities.', type='warning')

            # compute the angle between the normal vector of the plane and the vector z
            angle = np.arccos(np.vdot(tangent_vect, axis_Z))
        else:
            angle = 0.0

        # compute the number of voxels, assuming the segmentation is coded for partial volume effect between 0 and 1.
        number_voxels = np.sum(data_seg[:, :, iz])

        # compute CSA, by scaling with voxel size (in mm) and adjusting for oblique plane
        csa[iz - min_z_index] = number_voxels * px * py * np.cos(angle)
        angles[iz - min_z_index] = degrees(angle)

    sct.printv('\nSmooth CSA across slices...', verbose)
    if smoothing_param:
        from msct_smooth import smoothing_window
        sct.printv('.. Hanning window: ' + str(smoothing_param) + ' mm', verbose)
        csa_smooth = smoothing_window(csa, window_len=smoothing_param / pz, window='hanning', verbose=0)
        # display figure
        if verbose == 2:
            import matplotlib.pyplot as plt
            plt.figure()
            z_centerline_scaled = [x * pz for x in z_centerline]
            pltx, = plt.plot(z_centerline_scaled, csa, 'bo')
            pltx_fit, = plt.plot(z_centerline_scaled, csa_smooth, 'r', linewidth=2)
            plt.title("Cross-sectional area (CSA)")
            plt.xlabel('z (mm)')
            plt.ylabel('CSA (mm^2)')
            plt.legend([pltx, pltx_fit], ['Raw', 'Smoothed'])
            plt.show()
        # update variable
        csa = csa_smooth
    else:
        sct.printv('.. No smoothing!', verbose)

    # output volume of csa values
    sct.printv('\nCreate volume of CSA values...', verbose)
    data_csa = data_seg.astype(np.float32, copy=False)
    # loop across slices
    for iz in range(min_z_index, max_z_index + 1):
        # retrieve seg pixels
        x_seg, y_seg = (data_csa[:, :, iz] > 0).nonzero()
        seg = [[x_seg[i], y_seg[i]] for i in range(0, len(x_seg))]
        # loop across pixels in segmentation
        for i in seg:
            # replace value with csa value
            data_csa[i[0], i[1], iz] = csa[iz - min_z_index]
    # replace data
    im_seg.data = data_csa
    # set original orientation
    # TODO: FIND ANOTHER WAY!!
    # im_seg.change_orientation(orientation) --> DOES NOT WORK!
    # set file name -- use .gz because faster to write
    im_seg.setFileName('csa_volume_RPI.nii.gz')
    im_seg.changeType('float32')
    # save volume
    im_seg.save()

    # output volume of csa values
    sct.printv('\nCreate volume of angle values...', verbose)
    data_angle = data_seg.astype(np.float32, copy=False)
    # loop across slices
    for iz in range(min_z_index, max_z_index + 1):
        # retrieve seg pixels
        x_seg, y_seg = (data_angle[:, :, iz] > 0).nonzero()
        seg = [[x_seg[i], y_seg[i]] for i in range(0, len(x_seg))]
        # loop across pixels in segmentation
        for i in seg:
            # replace value with csa value
            data_angle[i[0], i[1], iz] = angles[iz - min_z_index]
    # replace data
    im_seg.data = data_angle
    # set original orientation
    # TODO: FIND ANOTHER WAY!!
    # im_seg.change_orientation(orientation) --> DOES NOT WORK!
    # set file name -- use .gz because faster to write
    im_seg.setFileName('angle_volume_RPI.nii.gz')
    im_seg.changeType('float32')
    # save volume
    im_seg.save()

    # get orientation of the input data
    im_seg_original = Image('segmentation.nii.gz')
    orientation = im_seg_original.orientation
    sct.run('sct_image -i csa_volume_RPI.nii.gz -setorient ' + orientation + ' -o csa_volume_in_initial_orientation.nii.gz')
    sct.run('sct_image -i angle_volume_RPI.nii.gz -setorient ' + orientation + ' -o angle_volume_in_initial_orientation.nii.gz')

    # come back to parent folder
    os.chdir('..')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(path_tmp + 'csa_volume_in_initial_orientation.nii.gz', output_folder + 'csa_image.nii.gz')  # extension already included in name_output
    sct.generate_output_file(path_tmp + 'angle_volume_in_initial_orientation.nii.gz', output_folder + 'angle_image.nii.gz')  # extension already included in name_output
    sct.printv('\n')

    # Create output text file
    sct.printv('Display CSA per slice:', verbose)
    file_results = open(output_folder + 'csa_per_slice.txt', 'w')
    file_results.write('# Slice (z),CSA (mm^2),Angle with respect to the I-S direction (degrees)\n')
    for i in range(min_z_index, max_z_index + 1):
        file_results.write(str(int(i)) + ',' + str(csa[i - min_z_index]) + ',' + str(angles[i - min_z_index]) + '\n')
        # Display results
        sct.printv('z = %d, CSA = %f mm^2, Angle = %f deg' % (i, csa[i - min_z_index], angles[i - min_z_index]), type='info')
    file_results.close()
    sct.printv('Save results in: ' + output_folder + 'csa_per_slice.txt\n', verbose)

    # Create output pickle file
    # data frame format
    results_df = pd.DataFrame({'Slice (z)': range(min_z_index, max_z_index + 1),
                               'CSA (mm^2)': csa,
                               'Angle with respect to the I-S direction (degrees)': angles})
    # # dictionary format
    # results_df = {'Slice (z)': range(min_z_index, max_z_index+1),
    #                            'CSA (mm^2)': csa,
    #                            'Angle with respect to the I-S direction (degrees)': angles}
    output_file = open(output_folder + 'csa_per_slice.pickle', 'wb')
    pickle.dump(results_df, output_file)
    output_file.close()
    sct.printv('Save results in: ' + output_folder + 'csa_per_slice.pickle\n', verbose)

    # average csa across vertebral levels or slices if asked (flag -z or -l)
    if slices or vert_levels:

        warning = ''
        if vert_levels and not fname_vertebral_labeling:
            sct.printv('\nERROR: You asked for specific vertebral levels (option -vert) but you did not provide any vertebral labeling file (see option -vertfile). The path to the vertebral labeling file is usually \"./label/template/PAM50_levels.nii.gz\". See usage.\n', 1, 'error')

        elif vert_levels and fname_vertebral_labeling:
            # from sct_extract_metric import get_slices_matching_with_vertebral_levels
            sct.printv('Selected vertebral levels... ' + vert_levels)

            # check if vertebral labeling file exists
            sct.check_file_exist(fname_vertebral_labeling)

            # convert the vertebral labeling file to RPI orientation
            im_vertebral_labeling = Image(fname_vertebral_labeling)
            im_vertebral_labeling.change_orientation(orientation='RPI')

            # get the slices corresponding to the vertebral levels
            # slices, vert_levels_list, warning = get_slices_matching_with_vertebral_levels(data_seg, vert_levels, im_vertebral_labeling.data, 1)
            slices, vert_levels_list, warning = get_slices_matching_with_vertebral_levels_based_centerline(vert_levels, im_vertebral_labeling.data, z_centerline_voxel)

        elif not vert_levels:
            vert_levels_list = []

        if slices is None:
            mean_CSA = 0.0
            std_CSA = 0.0
            mean_angle = 0.0
            std_angle = 0.0
            slices = '0'
            vert_levels_list = []

        else:
            # parse the selected slices
            slices_lim = slices.strip().split(':')
            slices_list = range(int(slices_lim[0]), int(slices_lim[-1]) + 1)
            sct.printv('Average CSA across slices ' + str(slices_lim[0]) + ' to ' + str(slices_lim[-1]) + '...', type='info')

            CSA_for_selected_slices = []
            angles_for_selected_slices = []
            # Read the file csa_per_slice.txt and get the CSA for the selected slices
            with open(output_folder + 'csa_per_slice.txt') as openfile:
                for line in openfile:
                    if line[0] != '#':
                        line_split = line.strip().split(',')
                        if int(line_split[0]) in slices_list:
                            CSA_for_selected_slices.append(float(line_split[1]))
                            angles_for_selected_slices.append(float(line_split[2]))

            # average the CSA and angle
            mean_CSA = np.mean(np.asarray(CSA_for_selected_slices))
            std_CSA = np.std(np.asarray(CSA_for_selected_slices))
            mean_angle = np.mean(np.asarray(angles_for_selected_slices))
            std_angle = np.std(np.asarray(angles_for_selected_slices))

        sct.printv('Mean CSA: ' + str(mean_CSA) + ' +/- ' + str(std_CSA) + ' mm^2', type='info')
        sct.printv('Mean angle: ' + str(mean_angle) + ' +/- ' + str(std_angle) + ' degrees', type='info')

        # write result into output file
        save_results(output_folder + 'csa_mean', overwrite, fname_segmentation, 'CSA', 'nb_voxels x px x py x cos(theta) slice-by-slice (in mm^2)', mean_CSA, std_CSA, slices, actual_vert=vert_levels_list, warning_vert_levels=warning)
        save_results(output_folder + 'angle_mean', overwrite, fname_segmentation, 'Angle with respect to the I-S direction', 'Unit z vector compared to the unit tangent vector to the centerline at each slice (in degrees)', mean_angle, std_angle, slices, actual_vert=vert_levels_list, warning_vert_levels=warning)

        # compute volume between the selected slices
        if slices == '0':
            volume = 0.0
        else:
            sct.printv('Compute the volume in between slices ' + str(slices_lim[0]) + ' to ' + str(slices_lim[-1]) + '...', type='info')
            nb_vox = np.sum(data_seg[:, :, slices_list])
            volume = nb_vox * px * py * pz
        sct.printv('Volume in between the selected slices: ' + str(volume) + ' mm^3', type='info')

        # write result into output file
        save_results(output_folder + 'csa_volume', overwrite, fname_segmentation, 'volume', 'nb_voxels x px x py x pz (in mm^3)', volume, np.nan, slices, actual_vert=vert_levels_list, warning_vert_levels=warning)

    elif (not (slices or vert_levels)) and (overwrite == 1):
        sct.printv('WARNING: Flag \"-overwrite\" is only available if you select (a) slice(s) or (a) vertebral level(s) (flag -z or -vert) ==> CSA estimation per slice will be output in .txt and .pickle files only.', type='warning')

    # Remove temporary files
    if remove_temp_files:
        sct.printv('\nRemove temporary files...')
        sct.run('rm -rf ' + path_tmp, error_exit='warning')

    # Sum up the output file names
    sct.printv('\nOutput a nifti file of CSA values along the segmentation: ' + output_folder + 'csa_image.nii.gz', verbose, 'info')
    sct.printv('Output result text file of CSA per slice: ' + output_folder + 'csa_per_slice.txt', verbose, 'info')
    if slices or vert_levels:
        sct.printv('Output result files of the mean CSA across the selected slices: \n\t\t' + output_folder + 'csa_mean.txt\n\t\t' + output_folder + 'csa_mean.xls\n\t\t' + output_folder + 'csa_mean.pickle', verbose, 'info')
        sct.printv('Output result files of the volume in between the selected slices: \n\t\t' + output_folder + 'csa_volume.txt\n\t\t' + output_folder + 'csa_volume.xls\n\t\t' + output_folder + 'csa_volume.pickle', verbose, 'info')


def label_vert(fname_seg, fname_label, verbose=1):
    """
    Label segmentation using vertebral labeling information
    :param fname_segmentation:
    :param fname_label:
    :param verbose:
    :return:
    """
    # Open labels
    im_disc = Image(fname_label)
    # retrieve all labels
    coord_label = im_disc.getNonZeroCoordinates()
    # compute list_disc_z and list_disc_value
    list_disc_z = []
    list_disc_value = []
    for i in range(len(coord_label)):
        list_disc_z.insert(0, coord_label[i].z)
        list_disc_value.insert(0, coord_label[i].value)

    list_disc_value = [x for (y, x) in sorted(zip(list_disc_z, list_disc_value), reverse=True)]
    list_disc_z = [y for (y, x) in sorted(zip(list_disc_z, list_disc_value), reverse=True)]
    # label segmentation
    from sct_label_vertebrae import label_segmentation
    label_segmentation(fname_seg, list_disc_z, list_disc_value, verbose=verbose)
    # Generate output files
    sct.printv('--> File created: ' + sct.add_suffix(fname_seg, '_labeled.nii.gz'), verbose)


# ======================================================================================================================
# Save CSA or volume estimation in a .txt file
# ======================================================================================================================
def save_results(fname_output, overwrite, fname_data, metric_name, method, mean, std, slices_of_interest, actual_vert, warning_vert_levels):

    # define vertebral levels and slices fields
    if actual_vert:
        vertebral_levels_field = str(int(actual_vert[0])) + ' to ' + str(int(actual_vert[1]))
        if warning_vert_levels:
            for i in range(0, len(warning_vert_levels)):
                vertebral_levels_field += ' [' + str(warning_vert_levels[i]) + ']'
    else:
        if slices_of_interest != '':
            vertebral_levels_field = str(np.nan)
        else:
            vertebral_levels_field = 'ALL'

    if slices_of_interest != '':
        slices_of_interest_field = slices_of_interest
    else:
        slices_of_interest_field = 'ALL'

    sct.printv('Save results in: ' + fname_output + '.txt\n')

    # Save results in a CSV text file
    # CSV format, header lines start with "#"
    fid_metric = open(fname_output + '.txt', 'w')

    # WRITE HEADER:
    # Write date and time
    fid_metric.write('# Date - Time: ' + time.strftime('%Y/%m/%d - %H:%M:%S'))
    # Write file with absolute path
    fid_metric.write('\n' + '# File: ' + os.path.abspath(fname_data))
    # Write metric
    fid_metric.write('\n' + '# Metric: ' + metric_name)
    # Write method used for the metric estimation
    fid_metric.write('\n' + '# Calculation method: ' + method)
    # Write selected vertebral levels
    fid_metric.write('\n# Vertebral levels: ' + vertebral_levels_field)
    # Write selected slices
    fid_metric.write('\n' + '# Slices (z): ' + slices_of_interest)
    # label headers
    fid_metric.write('%s' % ('\n' + '# MEAN, STDEV\n'))

    # WRITE RESULTS
    fid_metric.write('%f, %f\n' % (mean, std))

    # Close file .txt
    fid_metric.close()

    # Save results in a MS Excel file
    # if the user asked for no overwriting but the specified output file does not exist yet
    if (not overwrite) and (not os.path.isfile(fname_output + '.xls')):
        sct.printv('WARNING: You asked to edit the pre-existing file \"' + fname_output + '.xls\" but this file does not exist. It will be created.', type='warning')
        overwrite = 1

    if not overwrite:
        from xlrd import open_workbook
        from xlutils.copy import copy

        existing_book = open_workbook(fname_output + '.xls')

        # get index of the first empty row and leave one empty row between the two subjects
        row_index = existing_book.sheet_by_index(0).nrows

        book = copy(existing_book)
        sh = book.get_sheet(0)

    elif overwrite:
        from xlwt import Workbook

        book = Workbook()
        sh = book.add_sheet('Results', cell_overwrite_ok=True)

        # write header line
        sh.write(0, 0, 'Date - Time')
        sh.write(0, 1, 'File used for calculation')
        sh.write(0, 2, 'Metric')
        sh.write(0, 3, 'Calculation method')
        sh.write(0, 4, 'Vertebral levels')
        sh.write(0, 5, 'Slices (z)')
        sh.write(0, 6, 'MEAN across slices')
        sh.write(0, 7, 'STDEV across slices')

        row_index = 1

    # write results
    sh.write(row_index, 0, time.strftime('%Y/%m/%d - %H:%M:%S'))
    sh.write(row_index, 1, os.path.abspath(fname_data))
    sh.write(row_index, 2, metric_name)
    sh.write(row_index, 3, method)
    sh.write(row_index, 4, vertebral_levels_field)
    sh.write(row_index, 5, slices_of_interest_field)
    sh.write(row_index, 6, float(mean))
    sh.write(row_index, 7, str(std))

    book.save(fname_output + '.xls')

    # Save results in a pickle file
    # write results in a dictionary
    output_results = {}
    output_results['Date - Time'] = time.strftime('%Y/%m/%d - %H:%M:%S')
    output_results['File used for calculation'] = os.path.abspath(fname_data)
    output_results['Metric'] = metric_name
    output_results['Calculation method'] = method
    output_results['Vertebral levels'] = vertebral_levels_field
    output_results['Slices (z)'] = slices_of_interest_field
    output_results['MEAN across slices'] = float(mean)
    output_results['STDEV across slices'] = str(std)

    # save "output_results"
    import pickle
    output_file = open(fname_output + '.pickle', 'wb')
    pickle.dump(output_results, output_file)
    output_file.close()


# ======================================================================================================================
# Find min and max slices corresponding to vertebral levels based on the fitted centerline coordinates
# ======================================================================================================================
def get_slices_matching_with_vertebral_levels_based_centerline(vertebral_levels, vertebral_labeling_data, z_centerline):

    # Convert the selected vertebral levels chosen into a 2-element list [start_level end_level]
    vert_levels_list = [int(x) for x in vertebral_levels.split(':')]

    # If only one vertebral level was selected (n), consider as n:n
    if len(vert_levels_list) == 1:
        vert_levels_list = [vert_levels_list[0], vert_levels_list[0]]

    # Check if there are only two values [start_level, end_level] and if the end level is higher than the start level
    if (len(vert_levels_list) > 2) or (vert_levels_list[0] > vert_levels_list[1]):
        sct.printv('\nERROR:  "' + vertebral_levels + '" is not correct. Enter format "1:4". Exit program.\n', type='error')

    # Extract the vertebral levels available in the metric image
    vertebral_levels_available = np.array(list(set(vertebral_labeling_data[vertebral_labeling_data > 0])), dtype=np.int32)

    # Check if the vertebral levels selected are available
    warning = []  # list of strings gathering the potential following warning(s) to be written in the output .txt file
    if len(vertebral_levels_available) == 0:
        slices = None
        vert_levels_list = None
        warning.append('\tError: no slices with corresponding vertebral levels were found.')
        return slices, vert_levels_list, warning
    else:
        min_vert_level_available = min(vertebral_levels_available)  # lowest vertebral level available
        max_vert_level_available = max(vertebral_levels_available)  # highest vertebral level available

    if vert_levels_list[0] < min_vert_level_available:
        vert_levels_list[0] = min_vert_level_available
        warning.append('WARNING: the bottom vertebral level you selected is lower to the lowest level available --> '
                       'Selected the lowest vertebral level available: ' + str(int(vert_levels_list[0])))  # record the
                       # warning to write it later in the .txt output file
        sct.printv('WARNING: the bottom vertebral level you selected is lower to the lowest ' \
                                          'level available \n--> Selected the lowest vertebral level available: ' +\
              str(int(vert_levels_list[0])), type='warning')

    if vert_levels_list[1] > max_vert_level_available:
        vert_levels_list[1] = max_vert_level_available
        warning.append('WARNING: the top vertebral level you selected is higher to the highest level available --> '
                       'Selected the highest vertebral level available: ' + str(int(vert_levels_list[1])))  # record the
        # warning to write it later in the .txt output file

        sct.printv('WARNING: the top vertebral level you selected is higher to the highest ' \
                                          'level available \n--> Selected the highest vertebral level available: ' + \
              str(int(vert_levels_list[1])), type='warning')

    if vert_levels_list[0] not in vertebral_levels_available:
        distance = vertebral_levels_available - vert_levels_list[0]  # relative distance
        distance_min_among_negative_value = min(abs(distance[distance < 0]))  # minimal distance among the negative
        # relative distances
        vert_levels_list[0] = vertebral_levels_available[distance == distance_min_among_negative_value]  # element
        # of the initial list corresponding to this minimal distance
        warning.append('WARNING: the bottom vertebral level you selected is not available --> Selected the nearest '
                       'inferior level available: ' + str(int(vert_levels_list[0])))
        sct.printv('WARNING: the bottom vertebral level you selected is not available \n--> Selected the ' \
                             'nearest inferior level available: ' + str(int(vert_levels_list[0])), type='warning')  # record the
        # warning to write it later in the .txt output file

    if vert_levels_list[1] not in vertebral_levels_available:
        distance = vertebral_levels_available - vert_levels_list[1]  # relative distance
        distance_min_among_positive_value = min(abs(distance[distance > 0]))  # minimal distance among the negative
        # relative distances
        vert_levels_list[1] = vertebral_levels_available[distance == distance_min_among_positive_value]  # element
        # of the initial list corresponding to this minimal distance
        warning.append('WARNING: the top vertebral level you selected is not available --> Selected the nearest superior'
                       ' level available: ' + str(int(vert_levels_list[1])))  # record the warning to write it later in the .txt output file

        sct.printv('WARNING: the top vertebral level you selected is not available \n--> Selected the ' \
                             'nearest superior level available: ' + str(int(vert_levels_list[1])), type='warning')

    # Find slices included in the vertebral levels wanted by the user
    # if the median vertebral level of this slice is in the vertebral levels asked by the user, record the slice number
    sct.printv('\tFind slices corresponding to vertebral levels based on the centerline...')
    matching_slices_centerline_vert_labeling = []

    z_centerline = [x for x in z_centerline if 0 < int(x) < vertebral_labeling_data.shape[2]]
    vert_range = range(vert_levels_list[0], vert_levels_list[1] + 1)

    for idx, z_slice in enumerate(vertebral_labeling_data.T[z_centerline, :, :]):
        slice_idxs = np.nonzero(z_slice)
        if np.asarray(slice_idxs).shape != (2, 0) and int(np.median(z_slice[slice_idxs])) in vert_range:
            matching_slices_centerline_vert_labeling.append(idx)

    # now, find the min and max slices that are included in the vertebral levels
    if len(matching_slices_centerline_vert_labeling) == 0:
        slices = None
        vert_levels_list = None
        warning.append('\tError: no slices with corresponding vertebral levels were found.')
    else:
        slices = str(min(matching_slices_centerline_vert_labeling)) + ':' + str(max(matching_slices_centerline_vert_labeling))
        sct.printv('\t' + slices)

    return slices, vert_levels_list, warning


def b_spline_centerline(x_centerline, y_centerline, z_centerline):
    sct.printv('\nFitting centerline using B-spline approximation...')
    points = [[x_centerline[n], y_centerline[n], z_centerline[n]] for n in range(len(x_centerline))]
    nurbs = NURBS(3, 3000, points)
    # BE very careful with the spline order that you choose :
    # if order is too high ( > 4 or 5) you need to set a higher number of Control Points (cf sct_nurbs ).
    # For the third argument (number of points), give at least len(z_centerline)+500 or higher

    P = nurbs.getCourbe3D()
    x_centerline_fit = P[0]
    y_centerline_fit = P[1]
    Q = nurbs.getCourbe3D_deriv()
    x_centerline_deriv = Q[0]
    y_centerline_deriv = Q[1]
    z_centerline_deriv = Q[2]

    return x_centerline_fit, y_centerline_fit, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv


#=======================================================================================================================
# Normalization
#=======================================================================================================================
def normalize(vect):
    norm = np.linalg.norm(vect)
    return vect / norm


#=======================================================================================================================
# Ellipse fitting for a set of data
#=======================================================================================================================
# http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
def Ellipse_fit(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D =  np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a


#=======================================================================================================================
# Getting a and b parameter for fitted ellipse
#=======================================================================================================================
def ellipse_dim(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])


#=======================================================================================================================
# Detect edges of an image
#=======================================================================================================================
def edge_detection(f):

    img = Image.open(f)  # grayscale
    imgdata = np.array(img, dtype = float)
    G = imgdata
    #G = ndi.filters.gaussian_filter(imgdata, sigma)
    gradx = np.array(G, dtype = float)
    grady = np.array(G, dtype = float)

    mask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    mask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    width = img.size[1]
    height = img.size[0]

    for i in range(1, width - 1):
        for j in range(1, height - 1):

            px = np.sum(mask_x * G[(i - 1):(i + 1) + 1, (j - 1):(j + 1) + 1])
            py = np.sum(mask_y * G[(i - 1):(i + 1) + 1, (j - 1):(j + 1) + 1])
            gradx[i][j] = px
            grady[i][j] = py

    mag = scipy.hypot(gradx, grady)

    treshold = np.max(mag) * 0.9

    for i in range(width):
        for j in range(height):
            if mag[i][j] > treshold:
                mag[i][j] = 1
            else:
                mag[i][j] = 0

    return mag


if __name__ == "__main__":
    sct.start_stream_logger()
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main(sys.argv[1:])
