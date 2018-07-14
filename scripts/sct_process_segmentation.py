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

import sys, io, os, shutil, time, math, pickle

import numpy as np
import pandas as pd

import sct_utils as sct

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
    from msct_parser import Parser
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
                      deprecated_by='-vertfile')
    parser.add_option(name='-vertfile',
                      type_value='str',
                      description='Vertebral labeling file. Only use with flag -vert',
                      default_value='./label/template/PAM50_levels.nii.gz',
                      mandatory=False)
    parser.add_option(name='-discfile',
                      type_value='image_nifti',
                      description='Disc labeling with the convention "disc labelvalue=3 ==> disc C2/C3". Only use with -p label-vert',
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
    fname_vertebral_labeling = ''
    if "-ofolder" in arguments:
        output_folder = arguments["-ofolder"]
    else:
        output_folder = os.getcwd()

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
        from spinalcordtoolbox.process_segmentation.script import extract_centerline
        fname_output = extract_centerline(fname_segmentation, remove_temp_files, verbose=param.verbose, algo_fitting=param.algo_fitting, use_phys_coord=use_phys_coord)
        sct.copy(fname_output, output_folder)
        sct.display_viewer_syntax([fname_segmentation, os.path.join(output_folder, fname_output)], colormaps=['gray', 'red'], opacities=['', '1'])

    if name_process == 'csa':
        from spinalcordtoolbox.process_segmentation.script import compute_csa
        compute_csa(fname_segmentation, output_folder, overwrite, verbose, remove_temp_files, step, smoothing_param, slices, vert_lev, fname_vertebral_labeling, algo_fitting=param.algo_fitting, type_window=param.type_window, window_length=param.window_length, angle_correction=angle_correction, use_phys_coord=use_phys_coord)

    if name_process == 'label-vert':
        if '-discfile' in arguments:
            fname_disc_label = arguments['-discfile']
        else:
            sct.printv('\nERROR: Disc label file is mandatory (flag: -discfile).\n', 1, 'error')

        from spinalcordtoolbox.process_segmentation.script import label_vert
        label_vert(fname_segmentation, fname_disc_label)

    if name_process == 'length':
        from spinalcordtoolbox.process_segmentation.script import compute_length
        result_length = compute_length(fname_segmentation, remove_temp_files, output_folder, overwrite, slices, vert_lev, fname_vertebral_labeling, verbose=verbose)

    if name_process == 'shape':
        fname_disks = None
        if '-discfile' in arguments:
            fname_disks = arguments['-discfile']


        from spinalcordtoolbox.process_segmentation.script import compute_shape
        compute_shape(fname_segmentation, remove_temp_files, output_folder, overwrite, slices, vert_lev, fname_disks=fname_disks, verbose=verbose)

    # End of Main


if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main(sys.argv[1:])
