#!/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Function to segment the spinal cord using convolutional neural networks
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener & Charley Gros
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import division, absolute_import

import os
import sys

import sct_utils as sct
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.deepseg_sc.core import deep_segmentation_spinalcord
from spinalcordtoolbox.reports.qc import generate_qc
from msct_parser import Parser


def get_parser():
    """Initialize the parser."""
    parser = Parser(__file__)
    parser.usage.set_description("""Spinal Cord Segmentation using convolutional networks. \n\nReference: C Gros, B De Leener, et al. Automatic segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional neural networks (2018). arxiv.org/abs/1805.06349""")

    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="input image.",
                      mandatory=True,
                      example="t1.nii.gz")
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast.",
                      mandatory=True,
                      example=['t1', 't2', 't2s', 'dwi'])
    parser.add_option(name="-centerline",
                      type_value="multiple_choice",
                      description="Method used for extracting the centerline.\nsvm: automatic centerline detection, based on Support Vector Machine algorithm.\ncnn: automatic centerline detection, based on Convolutional Neural Network.\nviewer: semi-automatic centerline generation, based on manual selection of a few points using an interactive viewer, then approximation with NURBS.\nfile: use an existing centerline by specifying its filename with flag -file_centerline (e.g. -file_centerline t2_centerline_manual.nii.gz).\n",
                      mandatory=False,
                      example=['svm', 'cnn', 'viewer', 'file'],
                      default_value="svm")
    parser.add_option(name="-file_centerline",
                      type_value="image_nifti",
                      description="Input centerline file (to use with flag -centerline file).",
                      mandatory=False,
                      example="t2_centerline_manual.nii.gz")
    parser.add_option(name="-brain",
                      type_value="multiple_choice",
                      description="indicate if the input image is expected to contain brain sections:\n1: contains brain section\n0: no brain section.\nTo indicate this parameter could speed the segmentation process. Note that this flag is only effective with -centerline cnn.",
                      mandatory=False,
                      example=["0", "1"])
    parser.add_option(name="-kernel",
                      type_value="multiple_choice",
                      description="choice of 2D or 3D kernels for the segmentation. Note that segmentation with 3D kernels is significantely longer than with 2D kernels.",
                      mandatory=False,
                      example=['2d', '3d'],
                      default_value="2d")
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="output folder.",
                      mandatory=False,
                      example="My_Output_Folder/",
                      default_value="")
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="remove temporary files.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="1: display on, 0: display off (default)",
                      mandatory=False,
                      example=["0", "1"],
                      default_value="1")
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=None)
    parser.add_option(name='-igt',
                      type_value='image_nifti',
                      description='File name of ground-truth segmentation.',
                      mandatory=False)
    return parser


def main():
    """Main function."""
    sct.init_sct()
    parser = get_parser()
    args = sys.argv[1:]
    arguments = parser.parse(args)

    fname_image = os.path.abspath(arguments['-i'])
    contrast_type = arguments['-c']

    ctr_algo = arguments["-centerline"]

    if "-brain" not in args:
        if contrast_type in ['t2s', 'dwi']:
            brain_bool = False
        if contrast_type in ['t1', 't2']:
            brain_bool = True
    else:
        brain_bool = bool(int(arguments["-brain"]))

    kernel_size = arguments["-kernel"]
    if kernel_size == '3d' and contrast_type == 'dwi':
        kernel_size = '2d'
        sct.printv('3D kernel model for dwi contrast is not available. 2D kernel model is used instead.', type="warning")

    if '-ofolder' not in args:
        output_folder = os.getcwd()
    else:
        output_folder = arguments["-ofolder"]

    if ctr_algo == 'file' and "-file_centerline" not in args:
        sct.log.warning('Please use the flag -file_centerline to indicate the centerline filename.')
        sys.exit(1)
    
    if "-file_centerline" in args:
        manual_centerline_fname = arguments["-file_centerline"]
        ctr_algo = 'file'
    else:
        manual_centerline_fname = None

    remove_temp_files = int(arguments['-r'])

    verbose = arguments['-v']

    path_qc = arguments.get("-qc", None)

    algo_config_stg = '\nMethod:'
    algo_config_stg += '\n\tCenterline algorithm: ' + str(ctr_algo)
    algo_config_stg += '\n\tAssumes brain section included in the image: ' + str(brain_bool)
    algo_config_stg += '\n\tDimension of the segmentation kernel convolutions: ' + kernel_size + '\n'
    sct.printv(algo_config_stg)

    im_image = Image(fname_image)
    # note: below we pass im_image.copy() otherwise the field absolutepath becomes None after execution of this function
    im_seg, im_image_RPI_upsamp, im_seg_RPI_upsamp = deep_segmentation_spinalcord(
        im_image.copy(), contrast_type, ctr_algo=ctr_algo, ctr_file=manual_centerline_fname,
        brain_bool=brain_bool, kernel_size=kernel_size, remove_temp_files=remove_temp_files, verbose=verbose)

    # Save segmentation
    fname_seg = os.path.abspath(os.path.join(output_folder, sct.extract_fname(fname_image)[1] + '_seg' +
                                             sct.extract_fname(fname_image)[2]))
    im_seg.save(fname_seg)

    if path_qc is not None:
        generate_qc(fname_image, fname_seg=fname_seg, args=args, path_qc=os.path.abspath(path_qc),
                    process='sct_deepseg_sc')
    sct.display_viewer_syntax([fname_image, fname_seg], colormaps=['gray', 'red'], opacities=['', '0.7'])


if __name__ == "__main__":
    main()
