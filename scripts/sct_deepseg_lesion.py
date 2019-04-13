#!/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Function to segment the multiple sclerosis lesions using convolutional neural networks
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Charley Gros
# Modified: 2018-06-11
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import print_function, absolute_import, division

import os
import sys
import numpy as np

from msct_parser import Parser
import sct_utils as sct
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.deepseg_lesion.core import deep_segmentation_MSlesion


def get_parser():
    """Initialize the parser."""
    parser = Parser(__file__)
    parser.usage.set_description("""MS lesion Segmentation using convolutional networks. \n\nReference: C Gros, B De Leener, et al. Automatic segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional neural networks (2018). arxiv.org/abs/1805.06349""")

    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="input image.",
                      mandatory=True,
                      example="t1.nii.gz")
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast. \nt2: T2w scan with isotropic or anisotropic resolution. \nt2_ax: T2w scan with axial orientation and thick slices. \nt2s: T2*w scan with axial orientation and thick slices.",
                      mandatory=True,
                      example=['t2', 't2_ax', 't2s'])
    parser.add_option(name="-centerline",
                      type_value="multiple_choice",
                      description="Method used for extracting the centerline.\nsvm: automatic centerline detection, based on Support Vector Machine algorithm.\ncnn: automatic centerline detection, based on Convolutional Neural Network.\nviewer: semi-automatic centerline generation, based on manual selection of a few points using an interactive viewer, then approximation with NURBS.\nfile: use an existing centerline by specifying its filename with flag -file_centerline (e.g. -file_centerline t2_centerline_manual.nii.gz).\n",
                      mandatory=False,
                      example=['svm', 'cnn', 'viewer', 'file'],
                      default_value="svm")
    parser.add_option(name="-file_centerline",
                      type_value="image_nifti",
                      description="Input centerline file (to use with flag -centerline manual).",
                      mandatory=False,
                      example="t2_centerline_manual.nii.gz")
    parser.add_option(name="-brain",
                      type_value="multiple_choice",
                      description="indicate if the input image is expected to contain brain sections:\n1: contains brain section\n0: no brain section.\nTo indicate this parameter could speed the segmentation process. Note that this flag is only effective with -centerline cnn.",
                      mandatory=False,
                      example=["0", "1"],
                      default_value="1")
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
                      description="1: display on (default), 0: display off, 2: extended",
                      mandatory=False,
                      example=["0", "1", "2"],
                      default_value="1")
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

    fname_image = arguments['-i']
    contrast_type = arguments['-c']

    ctr_algo = arguments["-centerline"]

    brain_bool = bool(int(arguments["-brain"]))
    if "-brain" not in args and contrast_type in ['t2s', 't2_ax']:
        brain_bool = False

    if '-ofolder' not in args:
        output_folder = os.getcwd()
    else:
        output_folder = arguments["-ofolder"]

    if ctr_algo == 'file' and "-file_centerline" not in args:
        sct.printv('Please use the flag -file_centerline to indicate the centerline filename.', 1, 'error')
        sys.exit(1)
    
    if "-file_centerline" in args:
        manual_centerline_fname = arguments["-file_centerline"]
        ctr_algo = 'file'
    else:
        manual_centerline_fname = None

    remove_temp_files = int(arguments['-r'])

    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    algo_config_stg = '\nMethod:'
    algo_config_stg += '\n\tCenterline algorithm: ' + str(ctr_algo)
    algo_config_stg += '\n\tAssumes brain section included in the image: ' + str(brain_bool) + '\n'
    sct.printv(algo_config_stg)

    im_image = Image(fname_image)
    im_seg, im_labels_viewer, im_ctr = deep_segmentation_MSlesion(im_image, contrast_type, ctr_algo=ctr_algo, ctr_file=manual_centerline_fname,
                                        brain_bool=brain_bool, remove_temp_files=remove_temp_files, verbose=verbose)

    # Save segmentation
    fname_seg = os.path.abspath(os.path.join(output_folder, sct.extract_fname(fname_image)[1] + '_lesionseg' +
                                             sct.extract_fname(fname_image)[2]))
    im_seg.save(fname_seg)

    if ctr_algo == 'viewer':
        # Save labels
        fname_labels = os.path.abspath(os.path.join(output_folder, sct.extract_fname(fname_image)[1] + '_labels-centerline' +
                                               sct.extract_fname(fname_image)[2]))
        im_labels_viewer.save(fname_labels)

    if verbose == 2:
        # Save ctr
        fname_ctr = os.path.abspath(os.path.join(output_folder, sct.extract_fname(fname_image)[1] + '_centerline' +
                                               sct.extract_fname(fname_image)[2]))
        im_ctr.save(fname_ctr)

    sct.display_viewer_syntax([fname_image, fname_seg], colormaps=['gray', 'red'], opacities=['', '0.7'])


if __name__ == "__main__":
    main()
