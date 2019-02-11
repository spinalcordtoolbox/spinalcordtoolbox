#!/usr/bin/env python

from __future__ import division, absolute_import

import os
import sys

import numpy as np

import sct_utils as sct
from msct_parser import Parser
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import get_centerline, ParamCenterline, _call_viewer_centerline


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description("""This function extracts the spinal cord centerline. Two methods are 
    available: OptiC (automatic) and Viewer (manual). This function outputs (i) a NIFTI file with labels corresponding
    to the discrete centerline, and (ii) a csv file containing the float (more precise) coordinates of the centerline
    in the RPI orientation. \n\nReference: C Gros, B De Leener, et al. Automatic spinal cord 
    localization, robust to MRI contrast using global curve optimization (2017). doi.org/10.1016/j.media.2017.12.001""")

    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="Input image.",
                      mandatory=True,
                      example="t1.nii.gz")
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="Type of image contrast. Only with method=optic.",
                      mandatory=False,
                      example=['t1', 't2', 't2s', 'dwi'])
    parser.add_option(name="-method",
                      type_value="multiple_choice",
                      description="Method used for extracting the centerline.\n"
                                  "optic: automatic spinal cord detection method\n"
                                  "viewer: manually selected a few points, followed by interpolation with polynoms.",
                      mandatory=False,
                      example=['optic', 'viewer'],
                      default_value='optic')
    parser.add_option(name="-o",
                      type_value='file_output',
                      description='File name (without extension) for the centerline output files. By default, output'
                                  'file will be the input with suffix "_centerline"',
                      mandatory=False,
                      example="centerline_optic")
    parser.add_option(name="-gap",
                      type_value="float",
                      description="Gap in mm between manually selected points. Only with method=viewer.",
                      mandatory=False,
                      default_value='20.0')
    parser.add_option(name="-igt",
                      type_value="image_nifti",
                      description="File name of ground-truth centerline or segmentation (binary nifti).",
                      mandatory=False)
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="1: display on, 0: display off (default)",
                      mandatory=False,
                      example=["0", "1", "2"],
                      default_value="1")
    return parser


def run_main():
    sct.init_sct()
    parser = get_parser()
    args = sys.argv[1:]
    arguments = parser.parse(args)

    # Input filename
    fname_input_data = arguments["-i"]
    fname_data = os.path.abspath(fname_input_data)

    # Method used
    method = 'optic'
    if "-method" in arguments:
        method = arguments["-method"]

    # Contrast type
    contrast_type = ''
    if "-c" in arguments:
        contrast_type = arguments["-c"]
    if method == 'optic' and not contrast_type:
        # Contrast must be
        error = 'ERROR: -c is a mandatory argument when using Optic method.'
        sct.printv(error, type='error')
        return

    # Ga between slices
    interslice_gap = 10.0
    if "-gap" in arguments:
        interslice_gap = float(arguments["-gap"])

    # Output folder
    if "-o" in arguments:
        file_output = arguments["-o"]
    else:
        path_data, file_data, ext_data = sct.extract_fname(fname_data)
        file_output = os.path.join(path_data, file_data + '_centerline')

    # Verbosity
    verbose = 0
    if "-v" in arguments:
        verbose = int(arguments["-v"])

    if method == 'viewer':
        im_labels = _call_viewer_centerline(Image(fname_data), interslice_gap=interslice_gap)
        im_centerline, arr_centerline, _ = \
            get_centerline(im_labels, algo_fitting='polyfit', param=ParamCenterline(degree=3), minmax=True,
                           verbose=verbose)
    else:
        im_centerline, arr_centerline, _ = \
            get_centerline(Image(fname_data), algo_fitting='optic', param=ParamCenterline(contrast=contrast_type),
                           minmax=True, verbose=verbose)

    # save centerline as nifti (discrete) and csv (continuous) files
    im_centerline.save(file_output + '.nii.gz')
    np.savetxt(file_output + '.csv', arr_centerline.transpose(), delimiter=",")

    sct.display_viewer_syntax([fname_input_data, file_output+'.nii.gz'], colormaps=['gray', 'red'], opacities=['', '1'])


if __name__ == '__main__':
    run_main()
