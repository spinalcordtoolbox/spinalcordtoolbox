#!/usr/bin/env python

import os
import sys

import sct_utils as sct
from msct_parser import Parser
from spinalcordtoolbox.centerline import optic


def run_main():
    parser = Parser(__file__)
    parser.usage.set_description("""This program will use the OptiC method to detect the spinal cord centerline.""")

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

    parser.add_option(name="-init",
                      type_value="float",
                      description="axial slice where the propagation starts.",
                      mandatory=False)

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

    args = sys.argv[1:]
    arguments = parser.parse(args)

    # Input filename
    fname_input_data = arguments["-i"]
    fname_data = os.path.abspath(fname_input_data)

    # Contrast type
    contrast_type = arguments["-c"]

    # Init option
    init_option = None
    if "-init" in arguments:
        init_option = float(arguments["-init"])

    # Output folder
    if "-ofolder" in arguments:
        folder_output = sct.slash_at_the_end(arguments["-ofolder"], slash=1)
    else:
        folder_output = './'

    # Remove temporary files
    remove_temp_files = True
    if "-r" in arguments:
        remove_temp_files = bool(arguments["-r"])

    # Verbosity
    verbose = 0
    if "-v" in arguments:
        if arguments["-v"] is "1":
            verbose = 2

    # OptiC models
    path_script = os.path.dirname(__file__)
    path_sct = os.path.dirname(path_script)
    optic_models_path = os.path.join(path_sct,
                                   'data/optic_models',
                                   '{}_model'.format(contrast_type))

    # Execute OptiC binary
    optic_filename = optic.detect_centerline(fname_data, init_option, contrast_type, 
                                             optic_models_path, folder_output,
                                             remove_temp_files, verbose)

    sct.printv('\nDone! To view results, type:', verbose)
    sct.printv("fslview " + fname_input_data + " " + optic_filename + " -l Red -b 0,1 -t 0.7 &\n",
               verbose, 'info')

if __name__ == '__main__':
    run_main()
