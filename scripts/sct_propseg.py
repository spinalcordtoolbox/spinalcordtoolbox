#!/usr/bin/env python
#########################################################################################
#
# Parser for PropSeg binary.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Modified: 2015-03-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from msct_parser import Parser
import sys
import sct_utils as sct

if __name__ == "__main__":
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Utility function for labels.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="input image.",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name="-o",
                      type_value="folder_creation",
                      description="output folder.",
                      mandatory=False,
                      example="My_Output_Folder/",
                      default_value="./")
    parser.add_option(name="-t",
                      type_value="multiple_choice",
                      description="type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark",
                      mandatory=True,
                      example=['t1','t2'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="1: display on, 0: display off (default)",
                      mandatory=False,
                      example=["0","1"],
                      default_value="0")
    arguments = parser.parse(sys.argv[1:])

    input_filename = arguments["-i"]
    contrast_type = arguments["-t"]

    # Building the command
    cmd = "isct_propseg" + " -i " + input_filename + " -t " + contrast_type

    folder_output = "./"
    if "-o" in arguments:
        folder_output = arguments["-o"]
        cmd += " -o " + folder_output

    if "-v" in arguments:
        verbose = arguments["-v"]
        if verbose is "1":
            cmd += " -verbose"

    sct.runProcess(cmd, 1)

    sct.printv("Done!",1,"normal")
    sct.printv("Type the following command in the terminal to see the results:", 1, "normal")

    # extracting output filename
    path_fname, file_fname, ext_fname = sct.extract_fname(input_filename)
    output_filename = file_fname+"_seg"+ext_fname
    sct.printv("fslview "+input_filename+" "+folder_output+output_filename+" -l Red -b 0,1 -t 0.7")