#!/usr/bin/env python
#########################################################################################
#  This code permits to run the function sct_axial_rotation on multiple subjects
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2018 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Nicolas Pinon
#
# License: see the LICENSE.TXT
#########################################################################################

from __future__ import division, absolute_import

import sys, os, shutil

import sct_utils as sct

from msct_parser import Parser

import fnmatch

def get_parser():

    parser = Parser(__file__)
    parser.usage.set_description('Blablablabla')
    parser.add_option(name="-f",
                      type_value="folder",
                      description="path to folder to search for image",
                      mandatory=True,
                      example="path/to/data")
    parser.add_option(name="-i",
                      type_value="str",
                      description="File to search for in the folder",
                      mandatory=True,
                      example="t2.nii.gz")
    # parser.add_option(name="-iseg",
    #                   type_value="file",
    #                   description="Segmentation source",
    #                   mandatory=True,
    #                   example="src_seg.nii.gz")
    # parser.add_option(name="-dseg",
    #                   type_value="file",
    #                   description="Segmentation destination.",
    #                   mandatory=True,
    #                   example="dest_seg.nii.gz")
    parser.add_option(name="-o",
                      type_value="folder",
                      description="output folder",
                      mandatory=True,
                      example="path/to/output")
    # parser.add_option(name="-r",
    #                   type_value="multiple_choice",
    #                   description="""Remove temporary files.""",
    #                   mandatory=False,
    #                   default_value='1',
    #                   example=['0', '1'])
    # parser.add_option(name="-v",
    #                   type_value="multiple_choice",
    #                   description="""Verbose.""",
    #                   mandatory=False,
    #                   default_value='1',
    #                   example=['0', '1', '2'])

    return parser


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    parser = get_parser()
    arguments = parser.parse(args)
    fname_search = arguments['-i']
    path_data = arguments['-f']
    path_output = arguments['-o']

    # lets go find the path of the filenames of interest
    cwd = os.getcwd()
    os.chdir(path_output)

    list_file = []
    for root, dirnames, filenames in os.walk(path_data):
        for filename in fnmatch.filter(filenames, fname_search):
            list_file.append(os.path.join(root, filename))

    for file in list_file:
        from sct_axial_rotation import main as axrot
        sct.run("sct_propseg -i " + file + " -c t2", verbose=2)
        axrot(args=["-i", file, "-iseg", "t2_seg.nii.gz", "-f", "."])

    # TODO : check if manual segmentation present, if so, use it
    # back to cwd

    1+1


if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()