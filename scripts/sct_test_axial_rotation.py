#!/usr/bin/env python
#########################################################################################
#  This code permits to run the function sct_axial_rotation on multiple subjects
#  The user needs to input the folder to search for the files, the filename and the
#  segmentation filename
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
    parser.add_option(name="-iseg",
                      type_value="str",
                      description="Segmentation file to search for in the folder",
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
    fname_seg_search = arguments['-iseg']
    path_data = arguments['-f']
    path_output = arguments['-o']

    # lets go find the path of the filenames of interest
    cwd = os.getcwd()
    os.chdir(path_output)

    list_file = []
    list_seg = []
    sct.printv("Begin searching phase")
    for root, dirnames, filenames in os.walk(path_data):  # searching the given directory
        for filename in fnmatch.filter(filenames, fname_search):  # if file with given name found
            sct.printv("\nFile  " + os.path.join(root, filename) + " found for sct_axial_rotation")
            list_file.append(os.path.join(root, filename))  # add to the list
            filename_seg = fnmatch.filter(filenames, fname_seg_search)  # search for segmentation of this file
            if len(filename_seg) > 0:  # if segmentation found
                if len(filename_seg) > 1:
                    sct.printv("More than one segmentation found for file : " + filename + " in : " + root)
                    raise NotImplementedError
                else:
                    sct.printv("Segmentation : " + filename_seg[0] + " found for file : " + os.path.join(root, filename))
                    list_seg.append(os.path.join(root, filename_seg[0]))  # add it to the list
            else:
                sct.printv("No segmentation found for file : " + os.path.join(root, filename))
                list_seg.append(None)

    sct.printv("Searching phase done")

    sct.printv("Begin computing phase")

    for k, file in enumerate(list_file):
        sct.printv("\nComputing " + str(k) + " th image :")
        from sct_axial_rotation import main as axrot
        if list_seg[k] is None:
            sct.printv("Segmenting " + file + " with sct_propseg")
            sct.run("sct_propseg -i " + file + " -c t2", verbose=0)
            file_seg = "t2_seg.nii.gz"
        else:
            file_seg = list_seg[k]
        sct.printv("\nRunning sct_axial_rotation for file : " + file)
        axrot(args=["-i", file, "-iseg", file_seg, "-onumber", str(k), "-ofolder", path_output])

    sct.printv("Computing phase done")


    # back to cwd
    os.chdir(cwd)
    1+1


if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
