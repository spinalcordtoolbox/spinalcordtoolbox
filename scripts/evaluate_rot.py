#!/usr/bin/env python

# Script used to process one MRI image at a time (the image and its segmentation), made to be used with wrapper or alone

import sys, os
import sct_utils as sct
from msct_parser import Parser
from sct_register_to_template import main as sct_register_to_template
from sct_label_vertebrae import main as sct_label_vertebrae
from sct_apply_transfo import main as sct_apply_transfo
from sct_label_utils import main as sct_labels_utils
from sct_maths import main as sct_maths
from nicolas_scripts.functions_sym_rot import *
from spinalcordtoolbox.reports.qc import generate_qc
import csv

def get_parser():

    parser = Parser(__file__)
    parser.usage.set_description('Script to process a MRI image with its segmentation, blablabla what does this script do')
    parser.add_option(name="-i",
                      type_value="file",
                      description="File input",
                      mandatory=True,
                      example="/home/data/cool_T2_MRI.nii.gz")
    parser.add_option(name="-iseg",
                      type_value="file",
                      description="Segmentation of the input file",
                      mandatory=True,
                      example="/home/data/cool_T2_MRI_seg_manual.nii.gz")
    parser.add_option(name="-o",
                      type_value="folder",
                      description="output folder for test results",
                      mandatory=False,
                      example="path/to/output/folder")
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      mandatory=False)

    return parser


def main(args=None):

    #TODO define filenames

    # Parser :
    if not args:
        args = sys.argv[1:]
    parser = get_parser()
    arguments = parser.parse(args)
    fname_image = arguments['-i']
    fname_seg = arguments['-iseg']
    output_dir = arguments['-o']
    if '-qc' in arguments:
        path_qc = arguments['-qc']

    # creating output dir if it does not exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # creating qc dir if it does not exist
    if not os.path.isdir(path_qc):
        os.mkdir(path_qc)

    sct.printv("        Python processing file : " + fname_image + " with seg : " + fname_seg)

if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
