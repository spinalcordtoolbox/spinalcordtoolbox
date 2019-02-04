
import os
import numpy as np
from spinalcordtoolbox.image import Image
import matplotlib.pyplot as plt
from msct_parser import Parser
import sct_utils as sct
import sys, os, shutil
from functions_sym_rot import *
import fnmatch
import scipy
from extract_slice import main as extract_slice
from random import randint
from sct_deepseg_sc import main as sct_deepseg_sc

def get_parser():

    parser = Parser(__file__)
    parser.usage.set_description('Blablablabla')
    parser.add_option(name="-i",
                      type_value="folder",
                      description="Input folder with data to create test with",
                      mandatory=True,
                      example="/home/data")

    parser.add_option(name="-number",  # TODO find better name
                      type_value="str",
                      description="number of 2D axial slices per image",
                      mandatory=True,
                      example="4")
    parser.add_option(name="-o",
                      type_value="folder",
                      description="output folder for data set",
                      mandatory=False,
                      example="path/to/output/folder")

    return parser


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    parser = get_parser()
    arguments = parser.parse(args)
    input_folder = arguments['-i']
    no_of_slices = int(arguments['-number'])
    if '-o' in arguments:
        path_output = arguments['-o']
    else:
        path_output = os.getcwd()

    for root, dirnames, filenames in os.walk(input_folder):  # searching the given directory
        for filename in fnmatch.filter(filenames, "*.nii*"):  # if file with nii extension (.nii or .nii.gz) found
            if "seg" in filename or "dwi" in filename:
                continue  # do not consider it if it's a segmentation or dwi
            else:
                path_file = os.path.join(root, filename)
                if "T1w" in filename:
                    contrast = "t1"
                elif "T2w" in filename:
                    contrast = "t2"
                elif "T2s" in filename:
                    contrast = "t2s"
                else:
                    sct.printv("could not find contrast for file : " + filename)
                    continue

                sct_deepseg_sc(['-i', path_file, '-c', contrast, '-ofolder', root])
                nx, ny, nz = Image(path_file).change_orientation("RPI").data.shape
                slice_no = str(randint(0, nz-1))
                extract_slice(['-i', path_file, '-slice', slice_no, '-o', path_output])
                extract_slice(['-i', path_file.split(".nii")[0] + "_seg.nii" + path_file.split(".nii")[1],
                               '-slice', slice_no, '-o', path_output])



if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
