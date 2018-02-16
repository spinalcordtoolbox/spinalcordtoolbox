#!/usr/bin/env python
#########################################################################################
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# Modified:
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: currently it seems like cross_radius is given in pixel instead of mm

import os, sys
import getopt

import sys
import sct_utils as sct
import nibabel
import numpy as np

# DEFAULT PARAMETERS


class Param:
    # The constructor
    def __init__(self):
        self.debug               = 0


#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # Initialization
    fname_input = ''
    fname_segmentation = ''

    if param.debug:
        sct.printv( '\n*** WARNING: DEBUG MODE ON ***\n')
        path_sct_data = os.environ.get("SCT_TESTING_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__))), "testing_data")
        fname_input = ''
        fname_segmentation = os.path.join(path_sct_data, 't2', 't2_seg.nii.gz')
    else:
    # Check input param
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hi:t:')
        except getopt.GetoptError as err:
            sct.log.error(str(err))
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ('-i'):
                fname_input = arg
            elif opt in ('-t'):
                fname_segmentation = arg

    # display usage if a mandatory argument is not provided
    if fname_segmentation == '' or fname_input == '':
        usage()

    # check existence of input files
    sct.check_file_exist(fname_input)
    sct.check_file_exist(fname_segmentation)

    # read nifti input file
    img = nibabel.load(fname_input)
    # 3d array for each x y z voxel values for the input nifti image
    data = img.get_data()

    # read nifti input file
    img_seg = nibabel.load(fname_segmentation)
    # 3d array for each x y z voxel values for the input nifti image
    data_seg = img_seg.get_data()

    X, Y, Z = (data > 0).nonzero()
    status = 0
    for i in range(0, len(X)):
        if data_seg[X[i], Y[i], Z[i]] == 0:
            status = 1
            break;

    if status is not 0:
        sct.printv('ERROR: detected point is not in segmentation', 1, 'warning')
    else:
        sct.printv('OK: detected point is in segmentation')

    sys.exit(status)

#=======================================================================================================================
# usage
#=======================================================================================================================


def usage():
    print( 'USAGE: \n' \
        'This script check if the point contained in inputdata is in the spinal cord segmentation.\n'\
        '  isct_check_detection -i <inputdata> -t <segmentationdata>\n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        '  -i           input volume. Contains one point\n' \
        '  -t           segmentation volume.\n' \
        '\n'\
        'OPTIONAL ARGUMENTS\n' \
        '  -h           help. Show this message.\n' \
        '\n')

    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # initialize parameters
    param = Param()
    # call main function
    main()
