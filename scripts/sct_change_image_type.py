#!/usr/bin/env python
#########################################################################################
#
# Create, remove or display labels.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad
# Modified: 2014-10-29
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
import math


# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.fname_label_output = 'labels.nii.gz'
        self.labels = []
        self.verbose = 1


#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # Initialization
    fname_in = ''
    fname_out = ''
    type_output = 'int32'
    # get path of the toolbox
    path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))

    # Parameters for debug mode
    if param.debug:
        sct.log.debug('\n*** WARNING: DEBUG MODE ON ***\n')
        path_sct_data = os.environ.get("SCT_TESTING_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__))), "testing_data")
        fname_label = os.path.join(path_sct_data, 'mt', 'mt1.nii.gz')
        param.labels = '5,5,2,1:5,7,2,3'
        type_process = 'create'
        cross_radius = 5
        dilate = True
    else:
        # Check input param
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hi:o:c:r:t:l:dx:')
        except getopt.GetoptError as err:
            sct.log.error(str(err))
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ('-i'):
                fname_in = arg
            elif opt in ('-o'):
                fname_out = arg
            elif opt in ('-t'):
                type_output = arg

    # display usage if a mandatory argument is not provided
    if fname_in == '' or fname_out == '':
        sct.printv('\nERROR: All mandatory arguments are not provided. See usage (add -h).\n', 1, 'error')

    # check existence of input files
    sct.check_file_exist(fname_in)

    # read nifti input file
    img = nibabel.load(fname_in)
    # 3d array for each x y z voxel values for the input nifti image
    data = img.get_data()
    hdr = img.get_header()

    path_in, file_in, ext_in = sct.extract_fname(fname_in)
    path_output, file_output, ext_output = sct.extract_fname(fname_out)

    if type_output == 'int8':
        hdr.set_data_dtype(np.int8)
    elif type_output == 'int16':
        hdr.set_data_dtype(np.int16)
    elif type_output == 'int32':
        hdr.set_data_dtype(np.int32)
    elif type_output == 'int64':
        hdr.set_data_dtype(np.int64)
    elif type_output == 'uint8':
        hdr.set_data_dtype(np.uint8)
    elif type_output == 'uint16':
        hdr.set_data_dtype(np.uint16)
    elif type_output == 'uint32':
        hdr.set_data_dtype(np.uint32)
    elif type_output == 'uint64':
        hdr.set_data_dtype(np.uint64)
    elif type_output == 'float16':
        sct.printv('Error: voxel type (float16) not supported by nibabel (although it is supported by numpy)... See usage.', 1, 'error')
        # hdr.set_data_dtype(np.float16)
    elif type_output == 'float32':
        hdr.set_data_dtype(np.float32)
    elif type_output == 'float64':
        hdr.set_data_dtype(np.float64)
    else:
        sct.printv('Error: voxel type not supported... See usage.', 1, 'error')

    hdr.set_data_dtype(type_output)  # set imagetype to uint8, previous: int32.
    sct.printv('\nWrite NIFTI volumes...')
    # data.astype('int')
    img = nibabel.Nifti1Image(data, None, hdr)
    nibabel.save(img, 'tmp.' + file_output + '.nii.gz')
    sct.generate_output_file('tmp.' + file_output + '.nii.gz', file_output + ext_output)


#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print("""
{0}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Change the type of voxel in the image

USAGE
  {0} -i <data> -o <outputname> -t <type>

MANDATORY ARGUMENTS
  -i <data>         input image name
  -o <filename>     output image name

OPTIONAL ARGUMENTS
  -t <type>         type of output image. default: int32
                    Available:  int8, int16, int32, int64,
                                uint8, uint16, uint32, uint64,
                                float16, float32, float64
""".format(os.path.basename(__file__)))

    # exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()
