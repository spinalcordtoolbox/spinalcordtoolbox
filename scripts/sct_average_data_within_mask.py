#!/usr/bin/env python
# ==========================================================================================
# Average data within mask. Compute a weighted average if mask is non-binary (values distributed between 0 and 1).
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
# Author: Julien Cohen-Adad
# Modified: 2013-11-10
#
#
# About the license: see the file LICENSE.TXT
# ==========================================================================================


# TODO: add test
# TODO: do a zmin zmax

import sys
import getopt
import os
from numpy import asarray, sqrt
import nibabel
from sct_utils import printv
import sct_utils as sct
from msct_parser import Parser


# PARAMETERS
debugging           = 0  # automatic file names for debugging


# MAIN
# ==========================================================================================
def main():

    # Initialization
    fname_src = ''
    fname_mask = ''
    tmask = ''
    zmask = ''
    verbose = 1

    # Check input parameters
    if debugging:
        fname_src  = '/Users/julien/MRI/multisite_DTI/20131011_becky/2013-11-09/input/b0_mean_reg2template.nii'
        fname_mask = '/Users/julien/matlab/toolbox/spinalcordtoolbox_dev/data/atlas/WMtracts.nii'
        tmask = '0'
        zmask = '445'
    else:
        parser = get_parser()
        arguments = parser.parse(sys.argv[1:])

        fname_src = arguments['-i']
        fname_mask = arguments['-m']

        if '-nvol' in arguments:
            tmask = arguments['-nvol']
        if '-z' in arguments:
            zmask = arguments['-z']
        if '-v' in arguments:
            verbose = int(arguments['-v'])

    # sct.printv(arguments)
    if verbose:
        sct.printv('\nCheck input parameters...')
        sct.printv('.. Image:    ' + fname_src)
        sct.printv('.. Mask:     ' + fname_mask)
        sct.printv('.. tmask:    ' + str(tmask))
        sct.printv('.. zmask:    ' + str(zmask))

    # Extract path, file and extension
    #path_src, file_src, ext_src = extract_fname(fname_src)
    #path_mask, file_mask, ext_mask = extract_fname(fname_mask)

    weighted_average, weighted_std = average_within_mask(fname_src, fname_mask, tmask, zmask, verbose)

    return weighted_average


def average_within_mask(fname_src, fname_mask, tmask='', zmask='', verbose=1):
    """
    Average data within mask
    :param fname_src:
    :param fname_mask:
    :param tmask:
    :param zmask:
    :param verbose:
    :return: [mean, std]
    """
    # Quantify image within mask
    header_src = nibabel.load(fname_src)
    header_mask = nibabel.load(fname_mask)

    data_src = header_src.get_data()
    #data_mask = header_mask.get_data()[:,:,:,int(tmask)]

    # check if mask is 4D
    if tmask == '':
        data_mask = header_mask.get_data()
    else:
        assert len(header_mask.get_data().shape) == 4, 'ERROR: mask is not 4D, cannot use option -nvol.'
        data_mask = header_mask.get_data()[:, :, :, tmask]

    # if user specified zmin and zmax, put rest of slices to 0
    if zmask != '':
        data_mask[:, :, :zmask] = 0
        data_mask[:, :, zmask + 1:] = 0

    # find indices of non-zero elements the mask
    ind_nonzero = data_mask.nonzero()

    # perform a weighted average for all nonzero indices from the mask
    data = []
    weight = []
    for i in range(0, len(ind_nonzero[0][:])):
        # retrieve coordinates from mask
        x, y, z = ind_nonzero[0][i], ind_nonzero[1][i], ind_nonzero[2][i]
        # get values in mask
        weight.append(data_mask[x, y, z])
        # get value in the image
        data.append(data_src[x, y, z])
    # compute weighted average
    data = asarray(data)
    weight = asarray(weight)
    n = len(data)
    # compute weighted_average
    weighted_average = sum(data * weight) / sum(weight)
    # compute weighted STD
    weighted_std = sqrt(sum(weight * (data - weighted_average)**2) / ((n / (n - 1)) * sum(weight)))

    # sct.printv(result)
    printv('\n' + str(weighted_average) + ' +/- ' + str(weighted_std), verbose)

    return weighted_average, weighted_std

    # Display created files


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Average data within mask. Compute a weighted average if mask is non-binary (values distributed between 0 and 1).')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image to extract values from",
                      mandatory=True,
                      example='t2.nii.gz')
    parser.add_option(name="-m",
                      type_value="file",
                      description="Binary or weighted mask (values between 0 and 1).",
                      mandatory=True,
                      example='t2_seg.nii.gz')
    parser.add_option(name="-nvol",
                      type_value="int",
                      description="Volume number (if mask is 4D).",
                      mandatory=False,
                      example='2')
    parser.add_option(name="-t",
                      type_value=None,
                      description="Volume number (if mask is 4D).",
                      deprecated_by="-nvol",
                      mandatory=False)
    parser.add_option(name="-z",
                      type_value="int",
                      description="Slice number to compute average on (other slices will not be considered).",
                      mandatory=False,
                      example='5')
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')
    return parser

# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    main()
