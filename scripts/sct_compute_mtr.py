#!/usr/bin/env python
#########################################################################################
#
# Compute magnetization transfer ratio (MTR).
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-09-21
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import, division

import sys
import os
import argparse

import sct_utils as sct
from spinalcordtoolbox.utils import Metavar

# DEFAULT PARAMETERS


class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        # self.register = 1
        self.verbose = 1
        self.file_out = 'mtr'


# main
#=======================================================================================================================
def main():
    import numpy as np
    import spinalcordtoolbox.image as msct_image

    parser = get_parser()
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    # Initialization
    fname_mt0 = ''
    fname_mt1 = ''
    fname_mtr = ''
    # register = param.register
    # remove_temp_files = param.remove_temp_files
    # verbose = param.verbose

    # Check input parameters
    fname_mt0 = args.mt0
    fname_mt1 = args.mt1
    fname_mtr = args.o
    verbose = args.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    # compute MTR
    sct.printv('\nCompute MTR...', verbose)
    nii_mt1 = msct_image.Image(fname_mt1)
    data_mt1 = nii_mt1.data
    data_mt0 = msct_image.Image(fname_mt0).data
    data_mtr = 100 * (data_mt0 - data_mt1) / data_mt0
    # save MTR file
    nii_mtr = nii_mt1
    nii_mtr.data = data_mtr
    nii_mtr.save(fname_mtr)
    # sct.run(fsloutput+'fslmaths -dt double mt0.nii -sub mt1.nii -mul 100 -div mt0.nii -thr 0 -uthr 100 fname_mtr', verbose)

    sct.display_viewer_syntax([fname_mt0, fname_mt1, fname_mtr])


# ==========================================================================================
def get_parser():
    parser = argparse.ArgumentParser(
        description='Compute magnetization transfer ratio (MTR). Output is given in percentage.',
        add_help=None,
        prog=os.path.basename(__file__).strip(".py"))
    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        '-mt0',
        help='Image without MT pulse (MT0)',
        metavar=Metavar.float,
        required=False)
    mandatoryArguments.add_argument(
        '-mt1',
        help='Image with MT pulse (MT1)',
        metavar=Metavar.float,
        required=False)
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit")
    optional.add_argument(
        '-v',
        type=int,
        choices=(0, 1, 2),
        help='Verbose: 0 = nothing, 1 = classic, 2 = expended',
        default=1)
    optional.add_argument(
        '-o',
        help='Path to output file.',
        metavar=Metavar.str,
        default=os.path.join('.','mtr.nii.gz'))

    return parser


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # parse arguments
    # initialize parameters
    param = Param()
    # param_default = Param()
    # call main function
    main()
