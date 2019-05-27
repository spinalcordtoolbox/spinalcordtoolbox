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

import sct_utils as sct
from msct_parser import Parser

# DEFAULT PARAMETERS


class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        # self.register = 1
        self.verbose = 1
        self.file_out = 'mtr'
        self.remove_temp_files = 1


# main
#=======================================================================================================================
def main(args=None):
    import numpy as np
    import spinalcordtoolbox.image as msct_image

    # Initialization
    fname_mt0 = ''
    fname_mt1 = ''
    fname_mtr = ''
    # register = param.register
    # remove_temp_files = param.remove_temp_files
    # verbose = param.verbose

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Check input parameters
    parser = get_parser()
    arguments = parser.parse(args)

    fname_mt0 = arguments['-mt0']
    fname_mt1 = arguments['-mt1']
    fname_mtr = arguments['-o']
    verbose = int(arguments.get('-v'))
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

    # go to output file directory
    os.chdir(os.path.dirname(fname_mtr))

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(os.path.join('.', fname_mtr), '.')

    sct.display_viewer_syntax([fname_mt0, fname_mt1, fname_mtr])


# ==========================================================================================
def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Compute magnetization transfer ratio (MTR). Output is given in percentage.')
    parser.add_option(name="-mt0",
                      type_value="file",
                      description="Image without MT pulse (MT0)",
                      mandatory=True,
                      example='mt0.nii.gz')
    parser.add_option(name="-i",
                      type_value=None,
                      description="Image without MT pulse (MT0)",
                      mandatory=False,
                      deprecated_by='-mt0')
    parser.add_option(name="-mt1",
                      type_value="file",
                      description="Image with MT pulse (MT1)",
                      mandatory=True,
                      example='mt1.nii.gz')
    parser.add_option(name="-j",
                      type_value=None,
                      description="Image with MT pulse (MT1)",
                      mandatory=False,
                      deprecated_by="-mt1")
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')
    parser.add_option(name="-o",
                      type_value="str",
                      description="Path to output file.",
                      mandatory=False,
                      example='Users/john/data/My_New_File.nii.gz',
                      default_value=os.path.join('.','mtr.nii.gz'))

    return parser


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # initialize parameters
    param = Param()
    # param_default = Param()
    # call main function
    main()
