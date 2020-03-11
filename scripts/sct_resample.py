#!/usr/bin/env python
#########################################################################################
#
# Resample data using nibabel.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Sara Dupont
# 
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: add possiblity to resample to destination image

from __future__ import division, absolute_import

import os
import sys


import sct_utils as sct
from msct_parser import Parser
import spinalcordtoolbox.resampling

# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.fname_data = ''
        self.fname_out = ''
        self.new_size = ''
        self.new_size_type = ''
        self.interpolation = 'linear'
        self.ref = None
        self.x_to_order = {'nn': 0, 'linear': 1, 'spline': 2}
        self.mode = 'reflect'  # How to fill the points outside the boundaries of the input, possible options: constant, nearest, reflect or wrap
        # constant put the superior edges to 0, wrap does something weird with the superior edges, nearest and reflect are fine
        self.file_suffix = '_resampled'  # output suffix
        self.verbose = 1

# initialize parameters
param = Param()


def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('Anisotropic resampling of 3D or 4D data.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image to segment. Can be 3D or 4D. (Cannot be 2D)",
                      mandatory=True,
                      example='dwi.nii.gz')
    parser.usage.addSection('TYPE OF THE NEW SIZE INPUT : with a factor of resampling, in mm or in number of voxels\n'
                            'Please choose only one of the 3 options.')
    parser.add_option(name="-f",
                      type_value="str",
                      description="Resampling factor in each dimensions (x,y,z). Separate with \"x\"\n"
                                  "For 2x upsampling, set to 2. For 2x downsampling set to 0.5",
                      mandatory=False,
                      example='0.5x0.5x1')
    parser.add_option(name="-mm",
                      type_value="str",
                      description="New resolution in mm. Separate dimension with \"x\"",
                      mandatory=False,
                      example='0.1x0.1x5')
    parser.add_option(name="-vox",
                      type_value="str",
                      description="Resampling size in number of voxels in each dimensions (x,y,z). Separate with \"x\"",
                      mandatory=False)
    parser.add_option(name="-ref",
                      type_value="str",
                      description="Reference image to resample input image to. Uses world coordinates.",
                      mandatory=False)
    parser.usage.addSection('MISC')
    parser.add_option(name="-x",
                      type_value='multiple_choice',
                      description='Interpolation method.',
                      mandatory=False,
                      default_value='linear',
                      example=['nn', 'linear', 'spline'])

    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Output file name",
                      mandatory=False,
                      example='dwi_resampled.nii.gz')
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="verbose: 0 = nothing, 1 = classic, 2 = expended.",
                      mandatory=False,
                      default_value=1,
                      example=['0', '1', '2'])
    return parser


def run_main():
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    param.fname_data = arguments["-i"]
    arg = 0
    if "-f" in arguments:
        param.new_size = arguments["-f"]
        param.new_size_type = 'factor'
        arg += 1
    elif "-mm" in arguments:
        param.new_size = arguments["-mm"]
        param.new_size_type = 'mm'
        arg += 1
    elif "-vox" in arguments:
        param.new_size = arguments["-vox"]
        param.new_size_type = 'vox'
        arg += 1
    elif "-ref" in arguments:
        param.ref = arguments["-ref"]
        arg += 1
    else:
        sct.printv(parser.error('ERROR: you need to specify one of those three arguments : -f, -mm or -vox'))

    if arg > 1:
        sct.printv(parser.error('ERROR: you need to specify ONLY one of those three arguments : -f, -mm or -vox'))

    if "-o" in arguments:
        param.fname_out = arguments["-o"]
    if "-x" in arguments:
        if len(arguments["-x"]) == 1:
            param.interpolation = int(arguments["-x"])
        else:
            param.interpolation = arguments["-x"]
    param.verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=param.verbose, update=True)  # Update log level

    spinalcordtoolbox.resampling.resample_file(param.fname_data, param.fname_out, param.new_size, param.new_size_type,
                                               param.interpolation, param.verbose, fname_ref=param.ref)


if __name__ == "__main__":
    sct.init_sct()
    run_main()
