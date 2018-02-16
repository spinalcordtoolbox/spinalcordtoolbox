#!/usr/bin/env python
#########################################################################################
#
# Resample data using nipy.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Sara Dupont
# 
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys


import sct_utils as sct
from msct_parser import Parser
import spinalcordtoolbox.resample.nipy_resample

# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.fname_data = ''
        self.fname_out = ''
        self.new_size = ''
        self.new_size_type = ''
        self.interpolation = 'linear'
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
                      # example='50x50x20')
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
    # Parameters for debug mode
    if param.debug:
        sct.printv('\n*** WARNING: DEBUG MODE ON ***\n')
        # get path of the testing data
        path_sct_data = os.environ.get("SCT_TESTING_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__))), "testing_data")
        param.fname_data = os.path.join(path_sct_data, "fmri", "fmri.nii.gz")
        param.new_size = '2'  # '0.5x0.5x1'
        param.remove_tmp_files = 0
        param.verbose = 1
    else:
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
        else:
            sct.printv(parser.usage.generate(error='ERROR: you need to specify one of those three arguments : -f, -mm or -vox'))

        if arg > 1:
            sct.printv(parser.usage.generate(error='ERROR: you need to specify ONLY one of those three arguments : -f, -mm or -vox'))

        if "-o" in arguments:
            param.fname_out = arguments["-o"]
        if "-x" in arguments:
            if len(arguments["-x"]) == 1:
                param.interpolation = int(arguments["-x"])
            else:
                param.interpolation = arguments["-x"]
        if "-v" in arguments:
            param.verbose = int(arguments["-v"])

    spinalcordtoolbox.resample.nipy_resample.resample_file(param.fname_data,
        param.fname_out, param.new_size, param.new_size_type,
        param.interpolation, param.verbose)

if __name__ == '__main__':
    sct.start_stream_logger()
    run_main()

