#!/usr/bin/env python
#########################################################################################
#
# Motion correction of dMRI data.
#
# Inspired by Xu et al. Neuroimage 2013.
#
# Details of the algorithm:
# - grouping of DW data only (every n volumes, default n=5)
# - average all b0
# - average DWI data within each group
# - average DWI of all groups
# - moco on DWI groups
# - moco on b=0, using target volume: last b=0
# - moco on all dMRI data
# _ generating b=0 mean and DWI mean after motion correction
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
# Modified: 2014-08-15
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: if -f, we only need two plots. Plot 1: X params with fitted spline, plot 2: Y param with fitted splines. Each plot will have all Z slices (with legend Z=0, Z=1, ...) and labels: y; translation (mm), xlabel: volume #. Plus add grid.


import sys
import os
from spinalcordtoolbox.moco import ParamMoco, moco_wrapper

import sct_utils as sct
from msct_parser import Parser


def get_parser():

    # initialize parameters
    param_default = ParamMoco(is_diffusion=True, group_size=3, metric='MI', smooth='1')

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description(
        '  Motion correction of dMRI data. Some of the features to improve robustness were proposed in Xu et al. (http://dx.doi.org/10.1016/j.neuroimage.2012.11.014) and include:\n'
        '- group-wise (-g)\n'
        '- slice-wise regularized along z using polynomial function (-param). For more info about the method, type: isct_antsSliceRegularizedRegistration\n'
        '- masking (-m)\n'
        '- iterative averaging of target volume\n')
    parser.add_option(name='-i',
                      type_value='file',
                      description='Diffusion data',
                      mandatory=True,
                      example='dmri.nii.gz')
    parser.add_option(name='-bvec',
                      type_value='file',
                      description='Bvecs file',
                      mandatory=True,
                      example='bvecs.nii.gz')
    parser.add_option(name='-bval',
                      type_value='file',
                      description='Bvals file',
                      mandatory=False,
                      example='bvals.nii.gz')
    parser.add_option(name='-bvalmin',
                      type_value='float',
                      description='B-value threshold (in s/mm2) below which data is considered as b=0.',
                      mandatory=False,
                      example='50')
    parser.add_option(name='-g',
                      type_value='int',
                      description='Group nvols successive dMRI volumes for more robustness.',
                      mandatory=False,
                      default_value=param_default.group_size,
                      example=['2'])
    parser.add_option(name='-m',
                      type_value='file',
                      description='Binary mask to limit voxels considered by the registration metric.',
                      mandatory=False,
                      example=['dmri_mask.nii.gz'])
    parser.add_option(name='-param',
                      type_value=[[','], 'str'],
                      description="Advanced parameters. Assign value with \"=\"; Separate arguments with \",\"\n"
                                  "poly [int]: Degree of polynomial function used for regularization along Z. For no regularization set to 0. Default=" + param_default.poly + ".\n"
                                                "smooth [mm]: Smoothing kernel. Default=" + param_default.smooth + ".\n"
                                                  "metric {MI, MeanSquares, CC}: Metric used for registration. Default=" + param_default.metric + ".\n"
                                                  "gradStep [float]: Searching step used by registration algorithm. The higher the more deformation allowed. Default=" + param_default.gradStep + ".\n"
                                                    "sample [None or 0-1]: Sampling rate used for registration metric. Default=" + param_default.sampling + ".\n",
                      mandatory=False)
    parser.add_option(name='-x',
                      type_value='multiple_choice',
                      description='Final Interpolation.',
                      mandatory=False,
                      default_value=param_default.interp,
                      example=['nn', 'linear', 'spline'])
    parser.add_option(name='-ofolder',
                      type_value='folder_creation',
                      description='Output folder',
                      mandatory=False,
                      default_value=param_default.path_out,
                      example='dmri_moco_results/')
    parser.usage.addSection('MISC')
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
    return parser


def main():

    # initialization
    param = ParamMoco(is_diffusion=True, group_size=3, metric='MI', smooth='1')

    # Fetch user arguments
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    param.fname_data = arguments['-i']
    param.fname_bvecs = os.path.abspath(arguments['-bvec'])
    if '-bval' in arguments:
        param.fname_bvals = os.path.abspath(arguments['-bval'])
    if '-bvalmin' in arguments:
        param.bval_min = arguments['-bvalmin']
    if '-g' in arguments:
        param.group_size = arguments['-g']
    if '-m' in arguments:
        param.fname_mask = arguments['-m']
    if '-param' in arguments:
        param.update(arguments['-param'])
    if '-x' in arguments:
        param.interp = arguments['-x']
    if '-ofolder' in arguments:
        param.path_out = arguments['-ofolder']
    if '-r' in arguments:
        param.remove_temp_files = int(arguments['-r'])
    param.verbose = int(arguments.get('-v'))

    # Update log level
    sct.init_sct(log_level=param.verbose, update=True)

    # run moco
    moco_wrapper(param)


if __name__ == "__main__":
    sct.init_sct()
    main()
