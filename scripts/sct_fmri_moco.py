#!/usr/bin/env python
#########################################################################################
#
# Motion correction of fMRI data.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys

from spinalcordtoolbox.moco import ParamMoco, moco_wrapper

import sct_utils as sct
from msct_parser import Parser


def get_parser():
    # parser initialisation
    parser = Parser(__file__)

    # initialize parameters
    # TODO: create a class ParamFmriMoco which inheritates from ParamMoco
    param_default = ParamMoco(group_size=1, metric='MeanSquares', smooth='0')

    parser.usage.set_description("""Motion correction of fMRI data. Some robust features include:
  - group-wise (-g)
  - slice-wise regularized along z using polynomial function (-p)
    For more info about the method, type: isct_antsSliceRegularizedRegistration
  - masking (-m)
  - iterative averaging of target volume
The outputs of the motion correction process are:
  - the motion-corrected fMRI volumes
  - the time average of the corrected fMRI volumes
  - a time-series with 1 voxel in the XY plane, for the X and Y motion direction (two separate files), as required for FSL analysis.
  - a TSV file with the slice-wise average of the motion correction for XY (one file), that can be used for Quality Control.""")
    parser.add_option(name='-i',
                      type_value='image_nifti',
                      description='4D data',
                      mandatory=True,
                      example='fmri.nii.gz')
    parser.add_option(name='-g',
                      type_value='int',
                      description='Group nvols successive fMRI volumes for more robustness.',
                      mandatory=False,
                      default_value=param_default.group_size)
    parser.add_option(name='-m',
                      type_value='image_nifti',
                      description='Binary mask to limit voxels considered by the registration metric.',
                      mandatory=False)
    parser.add_option(name='-param',
                      type_value=[[','], 'str'],
                      description="Advanced parameters. Assign value with \"=\"; Separate arguments with \",\"\n"
                                  "poly [int]: Degree of polynomial function used for regularization along Z. For no regularization set to 0. Default=" + param_default.poly + ".\n"
                                  "smooth [mm]: Smoothing kernel. Default=" + param_default.smooth + ".\n"
                                  "iter [int]: Number of iterations. Default=" + param_default.iter + ".\n"
                                  "metric {MI, MeanSquares, CC}: Metric used for registration. Default=" + param_default.metric + ".\n"
                                  "gradStep [float]: Searching step used by registration algorithm. The higher the more deformation allowed. Default=" + param_default.gradStep + ".\n"
                                  "sampling [None or 0-1]: Sampling rate used for registration metric. Default=" + param_default.sampling + ".\n"
                                  "numTarget [int]: Target volume or group (starting with 0). Default=" + param_default.num_target + ".\n"
                                  "iterAvg [int]: Iterative averaging: Target volume is a weighted average of the previously-registered volumes. Default=" + str(param_default.iterAvg) + ".\n",
                      mandatory=False)
    parser.add_option(name='-ofolder',
                      type_value='folder_creation',
                      description='Output path.',
                      mandatory=False,
                      default_value='./')
    parser.add_option(name="-x",
                      type_value="multiple_choice",
                      description="""Final interpolation.""",
                      mandatory=False,
                      default_value='linear',
                      example=['nn', 'linear', 'spline'])
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="""Remove temporary files.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1', '2'])

    return parser


def main():

    # initialization
    param = ParamMoco(group_size=1, metric='MeanSquares', smooth='0')

    # Fetch user arguments
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    param.fname_data = arguments['-i']
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
