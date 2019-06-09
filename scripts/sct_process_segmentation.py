#!/usr/bin/env python
#########################################################################################
#
# Perform various types of processing from the spinal cord segmentation (e.g. extract centerline, compute CSA, etc.).
# (extract_centerline) extract the spinal cord centerline from the segmentation. Output file is an image in the same
# space as the segmentation.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Touati, Gabriel Mangeat
# Modified: 2014-07-20 by jcohenadad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: the import of scipy.misc imsave was moved to the specific cases (orth and ellipse) in order to avoid issue #62. This has to be cleaned in the future.

from __future__ import absolute_import, division

import sys, os

import sct_utils as sct
from msct_parser import Parser
from spinalcordtoolbox import process_seg
from spinalcordtoolbox.aggregate_slicewise import aggregate_per_slice_or_level, save_as_csv, func_wa, func_std, \
    _merge_dict
from spinalcordtoolbox.utils import parse_num_list
from spinalcordtoolbox.reports.qc import generate_qc


# TODO: Move this class somewhere else
class Param:
    def __init__(self):
        self.debug = 0
        self.verbose = 1  # verbose
        self.remove_temp_files = 1
        self.slices = ''
        self.window_length = 50  # for smooth_centerline @sct_straighten_spinalcord
        self.algo_fitting = 'bspline'  # polyfit, bspline
        self.perslice = None
        self.perlevel = None


def get_parser():
    """
    :return: Returns the parser with the command line documentation contained in it.
    """
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description(
        """Compute the following morphometric measures based on the spinal cord segmentation:
- area [mm^2]: Cross-sectional area, measured by counting pixels in each slice. Partial volume can be accounted for by inputing a mask comprising values within [0,1].
- angle_AP, angle_RL: Estimated angle between the cord centerline and the axial slice. This angle is used to correct for morphometric information.
- diameter_AP, diameter_RL: Finds the major and minor axes of the cord and measure their length
- eccentricity: Eccentricity of the ellipse that has the same second-moments as the spinal cord. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
- orientation: angle (in degrees) between the AP axis of the spinal cord and the AP axis of the image
- solidity: CSA(spinal_cord) / CSA_convex(spinal_cord). If perfect ellipse, it should be one. This metric is interesting for detecting non-convex shape (e.g., in case of strong compression)
""")
    parser.add_option(name='-i',
                      type_value='image_nifti',
                      description='Mask to compute morphometrics from. Could be binary or weighted. E.g., spinal cord segmentation.',
                      mandatory=True,
                      example='seg.nii.gz')
    parser.usage.addSection('Optional Arguments')
    parser.add_option(name='-o',
                      type_value='file_output',
                      description="Output file name (add extension). Default: csa.csv.",
                      mandatory=False)
    parser.add_option(name='-append',
                      type_value='int',
                      description='Append results as a new line in the output csv file instead of overwriting it.',
                      mandatory=False,
                      default_value=0)
    parser.add_option(name='-z',
                      type_value='str',
                      description='Slice range to compute the metrics across (requires \"-p csa\").',
                      mandatory=False,
                      example='5:23')
    parser.add_option(name='-perslice',
                      type_value='int',
                      description='Set to 1 to output one metric per slice instead of a single output metric.'
                                  'Please note that when methods ml or map is used, outputing a single '
                                  'metric per slice and then averaging them all is not the same as outputting a single'
                                  'metric at once across all slices.',
                      mandatory=False,
                      default_value=Param().perslice)
    parser.add_option(name='-vert',
                      type_value='str',
                      description='Vertebral levels to compute the metrics across. Example: 2:9 for C2 to T2.',
                      mandatory=False,
                      example='2:9')
    parser.add_option(name='-vertfile',
                      type_value='str',
                      description='Vertebral labeling file. Only use with flag -vert',
                      default_value='./label/template/PAM50_levels.nii.gz',
                      mandatory=False)
    parser.add_option(name='-perlevel',
                      type_value='int',
                      description='Set to 1 to output one metric per vertebral level instead of a single '
                                  'output metric. This flag needs to be used with flag -vert.',
                      mandatory=False,
                      default_value=Param().perlevel)
    parser.add_option(name='-r',
                      type_value='multiple_choice',
                      description='Removes temporary folder used for the algorithm at the end of execution',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name='-angle-corr',
                      type_value='multiple_choice',
                      description='Angle correction for computing morphometric measures. When angle '
                                  'correction is used, the cord within the slice is stretched/expanded by a factor '
                                  'corresponding to the cosine of the angle between the centerline and the axial plane.'
                                  ' If the cord is already quasi-orthogonal to the slab, you can set -angle-corr to 0.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=None)
    parser.add_option(name='-qc-dataset',
                      type_value='str',
                      description='If provided, this string will be mentioned in the QC report as the dataset the process was run on',
                      )
    parser.add_option(name='-qc-subject',
                      type_value='str',
                      description='If provided, this string will be mentioned in the QC report as the subject the process was run on',
                      )
    parser.add_option(name='-v',
                      type_value='multiple_choice',
                      description='1: display on, 0: display off (default)',
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')
    parser.add_option(name='-h',
                      type_value=None,
                      description='display this help',
                      mandatory=False)

    return parser


def main(args):
    parser = get_parser()
    arguments = parser.parse(args)
    param = Param()

    # Initialization
    slices = param.slices
    group_funcs = (('MEAN', func_wa), ('STD', func_std))  # functions to perform when aggregating metrics along S-I

    fname_segmentation = sct.get_absolute_path(arguments['-i'])
    fname_vert_levels = ''
    if '-o' in arguments:
        file_out = os.path.abspath(arguments['-o'])
    else:
        file_out = ''
    if '-append' in arguments:
        append = int(arguments['-append'])
    else:
        append = 0
    if '-vert' in arguments:
        vert_levels = arguments['-vert']
    else:
        vert_levels = ''
    if '-r' in arguments:
        remove_temp_files = arguments['-r']
    if '-vertfile' in arguments:
        fname_vert_levels = arguments['-vertfile']
    if '-perlevel' in arguments:
        perlevel = arguments['-perlevel']
    else:
        perlevel = Param().perlevel
    if '-z' in arguments:
        slices = arguments['-z']
    if '-perslice' in arguments:
        perslice = arguments['-perslice']
    else:
        perslice = Param().perslice
    if '-angle-corr' in arguments:
        if arguments['-angle-corr'] == '1':
            angle_correction = True
        elif arguments['-angle-corr'] == '0':
            angle_correction = False
    path_qc = arguments.get("-qc", None)
    qc_dataset = arguments.get("-qc-dataset", None)
    qc_subject = arguments.get("-qc-subject", None)

    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    # update fields
    metrics_agg = {}
    if not file_out:
        file_out = 'csa.csv'

    metrics = process_seg.compute_shape(fname_segmentation,
                                        algo_fitting='bspline',
                                        angle_correction=angle_correction,
                                        verbose=verbose)
    for key in metrics:
        metrics_agg[key] = aggregate_per_slice_or_level(metrics[key], slices=parse_num_list(slices),
                                                        levels=parse_num_list(vert_levels), perslice=perslice,
                                                        perlevel=perlevel, vert_level=fname_vert_levels,
                                                        group_funcs=group_funcs)
    metrics_agg_merged = _merge_dict(metrics_agg)
    save_as_csv(metrics_agg_merged, file_out, fname_in=fname_segmentation, append=append)

    # QC report (only show CSA for clarity)
    if path_qc is not None:
        generate_qc(fname_segmentation, args=args, path_qc=os.path.abspath(path_qc), dataset=qc_dataset,
                    subject=qc_subject, metric=metrics_agg_merged, process='sct_process_segmentation')

    sct.display_open(file_out)


if __name__ == "__main__":
    sct.init_sct()
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main(sys.argv[1:])
