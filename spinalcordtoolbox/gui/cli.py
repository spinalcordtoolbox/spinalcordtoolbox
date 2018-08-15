#!/usr/bin/env python

#
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
#
# About the license: see the file LICENSE.TXT
# Notes on how to use classes in this script.

from __future__ import absolute_import

import os
import sys

from ..image import Image

from scripts.msct_parser import Parser
from scripts.sct_utils import printv
from spinalcordtoolbox.gui import base
from spinalcordtoolbox.gui.centerline import launch_centerline_dialog
from spinalcordtoolbox.gui.sagittal import launch_sagittal_dialog


def get_parser():
    parser = Parser('sct_label_image')
    parser.usage.set_description('Manually create labels based on a reference nifti images.')
    parser.add_option(name='-i',
                      type_value='image_nifti',
                      description='Image to label',
                      mandatory=True,
                      example='t2.nii.gz')
    parser.add_option(name='-mode',
                      type_value='multiple_choice',
                      description='Choice of dialog types.',
                      mandatory=True,
                      default_value='centerline',
                      example=['centerline', 'sagittal'])
    parser.add_option(name='-n',
                      type_value='int',
                      description='Maximum number of labels to capture',
                      mandatory=False)
    parser.add_option(name='-o',
                      type_value='file_output',
                      description='Output image file the created labels',
                      mandatory=True,
                      example='t2_seg.nii.gz')
    parser.add_option(name='-labels',
                      type_value=[[','], 'int'],
                      description='List of labels to create, separated with ",". Example: 2,3,4,5',
                      mandatory=False)
    return parser


def segment_image_cli():
    parser = get_parser()
    args = sys.argv[1:]

    try:
        arguments = parser.parse(args)
    except SyntaxError as err:
        printv(str(err), type='error')

    launch_modes = {'centerline': launch_centerline_dialog,
                    'sagittal': launch_sagittal_dialog}

    mode = arguments['-mode']

    input_file_name = arguments['-i']
    output_file_name = arguments['-o']

    params = base.AnatomicalParams()
    params.input_file_name = input_file_name
    params.num_points = arguments.get('-n', params.num_points)
    params.vertebraes = arguments.get('-labels', params.vertebraes)
    input_file = Image(input_file_name)

    if os.path.exists:
        printv('The output file exists, the contents will be overwritten', type='warning')
    output_file = input_file.copy()
    output_file.data *= 0
    output_file.absolutepath = output_file_name

    controller = launch_modes[mode](input_file, output_file, params)
    if controller.saved:
        controller.as_niftii()
        printv('Output file %s' % output_file_name, type='info')
    else:
        printv('Aborted Manual labeling', type='warning')


if __name__ == "__main__":
    sys.exit(segment_image_cli())
