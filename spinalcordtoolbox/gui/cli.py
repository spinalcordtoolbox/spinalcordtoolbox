#!/usr/bin/env python

#
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
#
# About the license: see the file LICENSE.TXT
# Notes on how to use classes in this script.
import os
import sys
from PyQt4 import QtGui

from scripts.msct_image import Image
from scripts.msct_parser import Parser
from scripts.sct_utils import printv
from spinalcordtoolbox.gui import base
from spinalcordtoolbox.gui.propseg import PropSeg, PropSegController
from spinalcordtoolbox.gui.sagittal import SagittalControler, SagittalDialog


def get_parser():
    parser = Parser('sct_segment_image')
    parser.usage.set_description('Manually annotate Anatomic Images (nifti files)')
    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="Image to annotate",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name='-mode',
                      type_value='multiple_choice',
                      description='Choice of dialog types.',
                      mandatory=True,
                      default_value='centerline',
                      example=['propseg', 'registertotemplate', 'labelvertebrae'])
    parser.add_option(name='-start-label',
                      type_value='int',
                      description='The first vertebrae of interest',
                      mandatory=False)
    parser.add_option(name='-end-label',
                      type_value='int',
                      description='The last vertebrae of interest',
                      mandatory=False)
    parser.add_option(name='-n',
                      type_value='int',
                      description='Maximum number of points to capture',
                      mandatory=False)
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Output the nifti image file with the manual segments",
                      mandatory=True,
                      example="t2_seg.nii.gz")
    parser.add_option(name='-labels',
                      type_value=[[','], 'int'],
                      description='List of vertebraes labels you are interested in capturing',
                      mandatory=False)

    return parser


def launch_propseg_dialog(input_file, output_file, params):
    controller = PropSegController(input_file, params, output_file)
    controller.reformat_image()

    app = QtGui.QApplication([])
    dialog_ = PropSeg(controller)
    dialog_.show()
    app.exec_()
    return controller


def launch_labelvertebrae_dialog(input_file, output_file, params):
    if not params.vertebraes:
        params.vertebraes = [3, ]
    controller = SagittalControler(input_file, params, output_file)
    controller.reformat_image()

    app = QtGui.QApplication(sys.argv)
    dialog = SagittalDialog(controller)
    dialog.show()
    app.exec_()
    return controller


def launch_registertotemplate_dialog(input_file, output_file, params):
    if not params.vertebraes:
        params.vertebraes = [3, 5]
    controller = SagittalControler(input_file, params, output_file)
    controller.reformat_image()

    app = QtGui.QApplication(sys.argv)
    dialog = SagittalDialog(controller)
    dialog.show()
    app.exec_()
    return controller


def segment_image_cli():
    parser = get_parser()
    args = sys.argv[1:]

    try:
        arguments = parser.parse(args)
    except SyntaxError as err:
        printv(err.message, type='error')

    launch_modes = {'centerline': launch_propseg_dialog,
                    'sagittal': launch_registertotemplate_dialog}

    mode = arguments['-mode']

    input_file_name = arguments['-i']
    output_file_name = arguments['-o']

    params = base.AnatomicalParams()
    params.input_file_name = input_file_name
    params.start_label = arguments.get('-start-label', params.start_vertebrae)
    params.end_label = arguments.get('-end-label', params.end_vertebrae)
    params.num_points = arguments.get('-n', params.num_points)
    params.vertebraes = arguments.get('-labels', params.vertebraes)
    input_file = Image(input_file_name)

    if os.path.exists(output_file_name):
        output_file = Image(output_file_name)
    else:
        output_file = input_file.copy()
        output_file.data *= 0
        output_file.setFileName(output_file_name)

    controller = launch_modes[mode](input_file, output_file, params)
    if controller.saved:
        controller.as_niftii()
        printv('Output file %s' % output_file_name, type='info')
    else:
        printv('Manual segmentation was aborted')


if __name__ == "__main__":
    sys.exit(segment_image_cli())
