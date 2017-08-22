#!/usr/bin/env python

# Visualizer for MRI volumes
#
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
#


#
# About the license: see the file LICENSE.TXT
# Notes on how to use classes in this script.
# If you are interested into selecting manually some points in an image, you can use the following code.
import os
import sys

from msct_parser import Parser
from PyQt4 import QtGui
from scripts.msct_image import Image
from scripts.sct_utils import printv
from spinalcordtoolbox.gui import base
from spinalcordtoolbox.gui.groundtruth import GroundTruth, GroundTruthController
from spinalcordtoolbox.gui.labelvertebrae import LabelVertebrae, LabelVertebraeController
from spinalcordtoolbox.gui.propseg import PropSeg, PropSegController


def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('Manually annotate Anatomic Images (nifti files)')
    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="Image to annotate",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name='-mode',
                      type_value='multiple_choice',
                      description='Choice of dialog types.',
                      mandatory=False,
                      default_value='centerline',
                      example=['centerline', 'labelvertebrae', 'groundtruth'])
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
    parser.add_option(name='-param',
                      type_value=[[':'], 'str'],
                      description="""Parameters for visualization. Separate parameters with ":".
                     cmap: image colormap
                     interp: image interpolation.
                            Accepts: ['nearest' | 'bilinear' | 'bicubic' | 'spline16' |
                           'spline36' | 'hanning' | 'hamming' | 'hermite' | 'kaiser' |
                           'quadric' | 'catrom' | 'gaussian' | 'bessel' | 'mitchell' |
                           'sinc' | 'lanczos' | 'none']
                     vmin:
                     vmax:
                     vmean:
                     perc:""",
                      mandatory=False,
                      example=['cmap=red:vmin=0:vmax=1', 'cmap=grey'])
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Output the nifti image file with the manual segments",
                      mandatory=False,
                      example="t2_seg.nii.gz")
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="Verbose. 0: nothing. 1: basic. 2: extended.",
                      mandatory=False,
                      default_value="0",
                      example=['0', '1', '2'])

    return parser


def segment_image_cli():
    parser = get_parser()
    args = sys.argv[1:]
    arguments = parser.parse(args)

    modes = {'centerline': (PropSeg, PropSegController, '1. Select saggital slice -> 2. Select the Axial center of the spinalcord'),
             'labelvertebrae': (LabelVertebrae, LabelVertebraeController, '1. Select a label -> 2. Select a point in the sagittal plane'),
             'groundtruth': (GroundTruth, GroundTruthController, '')}

    mode = arguments['-mode']

    try:
        dialog, controller, init_message = modes[mode]
    except KeyError:
        printv('The mode is invalid: %s' % mode, type='error')

    input_file_name = arguments['-i']
    output_file_name = arguments['-o']

    params = base.AnatomicalParams()
    params.init_message = init_message
    params.start_label = arguments.get('-start-label', None)
    params.end_label = arguments.get('-end-label', None)
    params.num_points = arguments.get('-n', 0)
    input_file = Image(input_file_name)

    if os.path.exists(output_file_name):
        output_file = Image(output_file_name)
    else:
        output_file = Image(input_file_name)
        output_file.data *= 0
        output_file.file_name = output_file_name

    ctrl = controller(input_file, params, output_file)
    ctrl.align_image()

    app = QtGui.QApplication(sys.argv)
    dialog_ = dialog(ctrl)
    dialog_.show()
    app.exec_()
    ctrl.as_niftii(output_file_name)


if __name__ == "__main__":
    sys.exit(segment_image_cli())
