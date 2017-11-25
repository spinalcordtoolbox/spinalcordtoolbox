import sys

import sct_utils as sct
from msct_parser import Parser

from spinalcordtoolbox.deepgmseg import deepgmseg


def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('Spinal Cord Gray Matter (GM) Segmentation. '
                                 'This method uses the technique described in '
                                 'the paper \'Spinal cord gray matter '
                                 'segmentation using deep dilated '
                                 'convolutions\', where Deep Learning '
                                 'techniques are employed to segment the GM.')

    parser.add_option(name="-i",
                      type_value="file",
                      description="Image filename to segment (3D volume).",
                      mandatory=True,
                      example='t2s.nii.gz')

    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Output segmentation file name.",
                      mandatory=True,
                      example='sc_gm_seg.nii.gz',
                      default_value='sc_gm_seg.nii.gz')

    parser.usage.addSection('\nMISC')

    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="Verbose: 0 = no verbosity, 1 = verbose",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')

    return parser


def run_main():
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    input_filename = arguments["-i"]
    output_filename = arguments["-o"]
    verbosity = arguments["-v"]

    deepgmseg.segment_file(input_filename, output_filename,
                           int(verbosity))


if __name__ == '__main__':
    sct.start_stream_logger()
    run_main()
