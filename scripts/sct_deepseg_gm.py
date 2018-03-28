#!/usr/bin/env python
# -*- coding: utf-8
# This command-line tool is the interface for the deepseg_gm API
# that implements the model for the Spinal Cord Gray Matter Segmentation.
#
# Reference paper:
#     Perone, C. S., Calabrese, E., & Cohen-Adad, J. (2017).
#     Spinal cord gray matter segmentation using deep dilated convolutions.
#     URL: https://arxiv.org/abs/1710.01269

import sys, os

import sct_utils as sct
from msct_parser import Parser

from spinalcordtoolbox.deepseg_gm import deepseg_gm


def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('Spinal Cord Gray Matter (GM) Segmentation using deep dilated convolutions. '
                                 'Reference: CS Perone, E Calabrese, J Cohen-Adad. Spinal cord gray matter segmentation using deep dilated convolutions (2017). arxiv.org/abs/1710.01269')

    parser.add_option(name="-i",
                      type_value="file",
                      description="Image filename to segment (3D volume). Contrast must be similar to T2*-weighted, i.e., WM dark, GM bright and CSF bright.",
                      mandatory=True,
                      example='t2s.nii.gz')

    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Output segmentation file name.",
                      mandatory=False,
                      example='sc_gm_seg.nii.gz')

    parser.usage.addSection('\nMISC')

    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=os.path.expanduser('~/qc_data'))
    parser.add_option(name='-noqc',
                      type_value=None,
                      description='Prevent the generation of the QC report',
                      mandatory=False)

    parser.add_option(name="-m",
                      type_value='multiple_choice',
                      description="Model to use (large or challenge)."
                                  "The model 'large' will be slower but "
                                  "will yield better results. The model 'challenge' was built using data from the following challenge: goo.gl/h4AVar.",
                      mandatory=False,
                      example=['large', 'challenge'],
                      default_value='large')

    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="Verbose: 0 = no verbosity, 1 = verbose.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')

    return parser


def run_main():
    deepseg_gm.check_backend()
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    input_filename = arguments["-i"]

    try:
        output_filename = arguments["-o"]
    except KeyError:
        output_filename = sct.add_suffix(input_filename, '_gmseg')

    verbose = arguments["-v"]
    model_name = arguments["-m"]

    out_fname = deepseg_gm.segment_file(input_filename, output_filename,
                                        model_name, int(verbose))


    if '-qc' in arguments:
        qc_path = os.path.abspath(arguments['-qc'])
    if '-noqc' in arguments:
        qc_path = None
    if qc_path is not None:
        sct.printv('\nSave quality control images (in %s)...' % qc_path, verbose, 'normal')

        import spinalcordtoolbox.reports.qc as qc
        import spinalcordtoolbox.reports.slice as qcslice

        qc_param = qc.Params(input_filename, "sct_deepseg_gm", sys.argv[1:], 'Axial', qc_path)
        report = qc.QcReport(qc_param, '')

        @qc.QcImage(report, 'none', [qc.QcImage.listed_seg], stretch_contrast=False)
        def test(qslice):
            return qslice.mosaic()

        from msct_image import Image
        im_org = Image(input_filename)
        im_seg = Image(out_fname)

        test(qcslice.Axial(im_org, im_seg))
        sct.printv('Sucessfully generated the QC results in %s' % qc_param.qc_results)
        sct.printv('Use the following command to see the results in a browser')
        sct.printv('sct_qc -folder %s' % qc_path, type='info')


    sct.display_viewer_syntax([input_filename, format(out_fname)],
                              colormaps=['gray', 'red'],
                              opacities=['1', '0.7'],
                              verbose=verbose)


if __name__ == '__main__':
    sct.start_stream_logger()
    run_main()
