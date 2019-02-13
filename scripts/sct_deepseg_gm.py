#!/usr/bin/env python
# -*- coding: utf-8
# This command-line tool is the interface for the deepseg_gm API
# that implements the model for the Spinal Cord Gray Matter Segmentation.
#
# Reference paper:
#     Perone, C. S., Calabrese, E., & Cohen-Adad, J. (2017).
#     Spinal cord gray matter segmentation using deep dilated convolutions.
#     URL: https://arxiv.org/abs/1710.01269

from __future__ import absolute_import

import sys
import os

import sct_utils as sct
from msct_parser import Parser


def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('Spinal Cord Gray Matter (GM) Segmentation using deep dilated convolutions. '
                                 'Reference: CS Perone, E Calabrese, J Cohen-Adad. Spinal cord gray matter segmentation using deep dilated convolutions (2017). arxiv.org/abs/1710.01269')

    parser.add_option(name="-i",
                      type_value="file",
                      description="Image filename to segment (3D volume). "
                                  "Contrast must be similar to T2*-weighted, "
                                  "i.e., WM dark, GM bright and CSF bright.",
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
                      description="The path where the quality control generated "
                                  "content will be saved",
                      default_value=None)

    parser.add_option(name="-m",
                      type_value='multiple_choice',
                      description="Model to use (large or challenge)."
                                  "The model 'large' will be slower but "
                                  "will yield better results. The model "
                                  "'challenge' was built using data from "
                                  "the following challenge: goo.gl/h4AVar.",
                      mandatory=False,
                      example=['large', 'challenge'],
                      default_value='large')

    parser.add_option(name="-thr",
                      type_value='float',
                      description="Threshold to apply in the segmentation "
                                  "predictions, use 0 (zero) to disable it.",
                      mandatory=False,
                      default_value=0.999,
                      example=0.999)

    parser.add_option(name='-igt',
                      type_value='image_nifti',
                      description='File name of ground-truth segmentation.',
                      mandatory=False)

    parser.add_option(name="-t",
                      type_value=None,
                      description="Enable TTA (test-time augmentation). "
                                  "Better results, but takes more time and "
                                  "provides non-deterministic results.",
                      mandatory=False)

    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="Verbose: 0 = no verbosity, 1 = verbose.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')

    return parser


def generate_qc(fn_in, fn_seg, args, path_qc):
    """
    Generate a QC entry allowing to quickly review the segmentation process.
    """

    import spinalcordtoolbox.reports.qc as qc
    import spinalcordtoolbox.reports.slice as qcslice
    from spinalcordtoolbox.image import Image

    qc.add_entry(
     src=fn_in,
     process="sct_deepseg_gm",
     args=args,
     path_qc=path_qc,
     plane='Axial',
     qcslice=qcslice.Axial([Image(fn_in), Image(fn_seg)]),
     qcslice_operations=[qc.QcImage.listed_seg],
     qcslice_layout=lambda x: x.mosaic(),
    )


def run_main():
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    input_filename = arguments["-i"]

    try:
        output_filename = arguments["-o"]
    except KeyError:
        output_filename = sct.add_suffix(input_filename, '_gmseg')

    use_tta = "-t" in arguments
    verbose = arguments["-v"]
    model_name = arguments["-m"]
    threshold = arguments['-thr']

    if threshold > 1.0 or threshold < 0.0:
        raise RuntimeError("Threshold should be between 0.0 and 1.0.")

    # Threshold zero means no thresholding
    if threshold == 0.0:
        threshold = None

    from spinalcordtoolbox.deepseg_gm import deepseg_gm
    deepseg_gm.check_backend()

    out_fname = deepseg_gm.segment_file(input_filename, output_filename,
                                        model_name, threshold, int(verbose),
                                        use_tta)

    path_qc = arguments.get("-qc", None)
    if path_qc is not None:
        generate_qc(input_filename, out_fname, sys.argv[1:], os.path.abspath(path_qc))

    sct.display_viewer_syntax([input_filename, format(out_fname)],
                              colormaps=['gray', 'red'],
                              opacities=['1', '0.7'],
                              verbose=verbose)


if __name__ == '__main__':
    sct.init_sct()
    run_main()
