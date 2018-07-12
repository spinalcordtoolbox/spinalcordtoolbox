#!/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Function to segment the spinal cord using convolutional neural networks
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener & Charley Gros
# Modified: 2018-06-05
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys

import sct_utils as sct

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_parser():
    """Initialize the parser."""
    from msct_parser import Parser
    parser = Parser(__file__)
    parser.usage.set_description("""Spinal Cord Segmentation using convolutional networks. \n\nReference: C Gros, B De Leener, et al. Automatic segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional neural networks (2018). arxiv.org/abs/1805.06349""")

    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="input image.",
                      mandatory=True,
                      example="t1.nii.gz")
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast.",
                      mandatory=True,
                      example=['t1', 't2', 't2s', 'dwi'])
    parser.add_option(name="-centerline",
                      type_value="multiple_choice",
                      description="choice of spinal cord centerline algorithm.",
                      mandatory=False,
                      example=['svm', 'cnn'],
                      default_value="svm")
    parser.add_option(name="-brain",
                      type_value="multiple_choice",
                      description="indicate if the input image is expected to contain brain sections: 1: contains brain section, 0: no brain section. To indicate this parameter could speed the segmentation process. Default value is 1 if -c is t1 or t2 (likely includes the brain), or 0 otherwise. Note that this flag is only effective with -ctr cnn",
                      mandatory=False,
                      example=["0", "1"])
    parser.add_option(name="-kernel",
                      type_value="multiple_choice",
                      description="choice of 2D or 3D kernels for the segmentation. Note that segmentation with 3D kernels is significantely longer than with 2D kernels.",
                      mandatory=False,
                      example=['2d', '3d'],
                      default_value="2d")
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="output folder.",
                      mandatory=False,
                      example="My_Output_Folder/",
                      default_value="")
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="remove temporary files.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="1: display on, 0: display off (default)",
                      mandatory=False,
                      example=["0", "1"],
                      default_value="1")
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=None)
    parser.add_option(name='-igt',
                      type_value='image_nifti',
                      description='File name of ground-truth segmentation.',
                      mandatory=False)
    return parser


def generate_qc(fn_in, fn_seg, args, path_qc):
    """Generate a QC entry allowing to quickly review the segmentation process."""
    import spinalcordtoolbox.reports.qc as qc
    import spinalcordtoolbox.reports.slice as qcslice
    from msct_image import Image

    qc.add_entry(
     src=fn_in,
     process="sct_deepseg_sc",
     args=args,
     path_qc=path_qc,
     plane='Axial',
     qcslice=qcslice.Axial([Image(fn_in), Image(fn_seg)]),
     qcslice_operations=[qc.QcImage.listed_seg],
     qcslice_layout=lambda x: x.mosaic(),
    )


def main():
    """Main function."""
    sct.init_sct()
    parser = get_parser()
    args = sys.argv[1:]
    arguments = parser.parse(args)

    fname_image = arguments['-i']
    contrast_type = arguments['-c']

    ctr_algo = arguments["-centerline"]
    # if "-centerline" not in args and contrast_type == 't2s':
    #     ctr_algo = 'cnn'

    if "-brain" not in args:
        if contrast_type in ['t2s', 'dwi']:
            brain_bool = False
        if contrast_type in ['t1', 't2']:
            brain_bool = True
    else:
        brain_bool = bool(int(arguments["-brain"]))

    kernel_size = arguments["-kernel"]
    if kernel_size == '3d' and contrast_type == 'dwi':
        kernel_size = '2d'
        sct.printv('3D kernel model for dwi contrast is not available. 2D kernel model is used instead.', type="warning")

    if '-ofolder' not in args:
        output_folder = os.getcwd()
    else:
        output_folder = arguments["-ofolder"]

    remove_temp_files = int(arguments['-r'])

    verbose = arguments['-v']

    path_qc = arguments.get("-qc", None)

    algo_config_stg = '\nMethod:'
    algo_config_stg += '\n\tCenterline algorithm: ' + ctr_algo
    algo_config_stg += '\n\tAssumes brain section included in the image: ' + str(brain_bool)
    algo_config_stg += '\n\tDimension of the segmentation kernel convolutions: ' + kernel_size + '\n'
    sct.printv(algo_config_stg)

    from spinalcordtoolbox.deepseg_sc import script
    fname_seg = script.deep_segmentation_spinalcord(
     fname_image, contrast_type, output_folder,
     ctr_algo=ctr_algo, brain_bool=brain_bool, kernel_size=kernel_size,
     remove_temp_files=remove_temp_files, verbose=verbose)

    if path_qc is not None:
        generate_qc(fname_image, fname_seg, args, os.path.abspath(path_qc))

    sct.display_viewer_syntax([fname_image, os.path.join(output_folder, fname_seg)], colormaps=['gray', 'red'], opacities=['', '0.7'])


if __name__ == "__main__":
    main()
