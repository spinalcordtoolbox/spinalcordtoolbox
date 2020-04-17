#!/usr/bin/env python
# -*- coding: utf-8
"""
This command-line tool is the interface for the deepseg API that performs segmentation using deep learning from the
ivadomed package.
"""

# TODO: Add link to example image so users can decide wether their images look "close enought" to some of the proposed
#  models (e.g., mice, etc.).
# TODO: add test to make sure that all postprocessing flags match the core.DEFAULT dictionary items
# TODO: Fetch default value (and display) depending on the model that is used.

from __future__ import absolute_import

import sys
import os
import argparse

import spinalcordtoolbox as sct
import spinalcordtoolbox.deepseg.core
import spinalcordtoolbox.deepseg.models
import spinalcordtoolbox.utils

from sct_utils import init_sct, printv, display_viewer_syntax


def get_parser():

    parser = argparse.ArgumentParser(
        description="Segmentation using deep learning.",
        add_help=None,
        formatter_class=sct.utils.SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        help="Image to segment.",
        metavar=sct.utils.Metavar.file)

    seg = parser.add_argument_group('\nMODELS')
    seg.add_argument(
        "-m",
        help="Model to use. For a description of each available model, please write: 'sct_deepseg -list-models'.",
        choices=list(sct.deepseg.models.MODELS.keys()))
    seg.add_argument(
        "-list-models",
        action='store_true',
        help="Display a list of available models.")
    seg.add_argument(
        "-install-model",
        help="Install specified model.",
        choices=list(sct.deepseg.models.MODELS.keys()))
    seg.add_argument(
        "-install-default-models",
        action='store_true',
        help="Install all default models. Note: default models should already be installed during SCT installation.")
    # TODO: replace URL below
    seg.add_argument(
        "-mpath",
        help="Path to model, in case you would like to use a custom model. The model folder should follow the "
             "conventions listed in: URL.",
        metavar=sct.utils.Metavar.folder)

    misc = parser.add_argument_group('\nPARAMETERS')
    misc.add_argument(
        "-thr",
        type=float,
        help="Binarize segmentation with specified threshold. Set to 0 for no thresholding (i.e., soft segmentation). "
             "Default value is model-specific and was set during optimization "
             "(more info at https://github.com/sct-pipeline/deepseg-threshold).",
        metavar=sct.utils.Metavar.float)
    misc.add_argument(
        "-keep-largest-object",
        type=int,
        help="Remove false negative segmentation by only keeping the largest blob.",
        choices=(0, 1))
    misc.add_argument(
        "-fill-holes",
        type=int,
        help="Fill small holes in the segmentation.",
        choices=(0, 1))

    misc = parser.add_argument_group('\nMISC')
    misc.add_argument(
        "-o",
        help="Output segmentation suffix. In case of multi-class segmentation, class-specific suffixes will be added.",
        metavar=sct.utils.Metavar.str,
        default='_seg')
    misc.add_argument(
        "-v",
        type=int,
        help="Verbose: 0 = no verbosity, 1 = verbose.",
        choices=(0, 1),
        default=1)
    misc.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    args = {k: v for k, v in vars(args).items() if v is not None}
    # separate out the segmentation `param` args from the top level args
    # fname_input = args.pop('i')
    # name_model = args.pop('m')
    # list_models =
    # install_model = args.pop('install_model')
    # install_default_models = args.pop('install_default_models')

    # Deal with model
    if args['list_models']:
        sct.deepseg.models.list_models()
        exit(0)
    if 'install_model' in args:
        sct.deepseg.models.install_model(args['install_model'])
        exit(0)
    if args['install_default_models']:
        sct.deepseg.models.install_default_models()
        exit(0)

    # Deal with input/output
    if 'i' not in args:
        parser.error("The following arguments is required: -i")
    if not os.path.isfile(args['i']):
        parser.error("This file does not exist: {}".format(args['i']))

    # Get model path
    if 'm' in args:
        if not spinalcordtoolbox.deepseg.models.is_installed(args['m']):
            printv("Model {} is not installed. Installing it now...".format(args['m']))
            spinalcordtoolbox.deepseg.models.install_model(args['m'])
        path_model = spinalcordtoolbox.deepseg.models.folder(args['m'])
    elif 'path_model' in args:
        # TODO: check integrity of folder model
        path_model = args['path_model']
    else:
        parser.error("You need to specify either -m or -mpath.")

    fname_seg = sct.deepseg.core.segment_nifti(args['i'], path_model, args)

    display_viewer_syntax([args['i'], fname_seg], colormaps=['gray', 'red'], opacities=['', '0.7'])


if __name__ == '__main__':
    init_sct()
    main()
