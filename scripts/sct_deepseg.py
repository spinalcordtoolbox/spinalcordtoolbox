#!/usr/bin/env python
# -*- coding: utf-8
"""
This command-line tool is the interface for the deepseg API that performs segmentation using deep learning from the
ivadomed package.
"""

# TODO: Add link to example image so users can decide wether their images look "close enought" to some of the proposed
#  models (e.g., mice, etc.).


from __future__ import absolute_import

import sys
import os
import argparse

import spinalcordtoolbox as sct
import spinalcordtoolbox.deepseg.core
import spinalcordtoolbox.deepseg.models
from spinalcordtoolbox.utils import Metavar, SmartFormatter

from sct_utils import init_sct, printv


def get_parser():

    param_default = sct.deepseg.core.ParamDeepseg()

    parser = argparse.ArgumentParser(
        description="Segmentation using deep learning.",
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        help="Image to segment.",
        metavar=Metavar.file)

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
        help="Install default models. Note: these models are downloaded during normal SCT installation.")
    seg.add_argument(
        "-mpath",
        help="Path to model, in case you would like to use a custom model. The model folder should follow the "
             "conventions listed in: URL.",
        metavar=Metavar.folder)

    misc = parser.add_argument_group('\nMISC')
    misc.add_argument(
        "-o",
        help="Output segmentation suffix. In case of multi-class segmentation, class-specific suffixes will be added.",
        metavar=str,
        default=param_default.output_suffix)
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
    param = sct.deepseg.core.ParamDeepseg()

    parser = get_parser()
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    # TODO: instead of assigning each args param, we could pass args while instanciating ParamDeepseg(args), and the
    #  class would deal with assigning arguments to each field.

    if args.list_models:
        sct.deepseg.models.list_models()
        exit(0)

    if args.install_model is not None:
        sct.deepseg.models.install_model(args.install_model)
        exit(0)

    if args.install_default_models:
        sct.deepseg.models.install_default_models()
        exit(0)

    if 'i' not in args:
        parser.error("the following arguments is required: -i")

    if 'o' in args:
        param.output_suffix = args.o

    # Get model path
    if args.m:
        name_model = args.m
        sct.deepseg.models.is_model(name_model)  # TODO: no need for this (argparse already checks)
        if not spinalcordtoolbox.deepseg.models.is_installed(name_model):
            printv("Model {} is not installed. Installing it now...".format(name_model))
            spinalcordtoolbox.deepseg.models.install_model(name_model)
        path_model = spinalcordtoolbox.deepseg.models.folder(name_model)
    elif args.mpath:
        # TODO: check integrity of folder model
        path_model = args.mpath

    sct.deepseg.core.segment_nifti(args.i, path_model)


if __name__ == '__main__':
    init_sct()
    main()
