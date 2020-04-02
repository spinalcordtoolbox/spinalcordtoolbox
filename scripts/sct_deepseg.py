#!/usr/bin/env python
# -*- coding: utf-8
# This command-line tool is the interface for the deepseg API that performs segmentation using deep learning from the
# ivadomed package.

# TODO: implement feature with a new flag (e.g. path-to-model) that will give the possibility to point to a model
#  folder, in case a test model is not on OSF and is not listed in MODELS.

from __future__ import absolute_import

import sys
import os
import argparse

from spinalcordtoolbox.utils import Metavar, SmartFormatter
from spinalcordtoolbox.deepseg.core import ParamDeepseg, segment_nifti
from spinalcordtoolbox.deepseg.models import MODELS

from sct_utils import init_sct


def get_parser():

    param_default = ParamDeepseg()

    parser = argparse.ArgumentParser(
        description="Segmentation using deep learning.",
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        required=True,
        help="Image to segment.",
        metavar=Metavar.file)
    mandatory.add_argument(
        "-m",
        help="Model to use.",
        choices=list(MODELS.keys()))

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
    param = ParamDeepseg

    parser = get_parser()
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    # TODO: instead of assigning each args param, we could pass args while instanciating ParamDeepseg(args), and the
    #  class would deal with assigning arguments to each field.
    if 'o' in args:
        param.output_suffix = args.o

    segment_nifti(args.i, args.m)


if __name__ == '__main__':
    init_sct()
    main()
