#!/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Compute MT saturation map and T1 map from a PD-weigthed, a T1-weighted and MT-weighted FLASH images
#
# Reference paper:
#    Helms G, Dathe H, Kallenberg K, Dechent P. High-resolution maps of magnetization transfer with inherent correction
#    for RF inhomogeneity and T1 relaxation obtained from 3D FLASH MRI. Magn Reson Med 2008;60(6):1396-1407.

# ---------------------------------------------------------------------------------------
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys
import os
import json

from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, init_sct, printv, display_viewer_syntax, set_loglevel
from spinalcordtoolbox.qmri.mt import compute_mtsat
from spinalcordtoolbox.image import Image, splitext


def get_parser():
    parser = SCTArgumentParser(
        description='Compute MTsat and T1map. '
                    'Reference: Helms G, Dathe H, Kallenberg K, Dechent P. High-resolution maps of magnetization '
                    'transfer with inherent correction for RF inhomogeneity and T1 relaxation obtained from 3D FLASH '
                    'MRI. Magn Reson Med 2008;60(6):1396-1407.'
    )

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        "-mt",
        required=True,
        help="Image with MT_ON",
        metavar=Metavar.file,
    )
    mandatoryArguments.add_argument(
        "-pd",
        required=True,
        help="Image PD weighted (typically, the MT_OFF)",
        metavar=Metavar.file,
    )
    mandatoryArguments.add_argument(
        "-t1",
        required=True,
        help="Image T1-weighted",
        metavar=Metavar.file,
    )

    optional = parser.add_argument_group('\nOPTIONAL ARGUMENTS')
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-trmt",
        help="TR [in ms] for mt image. By default, will be fetch from the json sidecar (if it exists).",
        type=float,
        metavar=Metavar.float,
    )
    optional.add_argument(
        "-trpd",
        help="TR [in ms] for pd image. By default, will be fetch from the json sidecar (if it exists).",
        type=float,
        metavar=Metavar.float,
    )
    optional.add_argument(
        "-trt1",
        help="TR [in ms] for t1 image. By default, will be fetch from the json sidecar (if it exists).",
        type=float,
        metavar=Metavar.float,
    )
    optional.add_argument(
        "-famt",
        help="Flip angle [in deg] for mt image. By default, will be fetch from the json sidecar (if it exists).",
        type=float,
        metavar=Metavar.float,
    )
    optional.add_argument(
        "-fapd",
        help="Flip angle [in deg] for pd image. By default, will be fetch from the json sidecar (if it exists).",
        type=float,
        metavar=Metavar.float,
    )
    optional.add_argument(
        "-fat1",
        help="Flip angle [in deg] for t1 image. By default, will be fetch from the json sidecar (if it exists).",
        type=float,
        metavar=Metavar.float,
    )
    optional.add_argument(
        "-b1map",
        help="B1 map",
        metavar=Metavar.file,
        default=None)
    optional.add_argument(
        "-omtsat",
        metavar=Metavar.str,
        help="Output file for MTsat",
        default="mtsat.nii.gz")
    optional.add_argument(
        "-ot1map",
        metavar=Metavar.str,
        help="Output file for T1map",
        default="t1map.nii.gz")
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


def get_json_file_name(fname, check_exist=False):
    """
    Get json file name by replacing '.nii' or '.nii.gz' extension by '.json'.
    Check if input file follows NIFTI extension rules.
    Optional: check if json file exists.
    :param fname: str: Input NIFTI file name.
    check_exist: Bool: Check if json file exists.
    :return: fname_json
    """
    list_ext = ['.nii', '.nii.gz']
    basename, ext = splitext(fname)
    if ext not in list_ext:
        raise ValueError("Problem with file: {}. Extension should be one of {}".format(fname, list_ext))
    fname_json = basename + '.json'

    if check_exist:
        if not os.path.isfile(fname_json):
            raise FileNotFoundError(f"{fname_json} not found. Either provide the file alongside {fname}, or explicitly "
                                    f"set tr and fa arguments for this image type.")

    return fname_json


def fetch_metadata(fname_json, field):
    """
    Return specific field value from json sidecar.
    :param fname_json: str: Json file
    :param field: str: Field to retrieve
    :return: value of the field.
    """
    with open(fname_json) as f:
        metadata = json.load(f)
    if field not in metadata:
        KeyError("Json file {} does not contain the field: {}".format(fname_json, field))
    else:
        return metadata[field]


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    printv('Load data...', verbose)
    nii_mt = Image(arguments.mt)
    nii_pd = Image(arguments.pd)
    nii_t1 = Image(arguments.t1)
    if arguments.b1map is None:
        nii_b1map = None
    else:
        nii_b1map = Image(arguments.b1map)

    if arguments.trmt is None:
        arguments.trmt = fetch_metadata(get_json_file_name(arguments.mt, check_exist=True), 'RepetitionTime') * 1000  # converted from s to ms
    if arguments.trpd is None:
        arguments.trpd = fetch_metadata(get_json_file_name(arguments.pd, check_exist=True), 'RepetitionTime') * 1000  # converted from s to ms
    if arguments.trt1 is None:
        arguments.trt1 = fetch_metadata(get_json_file_name(arguments.t1, check_exist=True), 'RepetitionTime') * 1000  # converted from s to ms
    if arguments.famt is None:
        arguments.famt = fetch_metadata(get_json_file_name(arguments.mt, check_exist=True), 'FlipAngle')
    if arguments.fapd is None:
        arguments.fapd = fetch_metadata(get_json_file_name(arguments.pd, check_exist=True), 'FlipAngle')
    if arguments.fat1 is None:
        arguments.fat1 = fetch_metadata(get_json_file_name(arguments.t1, check_exist=True), 'FlipAngle')

    # compute MTsat
    nii_mtsat, nii_t1map = compute_mtsat(nii_mt, nii_pd, nii_t1,
                                         arguments.trmt, arguments.trpd, arguments.trt1,
                                         arguments.famt, arguments.fapd, arguments.fat1,
                                         nii_b1map=nii_b1map)

    # Output MTsat and T1 maps
    printv('Generate output files...', verbose)
    nii_mtsat.save(arguments.omtsat)
    nii_t1map.save(arguments.ot1map)

    display_viewer_syntax([arguments.omtsat, arguments.ot1map],
                              colormaps=['gray', 'gray'],
                              minmax=['-10,10', '0, 3'],
                              opacities=['1', '1'],
                              verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
