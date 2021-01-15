#!/usr/bin/env python
# ######################################################################################################################
#
#
# Compute TSNR using inputed anat.nii.gz and fmri.nii.gz files.
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Sara Dupont
# Created: 2015-03-12
#
# About the license: see the file LICENSE.TXT
# ######################################################################################################################

import sys
import os

import numpy as np

from spinalcordtoolbox.image import Image, add_suffix, empty_like
from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, init_sct, display_viewer_syntax, set_global_loglevel


class Param:
    def __init__(self):
        self.debug = 1
        self.verbose = 1


class Tsnr:
    def __init__(self, param=None, fmri=None, anat=None, out=None):
        if param is not None:
            self.param = param
        else:
            self.param = Param()
        self.fmri = fmri
        self.anat = anat
        self.out = out

    def compute(self):

        fname_data = self.fmri

        # open data
        nii_data = Image(fname_data)
        data = nii_data.data

        # compute mean
        data_mean = np.mean(data, 3)
        # compute STD
        data_std = np.std(data, 3, ddof=1)
        # compute TSNR
        data_tsnr = data_mean / data_std

        # save TSNR
        fname_tsnr = self.out
        nii_tsnr = empty_like(nii_data)
        nii_tsnr.data = data_tsnr
        nii_tsnr.save(fname_tsnr, dtype=np.float32)

        display_viewer_syntax([fname_tsnr])


# PARSER
# ==========================================================================================
def get_parser():
    parser = SCTArgumentParser(
        description="Compute temporal SNR (tSNR) in fMRI time series."
    )
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help="Input fMRI data. Example: fmri.nii.gz"
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode"
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help="tSNR data output file. Example: fmri_tsnr.nii.gz"
    )

    return parser


# MAIN
# ==========================================================================================
def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)

    param = Param()

    fname_src = arguments.i
    if arguments.o is not None:
        fname_dst = arguments.o
    else:
        fname_dst = add_suffix(fname_src, "_tsnr")

    # call main function
    tsnr = Tsnr(param=param, fmri=fname_src, out=fname_dst)
    tsnr.compute()


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

