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

from __future__ import absolute_import, division

import sys
import os
import argparse

import numpy as np

import sct_utils as sct
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.utils import Metavar, SmartFormatter, init_sct
from spinalcordtoolbox.image import Image


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
        nii_tsnr = msct_image.empty_like(nii_data)
        nii_tsnr.data = data_tsnr
        nii_tsnr.save(fname_tsnr, dtype=np.float32)

        sct.display_viewer_syntax([fname_tsnr])


# PARSER
# ==========================================================================================
def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute temporal SNR (tSNR) in fMRI time series.",
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py")
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
        choices=['0', '1'],
        default='1',
        help="Verbosity. 0: None, 1: Verbose"
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help="tSNR data output file. Example: fmri_tsnr.nii.gz"
    )

    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    parser = get_parser()
    param = Param()

    arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    fname_src = arguments.i
    if arguments.o is not None:
        fname_dst = arguments.o
    else:
        fname_dst = sct.add_suffix(fname_src, "_tsnr")

    verbose = int(arguments.v)
    init_sct(log_level=verbose, update=True)  # Update log level

    # call main function
    tsnr = Tsnr(param=param, fmri=fname_src, out=fname_dst)
    tsnr.compute()


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    init_sct()
    param = Param()
    main()
