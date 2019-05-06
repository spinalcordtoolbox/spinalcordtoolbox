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

import numpy as np

import sct_utils as sct
import spinalcordtoolbox.image as msct_image
from msct_parser import Parser
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
    parser = Parser(__file__)
    parser.usage.set_description('Compute temporal SNR (tSNR) in fMRI time series.')
    parser.add_option(name='-i',
                      type_value='file',
                      description='fMRI data',
                      mandatory=True,
                      example='fmri.nii.gz')
    parser.add_option(name='-v',
                      type_value='multiple_choice',
                      description='verbose',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name='-o',
                      type_value='file_output',
                      description='tSNR data output file',
                      mandatory=False,
                      example='fmri_tsnr.nii.gz')
    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    parser = get_parser()
    param = Param()

    arguments = parser.parse(sys.argv[1:])
    fname_src = arguments['-i']
    fname_dst = arguments.get("-o", sct.add_suffix(fname_src, "_tsnr"))
    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    # call main function
    tsnr = Tsnr(param=param, fmri=fname_src, out=fname_dst)
    tsnr.compute()


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    param = Param()
    main()
