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
import sct_utils as sct
from msct_parser import Parser
from msct_image import Image
import numpy as np


class Param:
    def __init__(self):
        self.debug = 1
        self.verbose = 1


class Tsnr:
    def __init__(self, param=None, fmri=None, anat=None):
        if param is not None:
            self.param = param
        else:
            self.param = Param()
        self.fmri = fmri
        self.anat = anat

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
        fname_tsnr = sct.add_suffix(fname_data, '_tsnr')
        nii_tsnr = nii_data
        nii_tsnr.data = data_tsnr
        nii_tsnr.setFileName(fname_tsnr)
        nii_tsnr.save(type='float32')

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
    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    parser = get_parser()
    param = Param()

    arguments = parser.parse(sys.argv[1:])
    fname_src = arguments['-i']
    verbose = int(arguments['-v'])

    # call main function
    tsnr = Tsnr(param=param, fmri=fname_src)
    tsnr.compute()


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    param = Param()
    main()
