#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_spinalcord script
#
#   replace the shell test script in sct 1.0
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/09/28
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sct_utils as sct
import test_all
import time
import shutil

class bcolors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    red = '\033[91m'

class Param:
    def __init__(self):
        self.contrasts = ['t2']
        self.subjects = []
        self.files = [['t2.nii.gz', 't2_segmentation_PropSeg.nii.gz']]
        self.sigma = str(4)


def test(data_file_path):
    # initialize parameters
    status = 0
    param = Param()
    sigma = param.sigma
    results_dir = 'results'

    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)

    os.makedirs(results_dir)
    os.chdir(results_dir)

    begin_log_file = "test ran at "+time.strftime("%y%m%d%H%M%S")+"\n"
    fname_log = "sct_register_multimodal.log"

    test_all.write_to_log_file(fname_log, begin_log_file, 'w')

    for contrast in param.contrasts:
        test_all.write_to_log_file(fname_log, bcolors.red+"Contrast: "+contrast+"\n", 'a')
        for f in param.files:
            # Run sct_smooth_spinalcord
            cmd = 'sct_smooth_spinalcord -i ../' + data_file_path + '/' + contrast + '/' + f[0] \
                + ' -c ../' + data_file_path + '/' + contrast + '/' + f[1] \
                + ' -s ' + sigma
            s, output = sct.run(cmd, 0)
            test_all.write_to_log_file(fname_log, output, 'a')
            status += s
            # Isotropic smoothing of the same image with same standard deviation (for the Gaussian) for comparison purposes
            cmd = 'sct_c3d ../' + data_file_path + '/' + contrast + f[0] \
                + ' -smooth ' + sigma + 'x' + sigma + 'x' + sigma + 'vox' \
                + ' -o ' + contrast + '_isotropic_smoothed.nii.gz'
            s, output = sct.run(cmd, 0)
            test_all.write_to_log_file(fname_log, output, 'a')
            status += s
            # Smoothing along Z (corresponding to X, given the orientation of the image)
            cmd = 'sct_c3d ' + data_file_path + '/' + f[0] \
                + ' -smooth ' + sigma + 'x0x0vox' \
                + ' -o ' + contrast + '_z_smoothed.nii.gz'
            s, output = sct.run(cmd, 0)
            test_all.write_to_log_file(fname_log, output, 'a')
            status += s

    os.chdir('..')
    return status


if __name__ == "__main__":
    # call main function
    test()