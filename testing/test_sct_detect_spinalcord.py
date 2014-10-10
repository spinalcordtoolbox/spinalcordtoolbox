#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_detect_spinalcord script
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

import test_all
import shutil
import os
import sct_utils as sct
import time


class bcolors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    red = '\033[91m'


class Param:
    def __init__(self):
        self.contrasts = ['t2']
        self.subjects = []
        self.files = ['t2.nii.gz']


def test(data_file_path):
    # initialize parameters
    status = 0
    param = Param()
    results_dir = 'results_sct_detect_spinalcord'

    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)

    os.makedirs(results_dir)
    os.chdir(results_dir)

    begin_log_file = "test ran at "+time.strftime("%y%m%d%H%M%S")+"\n"
    fname_log = "sct_convert_binary_to_trilinear.log"

    test_all.write_to_log_file(fname_log, begin_log_file, 'w')

    for contrast in param.contrasts:
        test_all.write_to_log_file(fname_log, bcolors.red+"Contrast: "+contrast+"\n", 'a')
        for f in param.files:
            cmd = "sct_detect_spinalcord -i ../"+data_file_path+'/'+contrast+'/'+f+\
                  " -o "+contrast+"_center.nii.gz -t "+contrast
            s, output = sct.run(cmd, 0)
            test_all.write_to_log_file(fname_log, output, 'a')
            status += s

    os.chdir('..')
    return status


if __name__ == "__main__":
    # call main function
    test()