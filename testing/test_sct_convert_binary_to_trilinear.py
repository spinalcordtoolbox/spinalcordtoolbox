#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_convert_binary_to_trilinear script
#
#   replace the shell test script in sct 1.0
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/09/25
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys
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
        self.files = ['t2_seg.nii.gz']


def test(data_file_path):
    # initialize parameters
    status = 0
    param = Param()
    results_dir = 'results'

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
            cmd = "sct_convert_binary_to_trilinear -i ../"+data_file_path+'/'+contrast+'/'+f+" -s 5"
            s, output = sct.run(cmd)
            test_all.write_to_log_file(fname_log, output, 'a')
            status += s

    return status


if __name__ == "__main__":
    # call main function
    test()




'''

# subject list
SUBJECT_LIST="errsm_23"
CONTRAST_LIST="t2"
file="t2_segmentation_PropSeg.nii.gz"

red='\e[1;31m'
green='\e[1;32m'
NC='\e[0m'

# create results folder and go inside it
mkdir results
cd results

# loop across subjects
for subject in $SUBJECT_LIST; do

  # loop across contrast
  for contrast in $CONTRAST_LIST; do

    # display subject
    echo
    printf "${green}Subject: $subject${NC}\n"
    printf "${red}Contrast: ${contrast}${NC}\n\n"
    cmd="sct_convert_binary_to_trilinear
        -i ../../data/${subject}/${contrast}/${file}
        -s 5"
    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd
  done
done

'''