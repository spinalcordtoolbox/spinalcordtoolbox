#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_extract_metric script
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
        self.contrasts = ['mt']
        self.subjects = []
        self.files = [['t2_segmentation_PropSeg.nii.gz','t2_manual_segmentation.nii.gz', 't2_seg.nii.gz']]


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
    fname_log = "sct_propseg.log"

    test_all.write_to_log_file(fname_log, begin_log_file, 'w')

    for contrast in param.contrasts:
        test_all.write_to_log_file(fname_log, bcolors.red+"Contrast: "+contrast+"\n", 'a')
        for f in param.files:
            cmd1 = 'sct_extract_metric -i ../' + data_file_path + '/' + contrast + '/' + f[0] \
                + ' -t ' + contrast \
                + ' -mesh'\
                + ' -cross'\
                + ' -centerline-binary'\
                + ' -verbose'
            cmd2 = 'sct_dice_coefficient ../' + data_file_path + '/' + contrast + '/' + f[1] \
                + ' ' + f[2] \
                + ' -bmax'
            s, output = sct.run(cmd1, 0)
            test_all.write_to_log_file(fname_log, output, 'a')
            status += s
            s, output = sct.run(cmd2, 0)
            test_all.write_to_log_file(fname_log, output, 'a')
            status += s

    os.chdir('..')
    return status


if __name__ == "__main__":
    # call main function
    test()



"""
#!/bin/bash
#
# test sct_propseg
# This program launch the testing of automatic spinal cord segmentation on five subjects (in data folder)
# Results can be validated using FSLVIEW or MITKWorkbench
#
# Common values of Dice coefficient when accurate segmentation: [0.85, 0.93]

# subject list
SUBJECT_LIST="errsm_23" #
CONTRAST_LIST="t2" # t1 or t2

red='\e[1;31m'
green='\e[1;32m'
NC='\e[0m'

# if results folder exists, delete it
if [ -e "results" ]; then
rm -rf results
fi

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
        cmd="sct_propseg
            -i ../../data/${subject}/${contrast}/${contrast}.nii.gz
            -t ${contrast}
            -mesh
            -cross
            -centerline-binary
            -verbose"
        echo ==============================================================================================
        echo "$cmd"
        echo ==============================================================================================
        $cmd

        cmd="sct_dice_coefficient
            ../../data/${subject}/${contrast}/${contrast}_manual_segmentation.nii.gz
            ${contrast}_seg.nii.gz
            -bmax"
        echo ==============================================================================================
        echo "$cmd"
        echo ==============================================================================================
        $cmd
    done
done
"""