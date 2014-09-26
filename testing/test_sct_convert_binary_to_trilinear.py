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

CONTRAST_LIST = ['t2']


class bcolors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'


class Param:
    def __init__(self):
        self.contrasts = []
        self.subjects = []
        self.files = []

def test():
    # initialize parameters
    param = Param()
    results_dir = '/resutls'

    param.contrasts.append(CONTRAST_LIST)
    param.files.append('t2_segmentation_PropSeg.nii.gz')

    if os.path.isdir(results_dir):
        os.remove(results_dir)

    os.makedirs(results_dir)
    os.chdir(results_dir)

    log_file = open()


















if __name__ == "__main__":
    # call main function
    test()





#!/bin/bash
#

# subject list
SUBJECT_LIST="errsm_23"
CONTRAST_LIST="t2"
file="t2_segmentation_PropSeg.nii.gz"

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
    cmd="sct_convert_binary_to_trilinear
        -i ../../data/${subject}/${contrast}/${file}
        -s 5"
    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd
  done
done

