#!/bin/bash
#
# This script tests sct_warp_atlas2metric.
#

# subject list
SUBJECT_LIST="errsm_23" 
CONTRAST_LIST="dmri"
file1="dmri.nii.gz"
file2="bvecs.txt"


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
    cmd="sct_dmri_moco.py
        -i ../../data/${subject}/${contrast}/${file1}
        -b ../../data/${subject}/${contrast}/${file2}
        -s sinc"
    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd
  done
done

