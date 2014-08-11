#!/bin/bash
#
# This script tests sct_extract_metric.
#

# subject list
SUBJECT_LIST="errsm_23" 
CONTRAST_LIST="mt" 
file=mtr.nii.gz
label_folder=template


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
    cmd="sct_extract_metric.py
        -i ../../data/${subject}/${contrast}/${file}
        -f ../../sct_warp_template/results/label/atlas
        -l 2,17
        -m wa
		-v 1:3
        -o quantif_${contrast}.txt"
    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd
  done
done

