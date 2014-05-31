#!/bin/bash
#
# This script tests sct_warp_atlas2metric.
#

# subject list
SUBJECT_LIST="errsm_23" 
CONTRAST_LIST="mt" 
file="mtr.nii.gz"
folder_atlas="atlas"


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
    cmd="sct_estimate_MAP_tracts.py
        -i ../../data/${subject}/${contrast}/${file}
        -t ../../data/${subject}/${folder_atlas}
        -m weightedaverage
        -o quantif_${CONTRAST_LIST}.txt"
    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd
  done
done

