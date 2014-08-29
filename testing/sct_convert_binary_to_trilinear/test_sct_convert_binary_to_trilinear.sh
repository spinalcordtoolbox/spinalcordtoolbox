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
    cmd="sct_convert_binary_to_trilinear.py
        -i ../../data/${subject}/${contrast}/${file}
        -s 5"
    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd
  done
done

