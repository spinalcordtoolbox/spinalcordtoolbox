#!/bin/bash
#
# This script tests sct_warp_atlas2metric.
#

# subject list
SUBJECT_LIST="errsm_03_sub"  # errsm_03 paris_16 paris_17 toronto_E19849"
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

    mkdir $subject
    cd $subject

    # display subject
    echo
    printf "${green}Subject: $subject${NC}\n"
    printf "${red}Contrast: ${contrast}${NC}\n\n"
    cmd="sct_dmri_moco.py
        -i $SCT_DATA_DIR/${subject}/${contrast}/${file1}
        -b $SCT_DATA_DIR/${subject}/${contrast}/${file2}
        -v 1
        -s 15
        -f 0
        -d 3
        -r 0
	-e 0
        -p sinc"
    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd

    cd ..

  done
done

