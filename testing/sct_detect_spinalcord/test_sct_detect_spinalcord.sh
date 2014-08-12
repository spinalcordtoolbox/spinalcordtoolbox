#!/bin/bash
#
# test sct_get_centerline.py

# subject list
SUBJECT_LIST="errsm_23" 
CONTRAST_LIST="t2" #"t1 t2"

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
    cmd="sct_detect_spinalcord 
      -i ../../data/${subject}/${contrast}/${contrast}.nii.gz 
      -o ${subject}_${contrast}_center.nii.gz 
      -t ${contrast}"
    echo "$cmd"; $cmd
  done
done

