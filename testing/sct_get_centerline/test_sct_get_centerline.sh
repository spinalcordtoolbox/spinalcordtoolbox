#!/bin/bash
#
# test sct_get_centerline.py

# subject list
SUBJECT_LIST="errsm_23" #"errsm_20 errsm_21 errsm_22 errsm_23 errsm_24"
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
    cmd="sct_get_centerline.py 
      -i ../../data/${subject}/${contrast}/${contrast}.nii.gz 
      -p ../../data/${subject}/${contrast}/${contrast}_centerline_init.nii.gz 
      -g 1 
      -k 4 
      -r 1"
    echo "$cmd"; $cmd
  done
done

