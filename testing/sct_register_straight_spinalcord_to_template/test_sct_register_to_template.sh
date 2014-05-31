#!/bin/bash
#
# test sct_register_straight_spinalcord_to_template.py

# subject list
SUBJECT_LIST="errsm_11" #"errsm_20 errsm_21 errsm_22 errsm_23 errsm_24"
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
    cmd="sct_register_straight_spinalcord_to_template.py 
      -i ../../data/${subject}/${contrast}/${subject}_${contrast}_rpi_straight.nii.gz 
      -l ../../data/${subject}/${contrast}/${subject}_${contrast}_rpi_straight_landmarks_C2_T2.nii.gz 
      -t ../../../data/template/MNI-Poly-AMU_T2.nii.gz 
      -f ../../../data/template/landmarks_C2_T2.nii.gz
      -m ../../../data/template/mask_gaussian_templatespace_sigma20.nii.gz
      -r 1"

    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd
  done
done

