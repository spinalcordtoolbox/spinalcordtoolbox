#!/bin/bash
#
# test sct_smooth_spinalcord.py

# subject list
SUBJECT_LIST="errsm_23" #"errsm_20 errsm_21 errsm_22 errsm_23 errsm_24"
CONTRAST_LIST="t1 t2" #"t1 t2"

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

    # Run sct_smooth_spinalcord.py
    cmd="sct_smooth_spinalcord.py
      -i ../../data/${subject}/${contrast}/${contrast}.nii.gz
      -c ../../data/${subject}/${contrast}/${contrast}_segmentation_PropSeg.nii.gz"
    echo "$cmd"
    $cmd
	
	# Isotropic smoothing of the same image with same standard deviation (for the Gaussian) for comparison purposes
	cmd="c3d
		../../data/${subject}/${contrast}/${contrast}.nii.gz
		-smooth 4x4x4vox
		-o ${contrast}_isotropic_smoothed.nii.gz"
	echo "$cmd"
	$cmd

  done
done

