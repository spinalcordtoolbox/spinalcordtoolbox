#!/bin/bash
#
# test sct_smooth_spinalcord.py

red='\e[1;31m'
green='\e[1;32m'
blue='\e[4;34m'
NC='\e[0m'

# subject list
SUBJECT_LIST="errsm_23" 
CONTRAST_LIST="t1 t2" #"t1 t2"

# standard deviation of the Gaussian kernel used to smooth the image
sigma=4 # default value

while getopts “s:” OPTION
do
  case $OPTION in
    s)
      sigma=$OPTARG
      ;;
  esac
done

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
	
	# display the standard deviation of the Gaussian kernel used to smooth the image
	echo
	printf "${blue}Standard deviation of the Gaussian kernel: $sigma${NC}\n"	
	
    # Run sct_smooth_spinalcord.py
    cmd="sct_smooth_spinalcord.py
      -i ../../data/${subject}/${contrast}/${contrast}.nii.gz
      -c ../../data/${subject}/${contrast}/${contrast}_segmentation_PropSeg.nii.gz
	  -s ${sigma}"
    echo "$cmd"
    $cmd
	
	# Isotropic smoothing of the same image with same standard deviation (for the Gaussian) for comparison purposes
	cmd="c3d
		../../data/${subject}/${contrast}/${contrast}.nii.gz
		-smooth ${sigma}x${sigma}x${sigma}vox
		-o ${contrast}_isotropic_smoothed.nii.gz"
	echo "$cmd"
	$cmd

	# Smoothing along Z
	cmd="c3d
		../../data/${subject}/${contrast}/${contrast}.nii.gz
		-smooth 0x0x${sigma}vox
		-o ${contrast}_z_smoothed.nii.gz"
	echo "$cmd"
	$cmd

  done
done

