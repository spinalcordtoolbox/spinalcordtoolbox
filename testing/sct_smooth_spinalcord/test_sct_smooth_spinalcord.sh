#!/bin/bash
#
# test sct_smooth_spinalcord

red='\e[1;31m'
green='\e[1;32m'
blue='\e[4;34m'
NC='\e[0m'

# subject list
SUBJECT_LIST="errsm_23" 
CONTRAST_LIST="t1" #"t1 t2"

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
	printf "${blue}Standard deviation of the Gaussian kernel (in terms of voxels): $sigma${NC}\n"	

	# create down sampled files
    sct_c3d ../../data/${subject}/${contrast}/${contrast}.nii.gz -resample 50%  -o ../../data/${subject}/${contrast}/${contrast}_DS.nii.gz
    sct_c3d ../../data/${subject}/${contrast}/${contrast}_DS.nii.gz -resample 50%  -o ../../data/${subject}/${contrast}/${contrast}_DS.nii.gz
    sct_c3d ../../data/${subject}/${contrast}/${contrast}_segmentation_PropSeg.nii.gz -resample 50%  -o ../../data/${subject}/${contrast}/${contrast}_segmentation_PropSeg_DS.nii.gz
    sct_c3d ../../data/${subject}/${contrast}/${contrast}_segmentation_PropSeg_DS.nii.gz -resample 50%  -o ../../data/${subject}/${contrast}/${contrast}_segmentation_PropSeg_DS.nii.gz

	echo ==============================================================================================
    echo Multiply voxels intensity
    echo ==============================================================================================
    fslmaths ../../data/${subject}/${contrast}/${contrast}_DS.nii.gz -mul 16 ../../data/${subject}/${contrast}/${contrast}_DS.nii.gz
    fslmaths ../../data/${subject}/${contrast}/${contrast}_segmentation_PropSeg_DS.nii.gz -mul 16 ../../data/${subject}/${contrast}/${contrast}_segmentation_PropSeg_DS.nii.gz

    # Run sct_smooth_spinalcord
    cmd="sct_smooth_spinalcord
      -i ../../data/${subject}/${contrast}/${contrast}_DS.nii.gz
      -c ../../data/${subject}/${contrast}/${contrast}_segmentation_PropSeg_DS.nii.gz
	  -s ${sigma}"
    echo "$cmd"
    $cmd
	
	# Isotropic smoothing of the same image with same standard deviation (for the Gaussian) for comparison purposes
	cmd="sct_c3d
		../../data/${subject}/${contrast}/${contrast}.nii.gz
		-smooth ${sigma}x${sigma}x${sigma}vox
		-o ${contrast}_isotropic_smoothed.nii.gz"
	echo "$cmd"
	$cmd

	# Smoothing along Z (corresponding to X, given the orientation of the image)
	cmd="sct_c3d
		../../data/${subject}/${contrast}/${contrast}.nii.gz
		-smooth ${sigma}x0x0vox
		-o ${contrast}_z_smoothed.nii.gz"
	echo "$cmd"
	$cmd

  done
done