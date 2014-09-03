#!/bin/bash
#
# This script tests sct_register_to_template.sh and computes the DICE coefficient with manual segmentation.
#
# It requires:
# - anatomical input
# - landmark (e.g., one point at C2, another at T3).
# - spinal cord segmentation or centerline  (e.g. obtained from sct_segmentation_propagation).
# 
# To run it, type:
#   ./test_sct_register_to_template.sh


# subject list
SUBJECT_LIST="errsm_23" # errsm_18
CONTRAST_LIST="t2" # t1 to t2

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

    #create down sampled files

    echo ==============================================================================================
    echo Down sampling files
    echo ==============================================================================================
    c3d ../../../data/template/MNI-Poly-AMU_T2.nii.gz -resample 50%  -o ../../data/MNI-Poly-AMU_T2.nii.gz
    c3d ../../data/MNI-Poly-AMU_T2.nii.gz -resample 50%  -o ../../data/MNI-Poly-AMU_T2.nii.gz
    c3d ../../../data/template/landmarks_center.nii.gz -resample 50% -o ../../data/landmarks_center.nii.gz
    c3d ../../data/landmarks_center.nii.gz -resample 50% -o ../../data/landmarks_center.nii.gz
    c3d ../../../data/template/MNI-Poly-AMU_cord.nii.gz -resample 50% -o ../../data/MNI-Poly-AMU_cord.nii.gz
    c3d ../../data/MNI-Poly-AMU_cord.nii.gz -resample 50% -o ../../data/MNI-Poly-AMU_cord.nii.gz

    echo ==============================================================================================
    echo Multiply voxels intensity
    echo ==============================================================================================
    fslmaths ../../data/MNI-Poly-AMU_T2.nii.gz -mul 16 ../../data/MNI-Poly-AMU_T2.nii.gz
    fslmaths ../../data/landmarks_center.nii.gz -mul 16 ../../data/landmarks_center.nii.gz
    fslmaths ../../data/MNI-Poly-AMU_cord.nii.gz -mul 16 ../../data/MNI-Poly-AMU_cord.nii.gz


    cmd="sct_register_to_template
        -i ../../data/${subject}/${contrast}/${contrast}.nii.gz
        -l ../../data/${subject}/${contrast}/${contrast}_landmarks_C2_T2_center.nii.gz
        -m ../../data/${subject}/${contrast}/${contrast}_segmentation_PropSeg.nii.gz
        -r 0
	-s superfast
	-o 1
	-t 1"
    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd

    cmd="sct_WarpImageMultiTransform
        3
        ${SCT_DIR}/data/template/MNI-Poly-AMU_cord.nii.gz
        templatecord2anat.nii.gz
        -R ../../data/${subject}/${contrast}/${contrast}.nii.gz 
        warp_template2anat.nii.gz
        --use-NN"
    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd

    cmd="sct_dice_coefficient
        ../../data/${subject}/${contrast}/${contrast}_manual_segmentation.nii.gz
        templatecord2anat.nii.gz
        -bmax"
    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd

    #cmd="rm ../../data/MNI-Poly-AMU_T2.nii.gz &&
    #    rm ../../data/landmarks_center.nii.gz &&
    #    rm ../../data/MNI-Poly-AMU_cord.nii.gz"
    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    #$cmd

  done
done

