#!/bin/bash
#
# test sct_segmentation_propagation
# This program launch the testing of automatic spinal cord segmentation on five subjects (in data folder)
# Results can be validated using FSLVIEW or MITKWorkbench
#
# Common values of Dice coefficient when accurate segmentation: [0.85, 0.93]

# subject list
SUBJECT_LIST="errsm_23" # 
CONTRAST_LIST="t2" # t1 or t2

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
        cmd="sct_propseg
            -i ../../data/${subject}/${contrast}/${contrast}.nii.gz
            -t ${contrast}
            -mesh
            -cross
            -centerline-binary
            -verbose"
        echo ==============================================================================================
        echo "$cmd"
        echo ==============================================================================================
        $cmd

        cmd="sct_dice_coefficient
            ../../data/${subject}/${contrast}/${contrast}_manual_segmentation.nii.gz
            segmentation_binary.nii.gz
            -bmax"
        echo ==============================================================================================
        echo "$cmd"
        echo ==============================================================================================
        $cmd
    done
done
