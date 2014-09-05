#!/bin/bash
#
# register anatomical data (e.g. template) to multimodal data (e.g., MT, diffusion, fMRI, gradient echo). 


# subject list
SUBJECT_LIST="errsm_23"


# START BATCH HERE
# =================================

# define nice colors
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

    # display subject
    echo
    printf "${green}Subject: $subject${NC}\n"

    cmd="sct_label_utils
        -i ../../data/${subject}/t2/landmarks_rpi.nii.gz
        -t cross
        -o landmarks_rpi_cross3x3.nii.gz
        -c 5
        -d"

    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd
done
