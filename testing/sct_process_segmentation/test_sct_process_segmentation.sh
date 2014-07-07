#!/bin/bash
#
# register anatomical data (e.g. template) to multimodal data (e.g., MT, diffusion, fMRI, gradient echo). 


# subject list
SUBJECT_LIST="errsm_23"
CONTRAST_LIST="t2"
file="t2_segmentation_PropSeg.nii.gz"


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

    # loop across contrast
    for contrast in $CONTRAST_LIST; do

        # display subject
        echo
        printf "${green}Subject: $subject${NC}\n"
        printf "${red}Contrast: ${contrast}${NC}\n\n"

	    cmd="sct_process_segmentation.py
	        -i ${SCT_DIR}/testing/data/${subject}/${contrast}/${file}
	        -p compute_CSA
		-m counting_z_plane
		-s 1
		-r 0
	        -b 1
		-v 1"

        echo ==============================================================================================
        echo "$cmd"
        echo ==============================================================================================
        $cmd
    done
done
