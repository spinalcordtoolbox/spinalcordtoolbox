#!/bin/bash
#
# register anatomical data (e.g. template) to multimodal data (e.g., MT, diffusion, fMRI, gradient echo). 


# subject list
SUBJECT_LIST="errsm_23"
CONTRAST_LIST="mt"
file="mtc1.nii.gz"


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

#        # register T2
#        cmd="sct_register_multimodal.py
#            -i ../../data/${subject}/t2/t2.nii.gz
#            -d ../../data/${subject}/${contrast}/${file}
#            -s ../../data/${subject}/t2/manual_segmentation.nii.gz
#            -t ../../data/${subject}/${contrast}/segmentation_binary.nii.gz
#            -o ${subject}_t2_reg_maskT2.nii.gz
#            -n 50x20
#            -r 0
#            -v 1"

	    # register template
	    cmd="sct_register_multimodal.py
	        -i ${SCT_DIR}/data/template/MNI-Poly-AMU_T2.nii.gz
	        -d ../../data/${subject}/${contrast}/${file}
	        -s ${SCT_DIR}/data/template/MNI-Poly-AMU_cord.nii.gz
	        -t ../../data/${subject}/${contrast}/segmentation_binary.nii.gz
	        -q ../../data/${subject}/template/warp_template2anat.nii.gz
	        -o template2${file}
	        -n 50x20
	        -r 1
	        -v 1"

        echo ==============================================================================================
        echo "$cmd"
        echo ==============================================================================================
        $cmd
    done
done
