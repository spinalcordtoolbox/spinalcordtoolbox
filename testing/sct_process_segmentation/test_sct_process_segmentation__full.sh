#!/bin/bash
#
# test sct_process_segmentation.
# This is a full test, which uses data from spinalcordtoolbox_data. 


# subject list
SUBJECT_LIST="errsm_23"
CONTRAST_LIST="t2"
FILE_LIST="t2_segmentation_PropSeg.nii.gz t2_segmentation_PropSeg_crop190-220.nii.gz t2_segmentation_PropSeg_Ysubsampled5.nii.gz"
METHOD_LIST="counting_z_plane counting_ortho_plane ellipse_z_plane ellipse_ortho_plane"

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

		for file in $FILE_LIST; do
			
			mkdir $file
			cd $file
			
			for method in $METHOD_LIST; do
		        # display subject
		        echo
		        printf "${green}Subject: $subject${NC}\n"
		        printf "${red}Contrast: ${contrast}${NC}\n\n"

				mkdir $method
				cd $method
			
			    cmd="sct_process_segmentation.py
	                -i ${SCT_DATA_DIR}/${subject}/${contrast}/$file
	                -p compute_CSA
	                -m counting_z_plane
	                -s 1
	                -r 0
	                -b 1
	                -f 1
	                -v 1"

		        echo ==============================================================================================
		        echo "$cmd"
		        echo ==============================================================================================
		        $cmd
				echo 
			
				# come back to parent folder
				cd ..
			done
			cd ..
		done
    done
done
