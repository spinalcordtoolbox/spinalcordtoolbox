#!/bin/bash
# julien cohen-adad
# 2013-07-09
#
# requires data to be: data.nii.gz, bvals.txt, bvecs.txt
#
# TODO: make a python version of it


#FOLDER_OUTPUT="results/"

# create one folder
#echo create folder /results
#if [ ! -e "results" ]; then
#  mkdir ${FOLDER_OUTPUT}
#fi

# crop data
#echo Crop data...
#fslroi data ${FOLDER_OUTPUT}data_crop 90 70 100 50 0 11

# extract first five b=0
echo Extract first b=0 image...
fslroi dmri b0 0 1

# Average b=0
#echo Average b=0 images...
#fslmaths ${FOLDER_OUTPUT}b0 -Tmean ${FOLDER_OUTPUT}b0_mean

# estimate tensors
echo Estimate tensors...
dtifit -k dmri -m b0 -o dti -r bvecs.txt -b bvals.txt 

# display useful stuff
echo Done! To look at the results type this:
echo "fslview b0 dti_FA dti_MD dti_V1 &"


