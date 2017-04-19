#!/bin/bash
#
# Example of commands to process multi-parametric data of the spinal cord
#
#### TO CHANGE
# For information about acquisition parameters, see: https://dl.dropboxusercontent.com/u/20592661/publications/Fonov_NIMG14_MNI-Poly-AMU.pdf
# N.B. The parameters are set for these type of data. With your data, parameters might be slightly different.
#
# To run without fslview output, type:
#   ./batch_processing_hawaii17.sh -nodisplay
#
# tested with v3.0_beta14 on 2016-07-16
#### TO CHANGE

# Check if display is on or off
if [[ $@ == *"-nodisplay"* ]]; then
  DISPLAY=false
  echo "Display mode turned off."
else
  DISPLAY=true
fi

# get SCT_DIR
source sct_env

# download example data
#sct_download_data -d sct_course_hawaii2017

# display starting time:
echo "Started at: $(date +%x_%r)"

# Go data folder
cd sct_course_hawaii17

# Spinal cord segmentation
# ===========================================================================================
# t2
cd sct_course_hawaii17/t2
sct_propseg -i t2.nii.gz -c t2
# Check results:
if [ $DISPLAY = true ]; then
  fslview t2 -b 0,800 t2_seg -l Red -t 0.5 t2_centerline_optic -l Blue &
fi

# t1
cd ../t1
sct_propseg -i t1.nii.gz -c t1
# Check results
if [ $DISPLAY = true ]; then
  fslview t1 -b 0,800 t1_seg -l Red -t 0.5 t1_centerline_optic -l Blue &
fi

# mt
cd ../mt
sct_propseg -i mt0.nii.gz -c t2
# Check results
if [ $DISPLAY = true ]; then
  fslview mt0.nii.gz mt0_seg.nii.gz -l Red -b 0,1 -t 0.5 mt0_centerline_optic -l Blue &
fi

# Registration to template
# ===========================================================================================
# t2
cd ../t2
# Label Vertebrae
sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2 -initcenter 7
# Check results
if [ $DISPLAY = true ]; then
  fslview t2.nii.gz t2_seg_labeled.nii.gz -l Random-Rainbow -t 0.5 &
fi

# Create Labels
sct_label_utils -i t2_seg_labeled.nii.gz -vert-body 3,9

# Register to template
sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -l labels.nii.gz
# Check results
if [ $DISPLAY = true ]; then
  fslview t2.nii.gz -b 0,800 template2anat -b 0,4000 &
fi

# Warp template objects
sct_warp_template -d t2.nii.gz -w warp_template2anat.nii.gz












