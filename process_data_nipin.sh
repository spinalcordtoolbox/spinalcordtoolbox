#!/bin/bash
#
# This script BLABLABLA
#
# NB: add the flag "-x" after "!/bin/bash" for full verbose of commands.
# Nicolas Pinon 10/04/2019

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1
PATH_RESULTS=$2
PATH_QC=$3
PATH_LOG=$4


cd anat/

for file in *.nii.gz
do
  if [[ $file == *"_seg"* ]];then #file is a seg
    continue
  else # file is an image
  echo "Processing $file file.."
  file_seg="${file%.nii.gz*}_seg.nii.gz"
  file_seg_manual="${file%.nii.gz*}_seg_manual.nii.gz"
    if [ -e $file_seg ]; then #segmentation of this file exist
      echo "   Processing file $file with seg $file_seg"
      evaluate_reg_rot -i $file -iseg $file_seg -o $PATH_RESULTS
    elif [ -e $file_seg ]; then
      echo "   Processing file $file with seg $file_seg_manual"
      evaluate_reg_rot -i $file -iseg $file_seg -o $PATH_RESULTS
    else
      echo "   Segmentation for file $file does not exist"
      continue
    fi
  fi
done
