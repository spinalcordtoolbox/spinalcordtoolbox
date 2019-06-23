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
SITE=$2
PATH_OUTPUT=$3
PATH_QC=$4
PATH_LOG=$5

PATH_RESULTS=$PATH_OUTPUT/$SITE

cd $SUBJECT/anat

for file in *.nii.gz
do
  if [[ $file == *"_seg"* ]];then #file is a seg
    continue
  else # file is an image
    echo " ======> Processing $file file from site $SITE"
    file_seg="${file%.nii.gz*}_seg.nii.gz"
    file_seg_manual="${file%.nii.gz*}_seg_manual.nii.gz"
    if [ -e $file_seg ]; then #segmentation of this file exist
      echo "======> Processing file $file with seg $file_seg"
      evaluate_rot -i $file -iseg $file_seg -o $PATH_RESULTS -qc $PATH_QC
    elif [ -e $file_seg_manual ]; then #manual segmentation of this file exist
      echo "======> Processing file $file with manual seg $file_seg_manual"
      evaluate_rot -i $file -iseg $file_seg_manual -o $PATH_RESULTS -qc $PATH_QC
    else
      echo "======> Segmentation for file $file does not exist, segmenting with sct_deepseg_sc"
      if [[ $file == *"T1w"* ]]; then
        contrast="t1"
      elif [[ $file == *"T2w"* ]]; then
        contrast="t2"
      elif [[ $file == *"T2s"* ]]; then
        contrast="t2s"
      else
        echo "======> Contrast for file $file not found or not supported"
        continue
      fi
      sct_deepseg_sc -i $file -c $contrast -ofolder $PATH_RESULTS
      echo "======> evaluate_reg with file $file and seg ${PATH_RESULTS}/${file_seg##*/}"
      evaluate_rot -i $file -iseg "${PATH_RESULTS}/${file_seg##*/}" -o $PATH_RESULTS -qc $PATH_QC
    fi
    echo "evaluate_rot done"
  fi
done
