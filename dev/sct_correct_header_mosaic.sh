#!/bin/bash
#
# When using the 2DRF diffusion sequence from Jürgen Finsterbusch, the DICOM information regarding the origin of the coordinate is incorrect, yielding wrong header when converting to NIFTI. This script corrects the NIFTI header.
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-04-13


#==========================================================================#

function usage() {
    echo usage:
    echo `basename ${0}` 'file_in'
}

function askhelp() {
    echo help!
}

if [ ! ${#@} -gt 0 ]; then
    usage `basename ${0}`
    exit 1
fi


#--------------------------------------------------------------------------#

# retrieve arguments
scriptname=$0
file_in=$1
folder_tmp="tmp.correctheader"

# create temp folder
echo
echo "create temp folder..."
# check if temp folder exist - if so, delete it
if [[ -e $folder_tmp ]]; then
  cmd="rm -rf $folder_tmp"
  echo ">> $cmd"; $cmd
fi
cmd="mkdir $folder_tmp"
echo ">> $cmd"; $cmd

# split data along t (4D --> 3D)
echo
echo "split data along t (4D --> 3D)..."
cmd="fslsplit $file_in $folder_tmp/dmri_splitT -t"
echo ">> $cmd"; $cmd;

# go to temp folder
cd $folder_tmp

# change orientation to RPS
echo
echo "change orientation to RPS for each file..."
FILES=`ls dmri_splitT*.*`
for file in $FILES; do
  cmd="sct_orientation -i $file -o rps_$file -orientation RPS"
  echo ">> $cmd"; $cmd;
done

# create matrix file
echo
echo "create matrix file..."
echo "1 0 0 0" > matrix.txt # create file
echo "0 1 0 0" >> matrix.txt # append
echo "0 0 -1 0" >> matrix.txt
echo "0 0 0 1" >> matrix.txt
more matrix.txt

# multiplies orientation by matrix
echo
echo "multiplies orientation by matrix..."
FILES=`ls rps_dmri_splitT*.*`
for file in $FILES; do
  cmd="sct_modif_header $file -mat matrix.txt -o correct_$file"
  echo ">> $cmd"; $cmd;
done

echo
echo "merge back into 4D file..."
cmd="fslmerge -t ../correct_$file_in correct_*.*"
echo ">> $cmd"; $cmd;


# come back to original folder
cd ..

# delete temp folder
echo
echo "delete temp folder..."
cmd="rm -rf $folder_tmp"
echo ">> $cmd"; $cmd

# display useful stuff
echo
echo "Done! Created file:"
echo "--> correct_$file_in"
echo
