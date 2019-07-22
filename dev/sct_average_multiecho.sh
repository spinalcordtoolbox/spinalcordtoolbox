#!/bin/bash
#
# convert to nifti and average multi-echo data together
# 
# julien cohen-adad <jcohen@polymtl.ca>
# 2013-07-12


#==========================================================================#

function usage() {
    echo usage:
    echo `basename ${0}` '<path_data>'
}

function askhelp() {
    echo help!
}

if [[ ! ${#@} -gt 0 ]]; then
    usage `basename ${0}`
    exit 1
fi


#--------------------------------------------------------------------------#

scriptname=$0
path_data=$1

# get all subfolders
subfolder=`ls ${path_data}`

# check if temp folder exist
if [[ -e "tmp.data" ]]; then
  cmd='rm -rf tmp.data'
  echo ">> $cmd"; $cmd
fi

# create temp folder
cmd='mkdir tmp.data'
echo ">> $cmd"; $cmd

# loop across subfolder
for ifolder in $subfolder; do
  echo $ifolder

  # convert to nifti
  cmd="dcm2nii -o ./tmp.data ${path_data}/$ifolder"
  echo ">> $cmd"; $cmd
done

# merge into 4d file
cmd='fslmerge -t tmp.data/data4d tmp.data/*.*'
echo ">> $cmd"; $cmd

# average in time
cmd='fslmaths tmp.data/data4d -Tmean data4d_mean'
echo ">> $cmd"; $cmd

# delete temp folder
cmd='rm -rf tmp.data'
echo ">> $cmd"; $cmd

exit 0

