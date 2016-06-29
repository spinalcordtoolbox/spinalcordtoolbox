#!/bin/bash
#
# Build and install the binaries of the spinal cord toolbox
#
# This script will compile (using CMake) the binaries of the spinal cord toolbox for the OS it is called.
# We assume here that this script is always in "install" folder of the toolbox

SCT_DIR_LOCAL=${PWD%/*}
echo ${SCT_DIR_LOCAL}

# program list
PROGRAM_LIST="isct_vesselness isct_propseg sct_change_nifti_pixel_type isct_crop_image sct_detect_spinalcord isct_dice_coefficient sct_hausdorff_distance sct_modif_header isct_orientation3d isct_bsplineapproximator" #

PATH_BIN_SCT=osx
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
  PATH_BIN_SCT=linux
fi

# loop across programs
for program in $PROGRAM_LIST; do
  
  echo
  cmd="cd ${SCT_DIR_LOCAL}/dev/${program}/"
  echo ">> $cmd"; $cmd

  echo
  cmd="rm ._*"
  echo ">> $cmd"; $cmd

  echo
  cmd="mkdir build"
  echo ">> $cmd"; $cmd

  echo
  cmd="cd build"
  echo ">> $cmd"; $cmd

  echo
  cmd="cmake .."
  echo ">> $cmd"; $cmd

  echo
  cmd="make -j 4"
  echo ">> $cmd"; $cmd

  echo
  cmd="cp ${program} ${SCT_DIR_LOCAL}/bin/"
  echo ">> $cmd"; $cmd

  echo
  cmd="make clean"
  echo ">> $cmd"; $cmd

  echo
  cmd="cd .."
  echo ">> $cmd"; $cmd

  echo
  cmd="rm -rf build"
  echo ">> $cmd"; $cmd


done
