#!/bin/bash
#
# Script to create links in installer
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# Modified: 2014-09-03


#==========================================================================#

function usage()
{
cat << EOF

`basename ${0}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Create soft links for python scripts.

USAGE
  `basename ${0}` -a

OPTIONAL ARGUMENTS
  -a    disable administrator rights on links creation

EOF
}

function askhelp()
{
    echo help!
}

is_admin=true
while getopts “ha” OPTION
do
    case $OPTION in
        h)
            usage
            exit 1
            ;;
        a)
            is_admin=false
            ;;
        ?)
            usage
            exit
            ;;
    esac
done

CURRENT_DIR=$PWD

# create soft link to each script in SCT_DIR/script
echo "Create soft link for each python script located in $SCT_DIR/script"
suffix_py='.py'
cd $SCT_DIR/scripts

for script in *.py
do
  echo ${script}
  cd ${SCT_DIR}/bin
  scriptname=${script%$suffix_py}
  cmd=
  if [ "$is_admin" = true ] ; then
    cmd="sudo ln -sf ../scripts/${script} ${scriptname}"
  else
    cmd="ln -sf ../scripts/${script} ${scriptname}"
  fi
  echo ">> $cmd"
  $cmd
done

#removing internal stuff or stuff under development
cmd="sudo rm sct_nurbs sct_segment_graymatter sct_utils sct_dmri_eddy_correct isct_check_detection sct_change_image_type sct_invert_image"
$cmd

cd ${CURRENT_DIR}
