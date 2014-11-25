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
  Create soft links for fsl binaries.

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
PATHFSL=$(cd $(dirname "$0"); which fslview)
PATHFSL=$(dirname ${PATHFSL})
echo "Create soft link to each fsl binaries in ${PATHFSL}"
prefix="fsl5.0-"

cd ${PATHFSL}
for binary in *
do
  echo ${binary}
  cd ${SCT_DIR}/bin
  scriptname=${binary#$prefix}
  cmd=
  if [ "$is_admin" = true ] ; then
    cmd="sudo ln -s $PATHFSL/${binary} ${SCT_DIR}/bin/${scriptname}"
  else
    cmd="ln -s $PATHFSL/${binary} ${SCT_DIR}/bin/${scriptname}"
  fi
  echo ">> $cmd"
  $cmd
done

cd ${CURRENT_DIR}
