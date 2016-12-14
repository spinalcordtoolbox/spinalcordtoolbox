#!/bin/bash
#
# Flip data in a specified direction. Note: this will NOT change the header, but it will change the way the data are stored.
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-04-18


#==========================================================================#

function usage()
{
cat << EOF

`basename ${0}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Flip data in a specified dimension (x,y,z or t).
  N.B. This script will NOT modify the header but the way the data are stored (so be careful!!).

USAGE
  `basename ${0}` -i <input> -d <x|y|z|t>

MANDATORY ARGUMENTS
  -i <input>                   image
  -d <dimension>               dimension: x|y|z|t

EOF
}

function askhelp()
{
    echo help!
}

if [ ! ${#@} -gt 0 ]; then
    usage `basename ${0}`
    exit 1
fi


#--------------------------------------------------------------------------#

# retrieve arguments
#scriptname=$0
#file_in=$1
#dimension=$2
folder_tmp="tmp.flipdata"

# Set the parameters
file_in=
dimension=
while getopts “hi:d:” OPTION
do
	case $OPTION in
		h)
			usage
			exit 1
			;;
         i)
		 	file_in=$OPTARG
         	;;
         d)
         	 dimension=$OPTARG
         	 ;;
         ?)
             usage
             exit
             ;;
     esac
done

# Check the parameters
if [[ -z $file_in ]]; then
	 echo "ERROR: $file_in does not exist. Exit program."
     exit 1
fi
if [[ -z $dimension ]]; then
	 usage
     exit 1
fi
 
# get input extension
if [ ${file_in: -2} == "gz" ]; then
	ext=".nii.gz"
	export FSLOUTPUTTYPE='NIFTI_GZ'
	prefix=${file_in%???????}
	file_out="${prefix}_flip.nii.gz"
elif [ ${file_in: -3} == "nii"]; then
	ext=".nii"
	export FSLOUTPUTTYPE='NIFTI'
	prefix=${file_in%????}
	file_out="${prefix}_flip.nii"
else
    echo "ERROR: Wrong input extension"
    exit 1
fi

# create temp folder
echo
echo "create temp folder..."
# check if temp folder exist - if so, delete it
if [ -e $folder_tmp ]; then
  cmd="rm -rf $folder_tmp"
  echo ">> $cmd"; $cmd
fi
cmd="mkdir $folder_tmp"
echo ">> $cmd"; $cmd

# split data
echo
echo "split data along $dimension..."
cmd="fslsplit $file_in ${folder_tmp}/data_split -$dimension"
echo ">> $cmd"; $cmd;

# go to temp folder
cd $folder_tmp

# concatenate in reverse order
echo
echo "concatenate in reverse order..."
FILES=`ls -r data_split*.*`
cmd="fslmerge -${dimension} ../${file_out}"
for file in $FILES; do
  cmd="$cmd ${file}"
done
echo ">> $cmd"; $cmd;

# come back to original folder
cd ..

# copy geometry
echo
echo "copy geometry..."
cmd="fslcpgeom ${file_in} ${file_out}"
echo ">> $cmd"; $cmd;

# delete temp folder
echo
echo "delete temp folder..."
cmd="rm -rf $folder_tmp"
echo ">> $cmd"; $cmd

# display useful stuff
echo
echo "Done! Created file:"
echo "--> ${file_out}"
echo
