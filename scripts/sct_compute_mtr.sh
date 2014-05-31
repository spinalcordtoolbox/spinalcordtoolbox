#!/bin/bash
#
# Register MTC0 on MTC1 and compute MTR
# 
# Dependence: FSL, ANTS
#
# Author: julien cohen-adad
# 2014-05-03



#==========================================================================#

function usage()
{
cat << EOF

`basename ${0}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Register image without (MTC0) and with magnetization transfer contrast (MTC1) and compute MTR.

USAGE
  `basename ${0}` -i <MTC0> -d <MTC1>

MANDATORY ARGUMENTS
  -i <MTC0>                    MTC0 image.
  -d <MTC1>                    MTC1 image.

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

#==========================================================================#

# Set the parameters
scriptname=$0
file_mt0=
file_mt1=
file_mtr='mtr'
while getopts “hi:d:” OPTION
do
	case $OPTION in
		 h)
			usage
			exit 1
			;;
         i)
		 	file_mt0=$OPTARG
         	;;
         d)
			file_mt1=$OPTARG
         	;;
         ?)
             usage
             exit
             ;;
     esac
done

# Check the parameters
if [[ -z $file_mt0 ]]; then
	 echo "ERROR: $file_mt0 does not exist. Exit program."
     exit 1
fi
if [[ -z $file_mt1 ]]; then
     echo "ERROR: $file_mt1 does not exist. Exit program."
     exit 1
fi

# register MTC0 on MTC1
cmd="sct_register_multimodal.py -i ${file_mt0} -d ${file_mt1}"
echo ">> $cmd"; $cmd

# retrieve file name
# TODO: find a better way to do it
file_mt0_reg=`ls *_reg.*`

# compute MTR
cmd="fslmaths -dt double ${file_mt0_reg} -sub ${file_mt1} -mul 100 -div ${file_mt0_reg} -thr 0 -uthr 100 ${file_mtr}"
echo ">> $cmd"; $cmd

# display results
echo Done! To look at the results type this:
echo "fslview ${file_mtr} ${file_mt1}  ${file_mt0_reg} ${file_mt0} &"

exit 0
