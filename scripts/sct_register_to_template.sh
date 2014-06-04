#!/bin/bash
#
# Register anatomical image to the template using the spinal cord centerline/segmentation.
#
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# Created: 2014-04-28
# Modified: 2014-04-28
# read -p "Press ENTER key to continue..."
# Get a log file: sct_register_to_template.sh [options] | tee sct_register_to_template_$(date +%s).log


#==========================================================================#

function usage()
{
cat << EOF

`basename ${0}`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Register anatomical image to the template using the spinal cord centerline/segmentation.

USAGE
  `basename ${0}` -i <input> -l <landmarks> -m <segmentation>

MANDATORY ARGUMENTS
  -i <input>                   anatomical image
  -l <landmarks>               landmarks at spinal cord center
  -m <mask>                    spinal cord segmentation

OPTIONAL ARGUMENTS
  -o <output>                  0 (warp) | 1 (warp+images). Default=0.
  -s <speed>                   slow | normal | fast. Speed of registration. Slow give the best results. Default=fast.

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

# Compute elapsed time
time_start=$(date +%s)

# retrieve arguments
#scriptname=$0
#file_in=$1
#dimension=$2
folder_tmp="tmp.$(date +%Y%m%d%I%M%S)"

# find template folder from script folder
SCRIPTDIR=$(dirname $0)
TEMPLATEDIR="${SCRIPTDIR%"scripts"}data/template"

# Set the parameters
file_in=
file_landmarks=
file_mask=
output_images=0
speed="fast"
while getopts “hi:l:m:s:o:” OPTION
do
	case $OPTION in
		 h)
			usage
			exit 1
			;;
         i)
		 	file_in=$OPTARG
         	;;
         l)
         	 file_landmarks=$OPTARG
         	 ;;
         m)
             file_mask=$OPTARG
             ;;
         o)
             output_images=1
             ;;
         s)
             speed=$OPTARG
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
if [[ -z $file_landmarks ]]; then
     echo "ERROR: $file_landmarks does not exist. Exit program."
     exit 1
fi
if [[ -z $file_mask ]]; then
     echo "ERROR: $file_mask does not exist. Exit program."
     exit 1
fi
 
# get input extension
file_in_prefix=
file_in_name=
file_in_ext=
if [ ${file_in: -2} == "gz" ]; then
	file_in_ext=".nii.gz"
	export FSLOUTPUTTYPE='NIFTI_GZ'
    temp=${file_in%???????}
    file_in_prefix=${temp%/*}
    file_in_name=${temp##$file_in_prefix/}
elif [ ${file_in: -3} == "nii"]; then
	file_in_ext=".nii"
	export FSLOUTPUTTYPE='NIFTI'
    temp=${file_in%????}
    file_in_prefix=${temp%/*}
    file_in_name=${temp##$file_in_prefix/}
else
    echo "ERROR: Wrong input extension"
    exit 1
fi

file_landmarks_prefix=
file_landmarks_name=
file_landmarks_ext=
if [ ${file_landmarks: -2} == "gz" ]; then
    file_landmarks_ext=".nii.gz"
    export FSLOUTPUTTYPE='NIFTI_GZ'
    temp=${file_landmarks%???????}
    file_landmarks_prefix=${temp%/*}
    file_landmarks_name=${temp##$file_landmarks_prefix/}
elif [ ${file_landmarks: -3} == "nii"]; then
    file_landmarks_ext=".nii"
    export FSLOUTPUTTYPE='NIFTI'
    temp=${file_landmarks%????}
    file_landmarks_prefix=${temp%/*}
    file_landmarks_name=${temp##$file_landmarks_prefix/}
else
    echo "ERROR: Wrong input extension"
    exit 1
fi

file_mask_prefix=
file_mask_name=
file_mask_ext=
if [ ${file_mask: -2} == "gz" ]; then
    file_mask_ext=".nii.gz"
    export FSLOUTPUTTYPE='NIFTI_GZ'
    temp=${file_mask%???????}
    file_mask_prefix=${temp%/*}
    file_mask_name=${temp##$file_mask_prefix/}
elif [ ${file_mask: -3} == "nii"]; then
    file_mask_ext=".nii"
    export FSLOUTPUTTYPE='NIFTI'
    temp=${file_mask%????}
    file_mask_prefix=${temp%/*}
    file_mask_name=${temp##$file_mask_prefix/}

else
    echo "ERROR: Wrong input extension"
    exit 1
fi

# Check speed parameter and create registration mode: slow 50x30, normal 50x15, fast 10x3 (default)
nb_iteration="10x3"
if [ ${speed} == "slow" ]; then
    nb_iteration="50x30"
elif [ ${speed} == "normal" ]; then
    nb_iteration="50x15"
elif [ ${speed} == "fast" ]; then
    nb_iteration="10x3"
elif [ ${speed} == "superfast" ]; then
    nb_iteration="3x1"
else
    echo "ERROR: Wrong input registration speed {slow, normal, fast}."
    exit 1
fi


echo
echo "Check input arguments..."
echo "Input anat image filename = ${file_in_name}${file_in_ext}"
echo "Input landmarks image filename = ${file_landmarks_name}${file_landmarks_ext}"
echo "Input mask image filename = ${file_mask_name}${file_mask_ext}"
echo
echo "Registration speed = ${speed}"
echo


# create temp folder
echo "create temp folder..."
# check if temp folder exist - if so, delete it
if [ -e $folder_tmp ]; then
  cmd="rm -rf $folder_tmp"
  echo ">> $cmd"
  $cmd
fi
cmd="mkdir $folder_tmp"
echo ">> $cmd"
$cmd

# 1. Change orientation of input images into rpi
cmd="sct_orientation
-i ${file_in}
-o $folder_tmp/${file_in_name}_rpi${file_in_ext}
-orientation RPI"
echo ==============================================================================================
echo "$cmd"
echo ==============================================================================================
$cmd

cmd="sct_orientation
-i ${file_mask}
-o $folder_tmp/${file_mask_name}_rpi${file_mask_ext}
-orientation RPI"
echo ==============================================================================================
echo "$cmd"
echo ==============================================================================================
$cmd

cmd="sct_orientation
-i ${file_landmarks}
-o $folder_tmp/${file_landmarks_name}_rpi${file_landmarks_ext}
-orientation RPI"
echo ==============================================================================================
echo "$cmd"
echo ==============================================================================================
$cmd

# go to temp folder
cd $folder_tmp

# 2. Straighten the spinal cord using centerline/segmentation - output filename: inputfilename+_straight+ext
cmd="sct_straighten_spinalcord.py
-i ${file_in_name}_rpi${file_in_ext}
-c ${file_mask_name}_rpi${file_mask_ext}
-r 1"
echo ==============================================================================================
echo "$cmd"
echo ==============================================================================================
$cmd

# 3. Label preparation
# Remove unused label on template. Keep only label present in the input label image
cmd="sct_label_utils.py
-t remove
-i ${TEMPLATEDIR}/landmarks_center.nii.gz
-o template_label.nii.gz
-r ${file_landmarks_name}_rpi${file_landmarks_ext}"
#${file_landmarks}
echo ==============================================================================================
echo "$cmd"
echo ==============================================================================================
$cmd

# Create a cross for the template labels - 5 mm
cmd="sct_label_utils.py
-t cross
-i template_label.nii.gz
-o template_label_cross.nii.gz
-c 5"
echo ==============================================================================================
echo "$cmd"
echo ==============================================================================================
$cmd

# Create a cross for the input labels and dilate for straightening preparation - 5 mm
cmd="sct_label_utils.py
-t cross
-i ${file_landmarks_name}_rpi${file_landmarks_ext}
-o ${file_landmarks_name}_rpi_cross3x3${file_landmarks_ext}
-c 5
-d"
echo ==============================================================================================
echo "$cmd"
echo ==============================================================================================
$cmd

# Push the input labels in the template space
cmd="WarpImageMultiTransform
3
${file_landmarks_name}_rpi_cross3x3${file_landmarks_ext}
${file_landmarks_name}_rpi_cross3x3_straight${file_landmarks_ext}
-R ${file_in_name}_rpi_straight${file_in_ext}
warp_curve2straight.nii.gz
--use-NN"
echo ==============================================================================================
echo "$cmd"
echo ==============================================================================================
$cmd

pause

# 4. Registration of the straight spinal cord on the template - ${nb_iteration} slow 50x30, normal 50x15, fast 10x3

# straighten the segmentation
# TODO: when using the segmentation in the future, de-comment this
#cmd="WarpImageMultiTransform
#3
#$folder_tmp/${file_mask_name}_rpi${file_mask_ext}
#$folder_tmp/${file_mask_name}_rpi_straight${file_mask_ext}
#-R ${file_in_name}_rpi_straight${file_in_ext}
#warp_curve2straight.nii.gz
#--use-NN"
#echo ==============================================================================================
#echo "$cmd"
#echo ==============================================================================================
#$cmd

# registration of straight spinal cord to template
cmd="sct_register_straight_spinalcord_to_template.py
-i ${file_in_name}_rpi_straight${file_in_ext}
-l ${file_landmarks_name}_rpi_cross3x3_straight${file_landmarks_ext}
-t ${TEMPLATEDIR}/MNI-Poly-AMU_T2.nii.gz
-f template_label_cross.nii.gz
-m ${TEMPLATEDIR}/mask_gaussian_templatespace_sigma20.nii.gz
-r 1
-n ${nb_iteration}
-v 1"

echo ==============================================================================================
echo "$cmd"
echo ==============================================================================================
$cmd

# come back to original folder
cd ..

# 5. Compose warping fields - template to anat / anat to template
cmd="ComposeMultiTransform
3
warp_template2anat.nii.gz
-R ${file_in}
${folder_tmp}/warp_straight2curve.nii.gz
${folder_tmp}/warp_template2straight.nii.gz"
echo ==============================================================================================
echo "$cmd"
echo ==============================================================================================
$cmd

cmd="ComposeMultiTransform
3
warp_anat2template.nii.gz
-R ${TEMPLATEDIR}/MNI-Poly-AMU_T2.nii.gz
${folder_tmp}/warp_straight2template.nii.gz
${folder_tmp}/warp_curve2straight.nii.gz"
echo ==============================================================================================
echo "$cmd"
echo ==============================================================================================
$cmd

# 6. Application of warping fields on anat and template image
if [[ $output_images -eq 1 ]]
then
    cmd="WarpImageMultiTransform
        3
        ${TEMPLATEDIR}/MNI-Poly-AMU_T2.nii.gz
        template2anat.nii.gz
        -R ${file_in}
        warp_template2anat.nii.gz"
    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd

    cmd="WarpImageMultiTransform
        3
        ${file_in}
        anat2template.nii.gz
        -R ${TEMPLATEDIR}/MNI-Poly-AMU_T2.nii.gz
        warp_anat2template.nii.gz"
    echo ==============================================================================================
    echo "$cmd"
    echo ==============================================================================================
    $cmd
fi


# delete temp folder
echo
echo "delete temp folder..."
cmd="rm -rf $folder_tmp"
echo ">> $cmd"; $cmd

# display useful stuff
echo
echo "Done! Created file:"
echo "--> warp_template2anat.nii.gz"
echo "--> warp_anat2template.nii.gz"
if [[ $output_images -eq 1 ]]
then
    echo "--> template2anat.nii.gz"
    echo "--> anat2template.nii.gz"
fi
echo

# Compute and display elapsed time
time_end=$(date +%s)
elapsed_time="$(expr $time_end - $time_start)"
echo | awk -v D=$elapsed_time '{printf "Elapsed time: %02d:%02d:%02d\n",D/(60*60),D%(60*60)/60,D%60}'
