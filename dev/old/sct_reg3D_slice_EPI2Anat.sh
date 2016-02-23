#!/bin/sh

usage()
{
cat << EOF
usage: $0 -i input_image -s source_image -o output_path -m mask

This script registers an nifti image into another one (typically the anatomical one)

DEPENDENCIES:
- FSL
- ANTs
- sct_c3d

MANDATORY INPUTS:
   -i	   Path of the input anatomical file
   -s	   Path of the source file to register

OPTIONAL INPUTS:
   -h      		   Show this message
   -o			   Output path where a directory with all created files by the script are stored [default: directory of the source file]
   -m [mask]       Mask the spinal cord (improves robustness). It uses the same mask for the source and destination.
   
EOF
}

# Initialization
CROPPING=
PATH_CENTERLINE=
MASKING=
PATH_SURFACE=
SMOOTHING=
PATH_ANAT=
PATH_TO_REGISTER=
METHOD=ants
PATH_DIRECTORY=
# Set the parameters
while getopts “hm:i:s:o:” OPTION
do
     case $OPTION in
         h)
             usage
             exit 1
             ;;
         m)
             MASKING=1
             PATH_SURFACE=$OPTARG
             ;;
         i)
         	 PATH_ANAT=$OPTARG
         	 ;;
         s)
         	 PATH_TO_REGISTER=$OPTARG
         	 PATH_DIRECTORY=$PATH_TO_REGISTER
         	 ;;
         o)
             PATH_DIRECTORY=$OPTARG
             ;;
         ?)
             usage
             exit
             ;;
     esac
done

# Check the parameters
if [[ -z $PATH_ANAT ]] || [[ -z $PATH_TO_REGISTER ]]
then
	 echo "Wrong path for the images"
     usage
     exit 1
fi

if [[ $MASKING -eq 1 ]] && [[ -z $PATH_SURFACE ]]
then
	 echo "Wrong path for the mask"
     usage
     exit 1
fi

#--------------------------------------------
# Filename and path of the anatomical file
#--------------------------------------------
if [[ "${PATH_ANAT##*.}" == "gz" ]] || [[ -e "${PATH_ANAT}.nii.gz" ]]
then
	EXT_ANAT=nii.gz
elif [[ "${PATH_ANAT##*.}" == "nii" ]] || [[ -e "${PATH_ANAT}.nii" ]]
then
	EXT_ANAT=nii	
fi

#Get the path and the extension
FILE_ANAT=${PATH_ANAT%"."$EXT_ANAT}
FILE_ANAT=${FILE_ANAT##*/}
PATH_ANAT=${PATH_ANAT%"."$EXT_ANAT}
PATH_ANAT=${PATH_ANAT%$FILE_ANAT}


if [[ -z $PATH_ANAT ]]
then
	PATH_ANAT="./"
fi
#--------------------------------------------

#--------------------------------------------
# Filename and path of the file to register
#--------------------------------------------
if [[ "${PATH_TO_REGISTER##*.}" == "gz" ]] || [[ -e "${PATH_TO_REGISTER}.nii.gz" ]]
then
	EXT_REG=nii.gz
elif [[ "${PATH_TO_REGISTER##*.}" == "nii" ]] || [[ -e "${PATH_TO_REGISTER}.nii" ]]
then
	EXT_REG=nii	
fi

#Get the path and the extension
FILE_TO_REGISTER=${PATH_TO_REGISTER%"."$EXT_REG}
FILE_TO_REGISTER=${FILE_TO_REGISTER##*/}
PATH_TO_REGISTER=${PATH_TO_REGISTER%"."$EXT_REG}
PATH_TO_REGISTER=${PATH_TO_REGISTER%$FILE_TO_REGISTER}

if [[ -z $PATH_TO_REGISTER ]]
then
	PATH_TO_REGISTER="./"
fi
#--------------------------------------------

#--------------------------------------------
# Filename and path of the surface
#--------------------------------------------
SURFACE=
if [[ -e $PATH_SURFACE ]]
then
	if [[ "${PATH_SURFACE##*.}" == "gz" ]] || [[ -e "${PATH_SURFACE}.nii.gz" ]]
	then
		EXT_SURF=nii.gz
	elif [[ "${PATH_SURFACE##*.}" == "nii" ]] || [[ -e "${PATH_SURFACE}.nii" ]]
	then
		EXT_SURF=nii	
	fi

	#Get the path and the extension
	SURFACE=${PATH_SURFACE%"."$EXT_SURF}
	SURFACE=${SURFACE##*/}
	PATH_SURFACE=${PATH_SURFACE%"."$EXT_SURF}
	PATH_SURFACE=${PATH_SURFACE%$SURFACE}

	if [[ -z $PATH_SURFACE ]]
	then
		PATH_SURFACE="./"
	fi
fi
#--------------------------------------------


#--------------------------------------------
# Format of the script is .nii
# FSL export
echo ">> From now, the script will work only with FSLOUTPUTTYPE='NIFTI'"
FSLOUTPUT_INIT=$(echo $FSLOUTPUTTYPE)
PREFIX_NIFTI="export FSLOUTPUTTYPE='NIFTI';"
EXT=nii
#--------------------------------------------


#Define experiment specific parameters
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Timer
start=$SECONDS

#specify folder names
NAME_OTHER_MODALITY=diff

#specify parameters for antsIntroduction.sh
#compulsory arguments
ImageDimension=3

#optional arguments
MaxIteration=10x10x10
MetricType='MI'

#sct_c3d
INTERP=nearestneighbor

#Initialization of the suffixes
SUFFIX_ANAT=
SUFFIX_OTHER=

#Smoothing
KernelGauss='3.5x3.5x0'
Sigma='2x2x0'

#The script starts here
#~~~~~~~~~~~~~~~~~~~~~
PATH_DIRECTORY=${PATH_DIRECTORY%/*}
PATH_OUTPUT=${PATH_DIRECTORY}/results_${NAME_OTHER_MODALITY}
mkdir ${PATH_OUTPUT}

FILE_DEST=${PATH_TO_REGISTER}${FILE_TO_REGISTER}
FILE_SRC=${PATH_ANAT}${FILE_ANAT}

#----------------------------------------
# STEP ONE: anat into the other modality
#----------------------------------------

# resample anat_space into mtON
echo "Putting source image into space of destination image..."
echo ">> sct_c3d ${FILE_DEST}.${EXT_REG} ${FILE_SRC}.${EXT_ANAT} -interpolation ${INTERP} -reslice-identity -o ${PATH_OUTPUT}/${FILE_ANAT}_resampled.${EXT}"
sct_c3d ${FILE_DEST}.${EXT_REG} ${FILE_SRC}.${EXT_ANAT} -interpolation ${INTERP} -reslice-identity -o ${PATH_OUTPUT}/${FILE_ANAT}_resampled.${EXT}
SUFFIX_ANAT=${SUFFIX_ANAT}_resampled

FILE_DEST=${PATH_OUTPUT}/${FILE_ANAT}${SUFFIX_ANAT}
FILE_SRC=${PATH_TO_REGISTER}${FILE_TO_REGISTER}${SUFFIX_OTHER}
echo "\n\n\n\n"
EXT_ANAT=nii

#----------------------------------------
# STEP ZERO-ZERO: masking the anatomical file
#----------------------------------------

if [[ "$MASKING" -eq 1 ]]
then
	echo ">> Masking ${PATH_ANAT}${FILE_ANAT}${SUFFIX_ANAT}.${EXT_ANAT} ..."

	cmd="sct_c3d ${FILE_DEST}.${EXT_ANAT} ${PATH_SURFACE}${SURFACE}.${EXT_SURF} -interpolation ${INTERP} -reslice-identity -o ${PATH_OUTPUT}/${SURFACE}_resliced.${EXT_SURF}"
	echo $cmd
	eval $cmd

	cmd="${PREFIX_NIFTI} fslmaths ${PATH_OUTPUT}/${SURFACE}_resliced.${EXT_SURF} -kernel gauss $KernelGauss -dilM -s $Sigma ${PATH_OUTPUT}/${SURFACE}_mask.${EXT}"
	echo $cmd
	eval $cmd

	cmd="${PREFIX_NIFTI} fslmaths ${FILE_DEST}.${EXT} -mul ${PATH_OUTPUT}/${SURFACE}_mask.${EXT} ${PATH_OUTPUT}/${FILE_ANAT}${SUFFIX_ANAT}_masked.${EXT}"
	echo $cmd
	eval $cmd

	echo ">> Masking ${PATH_TO_REGISTER}${FILE_TO_REGISTER}${SUFFIX_OTHER}.${EXT_OTHER} ..."

	cmd="${PREFIX_NIFTI} fslmaths ${FILE_SRC}.${EXT_REG} -mul ${PATH_OUTPUT}/${SURFACE}_mask.${EXT} ${PATH_OUTPUT}/${FILE_TO_REGISTER}${SUFFIX_OTHER}_masked.${EXT}"
	echo $cmd
	eval $cmd


	SUFFIX_ANAT=${SUFFIX_ANAT}_masked
	SUFFIX_OTHER=${SUFFIX_OTHER}_masked

	EXT_ANAT=nii
	EXT_OTHER=nii

	FILE_DEST=${PATH_OUTPUT}/${FILE_TO_REGISTER}${SUFFIX_OTHER}
	FILE_SRC=${PATH_OUTPUT}/${FILE_ANAT}${SUFFIX_ANAT}
	echo "\n\n\n\n"
else
	SUFFIX_ANAT=${SUFFIX_ANAT}
	SUFFIX_OTHER=${SUFFIX_OTHER}
	FILE_SRC=${FILE_SRC}
	FILE_DEST=${FILE_DEST}
fi

#---------------------


#--Restrict-Deformation 1x0x0 
cmd="ants $ImageDimension -m ${MetricType}[${FILE_DEST}.${EXT_ANAT},${FILE_SRC}.${EXT},1,4] --Restrict-Deformation 0x1x0 -t SyN -r Gauss[6,1] -o ${PATH_OUTPUT}/ -i ${MaxIteration}"
# --number-of-affine-iterations 10000x5000x1000 --rigid-affine true"
echo $cmd
eval $cmd

if [[ "$MASKING" -eq 1 ]]
then
	FILE_DEST=${PATH_OUTPUT}/${FILE_ANAT}_resampled
	FILE_SRC=${PATH_TO_REGISTER}${FILE_TO_REGISTER}
fi

# apply transformation
#${PATH_OUTPUT}/Affine.txt
cmd="WarpImageMultiTransform $ImageDimension ${FILE_SRC}.${EXT_REG} ${PATH_OUTPUT}/${FILE_TO_REGISTER}_reg.${EXT} ${PATH_OUTPUT}/Warp.nii.gz  -R $FILE_DEST.${EXT_ANAT} --use-BSpline"
echo $cmd
eval $cmd


# cmd="isct_antsRegistration -d $ImageDimension -r [${FILE_DEST}.${EXT_ANAT},${FILE_SRC}.${EXT} ,1] -m ${MetricType}[${FILE_DEST}.${EXT_ANAT},${FILE_SRC}.${EXT},1,4] --use-histogram-matching 1 -o ${PATH_OUTPUT}/ -t Affine -c [10000x10000x10000,1.e-8,20] -s 4x2x1vox -f 3x2x1"
# echo $cmd
# eval $cmd
# 
# cmd="ConvertTransformFile $ImageDimension ${PATH_OUTPUT}/0GenericAffine.mat ${PATH_OUTPUT}/Affine.txt"
# echo $cmd
# eval $cmd
# 
# cmd="antsApplyTransforms -d $ImageDimension -i ${FILE_SRC}.${EXT} -r ${FILE_DEST}.${EXT_ANAT} -t ${PATH_OUTPUT}/Affine.txt -n BSpline -o ${FILE_SRC}_reg.${EXT}"
# echo $cmd
# eval $cmd






echo "sct_c3d ${PATH_ANAT}${FILE_ANAT}.${EXT_ANAT} ${PATH_OUTPUT}/resliced.${EXT} -interpolation ${INTERP} -reslice-identity -o ${PATH_OUTPUT}/final.${EXT}"
sct_c3d ${PATH_ANAT}${FILE_ANAT}.${EXT_ANAT} ${PATH_OUTPUT}/${FILE_TO_REGISTER}_reg.${EXT} -interpolation ${INTERP} -reslice-identity -o ${PATH_OUTPUT}/final.${EXT}


#---------------------

# mkdir ${PATH_TO_REGISTER}Modified_Images
# mv ${FILE_ANAT}_* ${PATH_TO_REGISTER}Modified_Images
# if [[ "$MASKING" -eq 1 ]]
# then
# 	mv ${FILE_TO_REGISTER}_* ${PATH_TO_REGISTER}Modified_Images/
# fi
#-----
PREFIX_NIFTI="export FSLOUTPUTTYPE='${FSLOUTPUT_INIT}';"
eval ${PREFIX_NIFTI}

#----------

echo "All files are stored in the following directory:"
echo ">> ${PATH_OUTPUT}/"
echo "\nHere is the final image registered:"
echo ">> ${FILE_SRC}_reg.${EXT}"


echo "\nDone in $(( SECONDS - start )) seconds!"


