#!/bin/sh

usage()
{
cat << EOF
usage: $0 -i input_image -s source_image -r method -c centerline -m mask -g value

This script registers an nifti image into another one (typically the anatomical one)

DEPENDENCIES:
- FSL
- ANTs --> only if you choose the option "-r ants"
- c3d
- sct_cropping.m (optional)

MANDATORY INPUTS:
   -i	   Path of the input anatomical file
   -s	   Path of the source file to register

OPTIONAL INPUTS:
   -h      		   Show this message
   -r [method]	   Method of registration: 'ants' or 'flirt' (default 'ants')
   -m [mask]       Mask the spinal cord (improves robustness). It uses the same mask for the source and destination.
   
EOF
}

# Initialization
MASKING=
PATH_SURFACE=
PATH_ANAT=
PATH_TO_REGISTER=
METHOD=ants
# Set the parameters
while getopts “hm:g:i:s:r:” OPTION
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
         	 ;;
         r)
         	 METHOD=$OPTARG
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
	 echo "Wrong path for the surface"
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

#specify folder names
NAME_OTHER_MODALITY=diffusion_mean_affine

#specify parameters for antsIntroduction.sh
#compulsory arguments
ImageDimension=2

#optional arguments
MaxIteration=1x0x0
MetricType='MI'

#c3d
INTERP=nearestneighbor

#Initialization of the suffixes
SUFFIX_ANAT=
SUFFIX_OTHER=

#Smoothing
KernelGauss='7.5x7.5x0'
Sigma='5x5x0'

#The script starts here
#~~~~~~~~~~~~~~~~~~~~~

FILE_DEST=${PATH_TO_REGISTER}${FILE_TO_REGISTER}
FILE_SRC=${PATH_ANAT}${FILE_ANAT}

#----------------------------------------
# STEP ONE: anat into the other modality
#----------------------------------------

# resample anat_space into mtON
echo "Putting source image into space of destination image..."
echo ">> c3d ${FILE_DEST}.${EXT_REG} ${FILE_SRC}.${EXT_ANAT} -interpolation ${INTERP} -reslice-identity -o ${FILE_SRC}_resampled.${EXT}"
c3d ${FILE_DEST}.${EXT} ${FILE_SRC}.${EXT} -interpolation ${INTERP} -reslice-identity -o ${FILE_SRC}_resliced.${EXT}

SUFFIX_ANAT=${SUFFIX_ANAT}_resliced

FILE_DEST=${PATH_ANAT}${FILE_ANAT}${SUFFIX_ANAT}
FILE_SRC=${PATH_TO_REGISTER}${FILE_TO_REGISTER}${SUFFIX_OTHER}
echo "\n\n\n\n"
EXT_ANAT=nii

#----------------------------------------
# STEP ZERO-ZERO: masking the anatomical file
#----------------------------------------

if [[ "$MASKING" -eq 1 ]]
then
	echo ">> Masking ${PATH_ANAT}${FILE_ANAT}${SUFFIX_ANAT}.${EXT_ANAT} ..."

	cmd="c3d ${PATH_ANAT}${FILE_ANAT}${SUFFIX_ANAT}.${EXT_ANAT} ${PATH_SURFACE}${SURFACE}.${EXT_SURF} -interpolation ${INTERP} -reslice-identity -o ${PATH_SURFACE}${SURFACE}_resliced.${EXT_SURF}"
	echo $cmd
	eval $cmd

	cmd="${PREFIX_NIFTI} fslmaths ${PATH_SURFACE}${SURFACE}_resliced.${EXT_SURF} -kernel gauss $KernelGauss -dilM -s $Sigma ${PATH_SURFACE}${SURFACE}_mask.${EXT}"
	echo $cmd
	eval $cmd

	cmd="${PREFIX_NIFTI} fslmaths ${PATH_ANAT}${FILE_ANAT}${SUFFIX_ANAT}.${EXT_ANAT} -mul ${PATH_SURFACE}${SURFACE}_mask.${EXT} ${PATH_ANAT}${FILE_ANAT}${SUFFIX_ANAT}_masked.${EXT}"
	echo $cmd
	eval $cmd

	echo ">> Masking ${PATH_TO_REGISTER}${FILE_TO_REGISTER}${SUFFIX_OTHER}.${EXT_OTHER} ..."

	cmd="${PREFIX_NIFTI} fslmaths ${PATH_TO_REGISTER}${FILE_TO_REGISTER}${SUFFIX_OTHER}.${EXT_REG} -mul ${PATH_SURFACE}${SURFACE}_mask.${EXT} ${PATH_TO_REGISTER}${FILE_TO_REGISTER}${SUFFIX_OTHER}_masked.${EXT}"
	echo $cmd
	eval $cmd


	SUFFIX_ANAT=${SUFFIX_ANAT}_masked
	SUFFIX_OTHER=${SUFFIX_OTHER}_masked

	EXT_ANAT=nii
	EXT_OTHER=nii

	FILE_DEST=${PATH_TO_REGISTER}${FILE_TO_REGISTER}${SUFFIX_OTHER}
	FILE_SRC=${PATH_ANAT}${FILE_ANAT}${SUFFIX_ANAT}
	echo "\n\n\n\n"
else
	SUFFIX_ANAT=${SUFFIX_ANAT}
	SUFFIX_OTHER=${SUFFIX_OTHER}
	FILE_SRC=${FILE_SRC}
	FILE_DEST=${FILE_DEST}
fi

#----------------------------------------
# STEP TWO: split the images in 2D
#----------------------------------------

# create two directories where the 3D images will be split in 2D
mkdir ${PATH_TO_REGISTER}2D_slices
PATH_2DANAT=${PATH_TO_REGISTER}2D_slices/anat_to_${NAME_OTHER_MODALITY}
PATH_2DOTHER=${PATH_TO_REGISTER}2D_slices/${NAME_OTHER_MODALITY}

mkdir ${PATH_2DANAT}
mkdir ${PATH_2DOTHER}

echo "Splitting images in 2D..."

ORIENTATION=$(fslorient "${FILE_SRC}")
if [[ "$ORIENTATION" == "NEUROLOGICAL" ]]
then
	SWAP="AP LR IS"
elif [[ "$ORIENTATION" == "RADIOLOGICAL" ]]
then
	SWAP="AP RL IS"
fi


# reorient to AP RL IS (axial slices)
cmd="${PREFIX_NIFTI} fslswapdim ${FILE_DEST} $SWAP ${PATH_2DANAT}/${FILE_ANAT}_APRLIS"
echo $cmd
eval $cmd
# extract slices from nii volume
cmd="${PREFIX_NIFTI} fslsplit ${PATH_2DANAT}/${FILE_ANAT}_APRLIS ${PATH_2DANAT}/${FILE_ANAT}_ -z"
echo $cmd
eval $cmd

# reorient to AP RL IS (axial slices)
#echo ">> fslswapdim ${FILE_DEST} AP LR IS other_modality/${FILE_DEST}_APRLIS"
cmd="${PREFIX_NIFTI} fslswapdim ${FILE_SRC} $SWAP ${PATH_2DOTHER}/${FILE_TO_REGISTER}_APRLIS"
echo $cmd
eval $cmd
# extract slices from nii volume
cmd="${PREFIX_NIFTI} fslsplit ${PATH_2DOTHER}/${FILE_TO_REGISTER}_APRLIS ${PATH_2DOTHER}/${FILE_TO_REGISTER}_ -z"
echo $cmd
eval $cmd

NO_IMGS=$(find ${PATH_2DANAT}/${FILE_ANAT}_0* -type f | wc -l)
let "NO_IMGS -= 1" # subtract 1 because it begins at 0
echo "\n\n\n\n"

#----------------------------------------
# STEP THREE: register the 2D images into each others
#----------------------------------------

PATH_MATRICES=${PATH_TO_REGISTER}matrices
mkdir ${PATH_MATRICES}

echo "2D registration with $METHOD"

if [[ "$MASKING" -eq 1 ]]
then
	# create two directories where the 3D image non-smoothed will be split in 2D
	PATH_2DOTHER_NS=${PATH_TO_REGISTER}2D_slices/${NAME_OTHER_MODALITY}_non_smoothed
	mkdir ${PATH_2DOTHER_NS}

	echo "Splitting images in 2D..."

	ORIENTATION=$(fslorient "${FILE_SRC}")
	if [[ "$ORIENTATION" == "NEUROLOGICAL" ]]
	then
		SWAP="AP LR IS"
	elif [[ "$ORIENTATION" == "RADIOLOGICAL" ]]
	then
		SWAP="AP RL IS"
	fi

	# reorient to AP RL IS (axial slices)
	#echo ">> fslswapdim ${FILE_DEST} AP LR IS other_modality/${FILE_DEST}_APRLIS"
	cmd="${PREFIX_NIFTI} fslswapdim ${PATH_TO_REGISTER}${FILE_TO_REGISTER} $SWAP ${PATH_2DOTHER_NS}/${FILE_TO_REGISTER}_APRLIS"
	echo $cmd
	eval $cmd
	# extract slices from nii volume
	cmd="${PREFIX_NIFTI} fslsplit ${PATH_2DOTHER_NS}/${FILE_TO_REGISTER}_APRLIS ${PATH_2DOTHER_NS}/${FILE_TO_REGISTER}_ -z"
	echo $cmd
	eval $cmd
fi

for i in `seq 0 ${NO_IMGS}`;
do
	if [ "$i" -lt 10 ]
	then
		NO_IMG=0${i}
	else
		NO_IMG=${i}
	fi

	FILE_DEST=${PATH_2DANAT}/${FILE_ANAT}_00${NO_IMG}
	FILE_SRC=${PATH_2DOTHER}/${FILE_TO_REGISTER}_00${NO_IMG}
	OutPrefix=${PATH_MATRICES}/00${NO_IMG}_

	if [[ "$METHOD" == "ants" ]]
	then	
		cmd="isct_antsRegistration -d $ImageDimension -r [${FILE_DEST}.${EXT},${FILE_SRC}.${EXT} ,1] -m ${MetricType}[${FILE_DEST}.${EXT},${FILE_SRC}.${EXT},1,4] --use-histogram-matching 1 -o $OutPrefix -t Translation -c [10000x10000x10000,1.e-8,20] -s 4x2x1vox -f 3x2x1"
		echo $cmd
		eval $cmd

		cmd="ConvertTransformFile $ImageDimension ${OutPrefix}0GenericAffine.mat ${OutPrefix}Affine.txt"
		echo $cmd
		eval $cmd

		if [[ "$MASKING" -eq 1 ]]
		then
			FILE_SRC=${PATH_2DOTHER_NS}/${FILE_TO_REGISTER}_00${NO_IMG}
		fi

		cmd="antsApplyTransforms -d $ImageDimension -i ${FILE_SRC}.${EXT} -r ${FILE_DEST}.${EXT} -t ${OutPrefix}Affine.txt -n BSpline -o ${FILE_SRC}_reg.${EXT}"
		echo $cmd
		eval $cmd
	
	elif [[ "$METHOD" == "flirt" ]]
	then
		cmd="${PREFIX_NIFTI} flirt -in ${FILE_SRC}.${EXT} -ref ${FILE_DEST}.${EXT} -schedule schedule_2013_09.sch -omat ${OutPrefix}omat.mat -cost normcorr -forcescaling -out ${FILE_SRC}_reg.${EXT}"
		echo $cmd
		eval $cmd
		
		if [[ "$MASKING" -eq 1 ]]
		then
			FILE_SRC=${PATH_2DOTHER_NS}/${FILE_TO_REGISTER}_00${NO_IMG}
		fi

		cmd="${PREFIX_NIFTI} flirt -in ${FILE_SRC}.${EXT} -ref ${FILE_DEST}.${EXT} -interp nearestneighbour -applyxfm -init ${OutPrefix}omat.mat -out ${FILE_SRC}_reg.${EXT}"
		echo $cmd
		eval $cmd
	fi

done


echo "\n\n\n\n"

#----------------------------------------
# STEP FOUR: Reslice in the space of the raw anatomical file
#----------------------------------------

echo "Reslicing in the raw anatomical file... Last step !"

PATH_OUTPUT=${PATH_TO_REGISTER}2D_slices/results_${NAME_OTHER_MODALITY}
mkdir ${PATH_OUTPUT}

if [[ "${PATH_ANAT##*.}" == "gz" ]] || [[ -e "${PATH_ANAT}.nii.gz" ]]
then
	EXT_ANAT=nii.gz
elif [[ "${PATH_ANAT##*.}" == "nii" ]] || [[ -e "${PATH_ANAT}.nii" ]]
then
	EXT_ANAT=nii	
fi


#Another method would be to create a string for all the images and then use fslmaths
for i in `seq 0 ${NO_IMGS}`;
do

	if [ "$i" -lt 10 ]
	then
		NO_IMG=0${i}
	else
		NO_IMG=${i}
	fi

	NAME_OUTPUT=${PATH_OUTPUT}/final_${NO_IMG}
	FILE_SRC=${PATH_2DOTHER}/${FILE_TO_REGISTER}_00${NO_IMG}_reg

	if [[ "$MASKING" -eq 1 ]]
	then
		FILE_SRC=${PATH_2DOTHER_NS}/${FILE_TO_REGISTER}_00${NO_IMG}
	fi

	if [ "$i" -gt 0 ]
	  then
		cmd="${PREFIX_NIFTI} fslmerge -z ${PATH_OUTPUT}/tmp_$i.${EXT} ${PREVIOUS_NAME}.${EXT} ${FILE_SRC}.${EXT}"
		echo $cmd
		eval $cmd
		if [ "$i" -eq "${NO_IMGS}" ]
	  		then
	  		echo ">> mv 2D_slices/results/tmp_$i.${EXT} 2D_slices/results/final_add.${EXT}"
	  		mv ${PATH_OUTPUT}/tmp_$i.${EXT} ${PATH_OUTPUT}/resliced.${EXT}
	  		echo ">> rm -f 2D_slices/results/tmp*"
	  		rm -f ${PATH_OUTPUT}/tmp*
	  	fi
	fi
	
	PREVIOUS_NAME=${FILE_SRC}
	if [ "$i" -gt 0 ]
		then
			PREVIOUS_NAME=${PATH_OUTPUT}/tmp_$i.${EXT}
	fi

done

if [[ "$METHOD" == "ants" ]]
then	
	fslcpgeom ${PATH_2DOTHER}/${FILE_TO_REGISTER}_APRLIS ${PATH_OUTPUT}/resliced.${EXT}
fi

# resample to anat space
echo "c3d ${PATH_ANAT}${FILE_ANAT}.${EXT_ANAT} ${PATH_OUTPUT}/resliced_${NO_IMG}.${EXT} -interpolation ${INTERP} -reslice-identity -o ${NAME_OUTPUT}.${EXT}"
c3d ${PATH_ANAT}${FILE_ANAT}.${EXT_ANAT} ${PATH_OUTPUT}/resliced.${EXT} -interpolation ${INTERP} -reslice-identity -o ${PATH_OUTPUT}/final.${EXT}

rm -f ${PATH_OUTPUT}/resliced.${EXT}

echo "\n\n\n\n"



#---------------------

mkdir ${PATH_TO_REGISTER}Modified_Images
mv ${FILE_ANAT}_* ${PATH_TO_REGISTER}Modified_Images
if [[ "$MASKING" -eq 1 ]]
then
	mv ${PATH_TO_REGISTER}${FILE_TO_REGISTER}_* ${PATH_TO_REGISTER}Modified_Images/
	mv ${PATH_SURFACE}${SURFACE}_* ${PATH_TO_REGISTER}Modified_Images/
fi
#-----
PREFIX_NIFTI="export FSLOUTPUTTYPE='${FSLOUTPUT_INIT}';"
eval ${PREFIX_NIFTI}

#----------

echo "Summary of files created:"
echo ">> Modified raw images: \t${PATH_TO_REGISTER}Modified_Images "
echo ">> 2D anatomical image files: \t${PATH_2DANAT}"
echo ">> 2D EPI image files: \t\t${PATH_2DOTHER}"
echo ">> EPI image registered: \t${PATH_OUTPUT}" 
echo ">> Transform matrices: \t\t${PATH_MATRICES}"

echo "\nDone!"


