#!/bin/bash

########################## T2

mkdir labels_PMJ_C3-L1_19labels

SUBJECTS="errsm_02 errsm_04 errsm_05 errsm_09 errsm_10 errsm_11 errsm_12 errsm_13 errsm_14 errsm_16 errsm_17 errsm_18 errsm_20 errsm_21 errsm_22 errsm_23 errsm_24 errsm_25 errsm_26 errsm_30 errsm_31 errsm_32 errsm_33 1 7 ALT JD JW MD MLL MT T047 TR VC VG VP"

cd t2_crop/

for subject in $SUBJECTS; do

	sct_label_utils -i ${subject}_preprocessed_labels.nii.gz -o ${subject}_preprocessed_labels_incremented.nii.gz -t increment
	mv ${subject}_preprocessed_labels_incremented.nii.gz ../labels_PMJ_C3-L1_19labels

done

cd ..
sct_average_levels.py -i labels_PMJ_C3-L1_19labels -t t2_crop/errsm_02_preprocessed.nii.gz -n 19

mkdir t2_crop_aligned

for subject in $SUBJECTS; do

	sct_align_vertebrae.py -i t2_crop/${subject}_preprocessed.nii.gz -l labels_PMJ_C3-L1_19labels/${subject}_preprocessed_labels_incremented.nii.gz -R labels_PMJ_C3-L1_19labels/template_landmarks.nii.gz -o t2_crop_aligned/${subject}_preprocessed.nii.gz -t SyN -w spline

done

