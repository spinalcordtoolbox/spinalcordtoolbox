#!/bin/bash

path=$(pwd)

cd /home/django/jtouati/data/template_data/new_data


# if [ -e "t1_register" ];then
#     rm -rf t1_register
# fi

# mkdir t1_register
#
# SUBJECTS_MTL="errsm_02 errsm_04 errsm_31 errsm_32 errsm_33"

SUBJECTS_MAR="ALT JD JW MT T047 VG VP"

# cd Montreal/
#
# for i in $SUBJECTS_MTL; do
#
# 	sct_register_multimodal -i ${i}/T1/${i}_t1.nii.gz -d ${i}/T2/${i}_t2.nii.gz
# 	WarpImageMultiTransform 3 ${i}/T1/${i}_t1.nii.gz t1_registered.nii.gz -R ${i}/T2/${i}_t2.nii.gz --use-BSpline warp_src2dest.nii.gz
# 	mv t1_registered.nii.gz ../t1_register/${i}_t1.nii.gz
# done

cd Marseille/

for i in $SUBJECTS_MAR; do
	
	sct_register_multimodal -i ${i}/T1/${i}_t1.nii.gz -d ${i}/T2/${i}_t2.nii.gz
	WarpImageMultiTransform 3 ${i}/T1/${i}_t1.nii.gz t1_registered.nii.gz -R ${i}/T2/${i}_t2.nii.gz --use-BSpline warp_src2dest.nii.gz
	mv t1_registered.nii.gz ../t1_register/${i}_t1.nii.gz
	
done





