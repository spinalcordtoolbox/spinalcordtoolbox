#!/bin/bash

path=$(pwd)

cd /home/django/jtouati/data/template_data/

########################## T2

if [ -e "t2_output" ];then
    rm -rf t2_output
fi

mkdir t2_output

SUBJECTS_MTL="errsm_02 errsm_04 errsm_05 errsm_09 errsm_10 errsm_11 errsm_12 errsm_13 errsm_14 errsm_16 errsm_17 errsm_18 errsm_20 errsm_21 errsm_22 errsm_23 errsm_24 errsm_25 errsm_26 errsm_30 errsm_31 errsm_32 errsm_33"

SUBJECTS_MAR="1 7 ALT JD JW MD MLL MT T047 VC VG VP"

cd montreal/

for i in $SUBJECTS_MTL; do

	cd $i/T2
	echo $i

	if [ -e "second_step" ];then
	    rm -rf second_step
	fi
	mkdir second_step
	echo "Copying files"
	
	if [ -e "${i}_t2_crop_straight_crop_2temp.nii.gz" ];then
		cp ${i}_t2_crop_straight_crop_2temp.nii.gz second_step/${i}_preprocessed.nii.gz
	fi
	
	if [ -e "native2temp.nii.gz" ];then
		cp native2temp.nii.gz second_step/${i}_preprocessed.nii.gz
	fi
	
	cp centerline_straight_crop_2temp.nii.gz second_step/centerline_straight_${i}.nii.gz
	cd second_step/


	echo "Align_vertebrae"
	sct_align_vertebrae.py -i ${i}_preprocessed.nii.gz -l ../${i}_labels.nii.gz -R /home/django/jtouati/code/spinalcordtoolbox/dev/template_creation/template_shape-mask.nii.gz -o ${i}_aligned.nii.gz -t affine -w spline

	echo "applying transfo to the centerline"
	WarpImageMultiTransform 3 centerline_straight_${i}.nii.gz centerline_straight_${i}_aligned.nii.gz -R /home/django/jtouati/code/spinalcordtoolbox/dev/template_creation/template_shape-mask.nii.gz n2t.txt --use-BSpline


	echo "normalizing"
	sct_normalize.py -i ${i}_aligned.nii.gz -c centerline_straight_${i}_aligned.nii.gz

	echo "moving file"
	mv ${i}_aligned_normalized.nii.gz ../../../../t2_output/${i}_preprocessed.nii.gz

    cd ..

    rm -rf second_step

	cd ../..

done

cd ../marseille/

for i in $SUBJECTS_MAR; do

	cd $i/T2
	echo $i

	if [ -e "second_step" ];then
	    rm -rf second_step
	fi
	mkdir second_step
	echo "Copying files"
	
	if [ -e "${i}_t2_crop_straight_crop_2temp.nii.gz" ];then
		cp ${i}_t2_crop_straight_crop_2temp.nii.gz second_step/${i}_preprocessed.nii.gz
	fi
	
	if [ -e "native2temp.nii.gz" ];then
		cp native2temp.nii.gz second_step/${i}_preprocessed.nii.gz
	fi
	
	cp centerline_straight_crop_2temp.nii.gz second_step/centerline_straight_${i}.nii.gz
	cd second_step/


	echo "Align_vertebrae"
	sct_align_vertebrae.py -i ${i}_preprocessed.nii.gz -l ../${i}_labels.nii.gz -R /home/django/jtouati/code/spinalcordtoolbox/dev/template_creation/template_shape-mask.nii.gz -o ${i}_aligned.nii.gz -t affine -w spline

	echo "applying transfo to the centerline"
	WarpImageMultiTransform 3 centerline_straight_${i}.nii.gz centerline_straight_${i}_aligned.nii.gz -R /home/django/jtouati/code/spinalcordtoolbox/dev/template_creation/template_shape-mask.nii.gz n2t.txt --use-BSpline

	echo "normalizing"
	sct_normalize.py -i ${i}_aligned.nii.gz -c centerline_straight_${i}_aligned.nii.gz

	echo "moving file"
	mv ${i}_aligned_normalized.nii.gz ../../../../t2_output/${i}_preprocessed.nii.gz

    cd ..

    rm -rf second_step


	cd ../..


done

# cd /home/django/jtouati/data/template_data
#
#
# if [ -e "t1_output" ];then
#     rm -rf t1_output
# fi
#
# mkdir t1_output
#
# SUBJECTS_MTL="errsm_02 errsm_04 errsm_31 errsm_32 errsm_33"
#
# SUBJECTS_MAR="ALT JD JW MT T047 T020b VG VP"
#
# cd Montreal/
#
# for i in $SUBJECTS_MTL; do
#
# 	cd $i/T1
# 	echo $i
#
# 	if [ -e "second_step" ];then
# 	    rm -rf second_step
# 	fi
# 	mkdir second_step
# 	echo "Copying files"
# 	cp ${i}_t1_crop_straight_crop_2temp.nii.gz second_step/${i}_preprocessed.nii.gz
# 	cp centerline_straight_crop_2temp.nii.gz second_step/centerline_straight_${i}.nii.gz
# 	cd second_step/
#
#
# 	echo "Align_vertebrae"
# 	sct_align_vertebrae.py -i ${i}_preprocessed.nii.gz -l ../../T2/${i}_labels.nii.gz -R /home/django/jtouati/code/spinalcordtoolbox/dev/template_creation/template_shape-mask.nii.gz -o ${i}_aligned.nii.gz -t affine - w spline
#
# 	echo "applying transfo to the centerline"
# 	WarpImageMultiTransform 3 centerline_straight_${i}.nii.gz centerline_straight_${i}_aligned.nii.gz -R /home/django/jtouati/code/spinalcordtoolbox/dev/template_creation/template_shape-mask.nii.gz n2t.txt --use-BSpline
#
#
# 	echo "normalizing"
# 	sct_normalize.py -i ${i}_aligned.nii.gz -c centerline_straight_${i}_aligned.nii.gz
#
# 	echo "moving file"
# 	mv ${i}_aligned_normalized.nii.gz ../../../../t1_output/${i}_preprocessed.nii.gz
#
# 	cd ../../..
#
# done
#
# cd ../Marseille/
#
# for i in $SUBJECTS_MAR; do
#
# 	cd $i/T1
# 	echo $i
#
# 	if [ -e "second_step" ];then
# 	    rm -rf second_step
# 	fi
# 	mkdir second_step
# 	echo "Copying files"
# 	cp ${i}_t1_crop_straight_crop_2temp.nii.gz second_step/${i}_preprocessed.nii.gz
# 	cp centerline_straight_crop_2temp.nii.gz second_step/centerline_straight_${i}.nii.gz
# 	cd second_step/
#
#
# 	echo "Align_vertebrae"
# 	sct_align_vertebrae.py -i ${i}_preprocessed.nii.gz -l  ../../T2/${i}_labels.nii.gz -R /home/django/jtouati/code/spinalcordtoolbox/dev/template_creation/template_shape-mask.nii.gz -o ${i}_aligned.nii.gz -t affine - w spline
#
# 	echo "applying transfo to the centerline"
# 	WarpImageMultiTransform 3 centerline_straight_${i}.nii.gz centerline_straight_${i}_aligned.nii.gz -R /home/django/jtouati/code/spinalcordtoolbox/dev/template_creation/template_shape-mask.nii.gz n2t.txt --use-BSpline
#
# 	echo "normalizing"
# 	sct_normalize.py -i ${i}_aligned.nii.gz -c centerline_straight_${i}_aligned.nii.gz
#
# 	echo "moving file"
# 	mv ${i}_aligned_normalized.nii.gz ../../../../t1_output/${i}_preprocessed.nii.gz
#
# 	cd ../../..
#
#
# done
#
#


