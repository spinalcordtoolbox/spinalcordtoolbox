#!/bin/bash

# rater number
# 1=benjamin
# 2=julien
# 3=simon
# 4=tanguy


echo DICE coefficients ---Rater 1---
echo Left CST
echo Right CST
echo Dorsal column
echo 
sct_dice_coefficient ../manual_masks/benjamin/manual_csl.nii.gz ../manual_masks/benjamin/manual_csl_2.nii.gz
sct_dice_coefficient ../manual_masks/benjamin/manual_csr.nii.gz ../manual_masks/benjamin/manual_csr_2.nii.gz
sct_dice_coefficient ../manual_masks/benjamin/manual_dc.nii.gz ../manual_masks/benjamin/manual_dc_2.nii.gz

echo ---------------------------------------------------------------------------------------------------

echo DICE coefficients ---Rater 2---
echo Left CST
echo Right CST
echo Dorsal column
echo 
sct_dice_coefficient ../manual_masks/julien/manual_csl.nii.gz ../manual_masks/julien/manual_csl_2.nii.gz
sct_dice_coefficient ../manual_masks/julien/manual_csr.nii.gz ../manual_masks/julien/manual_csr_2.nii.gz
sct_dice_coefficient ../manual_masks/julien/manual_dc.nii.gz ../manual_masks/julien/manual_dc_2.nii.gz

echo ---------------------------------------------------------------------------------------------------

echo DICE coefficients ---Rater 3---
echo Left CST
echo Right CST
echo Dorsal column
echo 
sct_dice_coefficient ../manual_masks/simon/manual_csl.nii.gz ../manual_masks/simon/manual_csl_2.nii.gz
sct_dice_coefficient ../manual_masks/simon/manual_csr.nii.gz ../manual_masks/simon/manual_csr_2.nii.gz
sct_dice_coefficient ../manual_masks/simon/manual_dc.nii.gz ../manual_masks/simon/manual_dc_2.nii.gz

echo ---------------------------------------------------------------------------------------------------

echo DICE coefficients ---Rater 4---
echo Left CST
echo Right CST
echo Dorsal column
echo 
sct_dice_coefficient ../manual_masks/tanguy/manual_csl.nii.gz ../manual_masks/tanguy/manual_csl_2.nii.gz
sct_dice_coefficient ../manual_masks/tanguy/manual_csr.nii.gz ../manual_masks/tanguy/manual_csr_2.nii.gz
sct_dice_coefficient ../manual_masks/tanguy/manual_dc.nii.gz ../manual_masks/tanguy/manual_dc_2.nii.gz

echo ---------------------------------------------------------------------------------------------------

echo Mean left CST = 0.5928
echo Mean right CST = 0.6170
echo Mean dorsal columns = 0.7030