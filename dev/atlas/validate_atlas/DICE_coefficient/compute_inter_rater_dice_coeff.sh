#!/bin/bash

# rater number
# 1=charles
# 2=julien
# 3=simon
# 4=tanguy


echo DICE coefficients ---Rater 1 X Rater 2---
echo Left CST
echo Right CST
echo Dorsal column
echo 
sct_dice_coefficient ../manual_masks/charles/manual_csl.nii.gz ../manual_masks/julien/manual_csl.nii.gz
sct_dice_coefficient ../manual_masks/charles/manual_csr.nii.gz ../manual_masks/julien/manual_csr.nii.gz
sct_dice_coefficient ../manual_masks/charles/manual_dc.nii.gz ../manual_masks/julien/manual_dc.nii.gz

echo ---------------------------------------------------------------------------------------------------

echo DICE coefficients ---Rater 1 X Rater 3---
echo Left CST
echo Right CST
echo Dorsal column
echo 
sct_dice_coefficient ../manual_masks/charles/manual_csl.nii.gz ../manual_masks/simon/manual_csl.nii.gz
sct_dice_coefficient ../manual_masks/charles/manual_csr.nii.gz ../manual_masks/simon/manual_csr.nii.gz
sct_dice_coefficient ../manual_masks/charles/manual_dc.nii.gz ../manual_masks/simon/manual_dc.nii.gz

echo ---------------------------------------------------------------------------------------------------

echo DICE coefficients ---Rater 1 X Rater 4---
echo Left CST
echo Right CST
echo Dorsal column
echo 
sct_dice_coefficient ../manual_masks/charles/manual_csl.nii.gz ../manual_masks/tanguy/manual_csl.nii.gz
sct_dice_coefficient ../manual_masks/charles/manual_csr.nii.gz ../manual_masks/tanguy/manual_csr.nii.gz
sct_dice_coefficient ../manual_masks/charles/manual_dc.nii.gz ../manual_masks/tanguy/manual_dc.nii.gz

echo ---------------------------------------------------------------------------------------------------

echo DICE coefficients ---Rater 2 X Rater 3---
echo Left CST
echo Right CST
echo Dorsal column
echo 
sct_dice_coefficient ../manual_masks/julien/manual_csl.nii.gz ../manual_masks/simon/manual_csl.nii.gz
sct_dice_coefficient ../manual_masks/julien/manual_csr.nii.gz ../manual_masks/simon/manual_csr.nii.gz
sct_dice_coefficient ../manual_masks/julien/manual_dc.nii.gz ../manual_masks/simon/manual_dc.nii.gz

echo ---------------------------------------------------------------------------------------------------

echo DICE coefficients ---Rater 2 X Rater 4---
echo Left CST
echo Right CST
echo Dorsal column
echo 
sct_dice_coefficient ../manual_masks/julien/manual_csl.nii.gz ../manual_masks/tanguy/manual_csl.nii.gz
sct_dice_coefficient ../manual_masks/julien/manual_csr.nii.gz ../manual_masks/tanguy/manual_csr.nii.gz
sct_dice_coefficient ../manual_masks/julien/manual_dc.nii.gz ../manual_masks/tanguy/manual_dc.nii.gz

echo ---------------------------------------------------------------------------------------------------

echo DICE coefficients ---Rater 3 X Rater 4---
echo Left CST
echo Right CST
echo Dorsal column
echo 
sct_dice_coefficient ../manual_masks/simon/manual_csl.nii.gz ../manual_masks/tanguy/manual_csl.nii.gz
sct_dice_coefficient ../manual_masks/simon/manual_csr.nii.gz ../manual_masks/tanguy/manual_csr.nii.gz
sct_dice_coefficient ../manual_masks/simon/manual_dc.nii.gz ../manual_masks/tanguy/manual_dc.nii.gz