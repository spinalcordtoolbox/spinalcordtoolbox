#!/bin/bash

# add path to scripts
export PATH=${PATH}:$SCT_DIR/dev/template_creation


###Marseille :
##1:
#(nurbs 13)

sct_crop_image -i mar_1_t1.nii.gz -o mar_1_t1_crop.nii.gz -dim 2 -start 27 -end 559

sct_straighten_spinalcord.py -i mar_1_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i mar_1_t1_crop_straight.nii.gz -o mar_1_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 570

sct_create_cross.py -i mar_1_t1_crop_straight_crop.nii.gz -x 48 -y 158

sct_push_into_template_space.py -i mar_1_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


##7 (nurbs 13)

sct_crop_image -i mar_7_t1.nii.gz -o mar_7_t1_crop.nii.gz -start 22 -end 578 -dim 2

sct_straighten_spinalcord.py -i mar_7_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i mar_7_t1_crop_straight.nii.gz -o mar_7_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 592

sct_create_cross.py -i mar_7_t1_crop_straight_crop.nii.gz -x 53 -y 161

sct_push_into_template_space.py -i mar_7_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz



##errsm_03 (nurbs 20)

sct_crop_image -i errsm_03_t1.nii.gz -o errsm_03_t1_crop.nii.gz -dim 2 -start 23 -end 569

sct_straighten_spinalcord.py -i errsm_03_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_03_t1_crop_straight.nii.gz -o errsm_03_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 588

sct_create_cross.py -i errsm_03_t1_crop_straight_crop.nii.gz -x 55 -y 229

sct_push_into_template_space.py -i errsm_03_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz

fslmaths native2temp.nii.gz -inm 1000 errsm_03_normalized.nii.gz

sct_align_vertebrae.py -i errsm_03_normalized.nii.gz -l /home/django/jtouati/data/Align_VERTEBRES/less_landmarks/masks/errsm_03_preprocessed-mask.nii.gz -R /home/django/jtouati/data/Align_VERTEBRES/less_landmarks/masks/template_shape-mask -o errsm_03_aligned.nii.gz -t affine -w spline

## errsm 05 (20) 

sct_crop_image -i errsm_05_t1.nii.gz -o errsm_05_t1_crop.nii.gz -start 35 -end 528 -dim 2

sct_straighten_spinalcord.py -i errsm_05_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_05_t1_crop_straight.nii.gz -o errsm_05_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 542

sct_create_cross.py -i errsm_05_t1_crop_straight_crop.nii.gz -x 54 -y 221
 


## errsm_09 (20)

sct_crop_image -i errsm_09_t1.nii.gz -o errsm_09_t1_crop.nii.gz -dim 2 -start 52 -end 540

sct_straighten_spinalcord.py -i errsm_09_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_09_t1_crop_straight.nii.gz -o errsm_09_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 528

sct_create_cross.py -i errsm_09_t1_crop_straight_crop.nii.gz -x 55 -y 226

sct_push_into_template_space.py -i errsm_09_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz

## errsm_10 (20)

sct_crop_image -i errsm_10_t1.nii.gz -o errsm_10_t1_crop.nii.gz -dim 2 -start 23 -end 568 

sct_straighten_spinalcord.py -i errsm_10_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_10_t1_crop_straight.nii.gz -o errsm_10_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 589

sct_create_cross.py -i errsm_10_t1_crop_straight_crop.nii.gz -x 53 -y 242

sct_push_into_template_space.py -i errsm_10_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

## errsm_11 (20)

sct_crop_image -i errsm_11_t1.nii.gz -o errsm_11_t1_crop.nii.gz -dim 2 -start 17 -end 557

sct_straighten_spinalcord.py -i errsm_11_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_11_t1_crop_straight.nii.gz -o errsm_11_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 590

sct_create_cross.py -i errsm_11_t1_crop_straight_crop.nii.gz -x 55 -y 222

sct_push_into_template_space.py -i errsm_11_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

## errsm_12 (20)

sct_crop_image -i errsm_12_t1.nii.gz -o errsm_12_t1_crop.nii.gz -dim 2 -start 115 -end 610

sct_straighten_spinalcord.py -i errsm_12_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_12_t1_crop_straight.nii.gz -o errsm_12_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 540

sct_create_cross.py -i errsm_12_t1_crop_straight_crop.nii.gz -x 55 -y 221

sct_push_into_template_space.py -i errsm_12_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

## errsm_13 (20)

sct_crop_image -i errsm_13_t1.nii.gz -o errsm_13_t1_crop.nii.gz -dim 2 -start 21 -end 618 

sct_straighten_spinalcord.py -i errsm_13_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_13_t1_crop_straight.nii.gz -o errsm_13_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 637

sct_create_cross.py -i errsm_13_t1_crop_straight_crop.nii.gz -x 54 -y 221

sct_push_into_template_space.py -i errsm_13_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz


## errsm_14 (20)

sct_crop_image -i errsm_14_t1.nii.gz -o errsm_14_t1_crop.nii.gz -dim 2 -start 42 -end 515

sct_straighten_spinalcord.py -i errsm_14_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_14_t1_crop_straight.nii.gz -o errsm_14_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 516

sct_create_cross.py -i errsm_14_t1_crop_straight_crop.nii.gz -x 54 -y 221

sct_push_into_template_space.py -i errsm_14_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


## errsm_16 (20)

sct_crop_image -i errsm_16_t1.nii.gz -o errsm_16_t1_crop.nii.gz -dim 2 -start 114 -end 618

sct_straighten_spinalcord.py -i errsm_16_t1_crop.nii.gz -c full_centerline.nii.gz -v 2 

sct_crop_image -i errsm_16_t1_crop_straight.nii.gz -o errsm_16_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 546 

sct_create_cross.py -i errsm_16_t1_crop_straight_crop.nii.gz -x 54 -y 221 

sct_push_into_template_space.py -i errsm_16_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 



# ## errsm_17 (20)
# sct_crop_image -i errsm_17_t1.nii.gz -o errsm_17_t1_crop.nii.gz -dim 2 -start 51 -end 595
#
# sct_segmentation_propagation -i errsm_17_t1_crop.nii.gz -t t1 -centerline-binary
#
# sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz
#
# sct_generate_centerline.py -i down.nii.gz -o cdown.nii.gz
#
# sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 127
#
# sct_erase_centerline.py -i centerline_erased.nii.gz -s 476 -e 544
#
# sct_straighten_spinalcord.py -i errsm_17_t1_crop.nii.gz -c full_centerline.nii.gz -v 2
#
# sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_17_t1_crop_straight.nii.gz warp_curve2straight.nii.gz
#
# sct_detect_extrema.py -i centerline_straight.nii.gz
#
# sct_crop_image -i errsm_17_t1_crop_straight.nii.gz -o errsm_17_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 585
#
# sct_create_cross.py -i errsm_17_t1_crop_straight_crop.nii.gz -x 54 -y 221
#
# sct_push_into_template_space.py -i errsm_17_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz

## errsm_17 BIS (30)

sct_crop_image -i errsm_17_t1.nii.gz -o errsm_17_t1_crop.nii.gz -dim 2 -start 86 -end 595

sct_straighten_spinalcord.py -i errsm_17_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_17_t1_crop_straight.nii.gz -o errsm_17_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 550

sct_create_cross.py -i errsm_17_t1_crop_straight_crop.nii.gz -x 55 -y 222

sct_push_into_template_space.py -i errsm_17_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz


# ## errsm_18 (20)
#
# sct_crop_image -i errsm_18_t1.nii.gz -o errsm_18_t1_crop.nii.gz -dim 2 -start 16 -end 567
#
# sct_straighten_spinalcord.py -i errsm_18_t1_crop.nii.gz -c full_centerline.nii.gz -v 2
#
# sct_crop_image -i errsm_18_t1_crop_straight.nii.gz -o errsm_18_t1_crop_straight_straight.nii.gz -start 30 -end 591 -dim 2
#
# sct_create_cross.py -i errsm_18_t1_crop_straight_straight.nii.gz -x 54 -y 222
#
# sct_push_into_template_space.py -i errsm_18_t1_crop_straight_straight.nii.gz -n landmark_native.nii.gz


## errsm_18 BIS (30)

sct_crop_image -i errsm_18_t1.nii.gz -o errsm_18_t1_crop.nii.gz -dim 2 -start 15 -end 568

sct_straighten_spinalcord.py -i errsm_18_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_18_t1_crop_straight.nii.gz -o errsm_18_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 593

sct_create_cross.py -i errsm_18_t1_crop_straight_crop.nii.gz -x 24 -y 222

sct_push_into_template_space.py -i errsm_18_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz


## errsm_20 (30)

sct_crop_image -i errsm_20_t1.nii.gz -o errsm_20_t1_crop.nii.gz -dim 2 -start 39 -end 604 

sct_straighten_spinalcord.py -i errsm_20_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_20_t1_crop_straight.nii.gz -o errsm_20_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 624 

sct_create_cross.py -i errsm_20_t1_crop_straight_crop.nii.gz -x 55 -y 223

sct_push_into_template_space.py -i errsm_20_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


## errsm_21 (30)

sct_crop_image -i errsm_21_t1.nii.gz -o errsm_21_t1_crop.nii.gz -dim 2 -start 48 -end 591

sct_straighten_spinalcord.py -i errsm_21_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_21_t1_crop_straight.nii.gz -o errsm_21_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 585 

sct_create_cross.py -i errsm_21_t1_crop_straight_crop.nii.gz -x 51 -y 221
 
sct_push_into_template_space.py -i errsm_21_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz



## errsm_22 (30)

sct_crop_image -i errrsm_22_t1.nii.gz -o errsm_22_t1_crop.nii.gz -dim 2 -start 65 -end 582

sct_straighten_spinalcord.py -i errsm_22_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_22_t1_crop_straight.nii.gz -o errsm_22_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 558 

sct_create_cross.py -i errsm_22_t1_crop_straight_crop.nii.gz -x 54 -y 222 

sct_push_into_template_space.py -i errsm_22_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


# errsm_23 (30)

sct_crop_image -i errsm_23_t1.nii.gz -o errsm_23_t1_crop.nii.gz -dim 2 -start 62 -end 582

sct_straighten_spinalcord.py -i errsm_23_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_23_t1_crop_straight.nii.gz -o errsm_23_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 557

sct_create_cross.py -i errsm_23_t1_crop_straight_crop.nii.gz -x 55 -y 222

sct_push_into_template_space.py -i errsm_23_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


## errsm_24 (30)

sct_crop_image -i errsm_24_t1.nii.gz -o errsm_24_t1_crop.nii.gz -dim 2 -start 26 -end 579

sct_straighten_spinalcord.py -i errsm_24_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_24_t1_crop_straight.nii.gz -o errsm_24_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 597

sct_create_cross.py -i errsm_24_t1_crop_straight_crop.nii.gz -x 55 -y 221

sct_push_into_template_space.py -i errsm_24_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


## errsm_25 (30)

## centerline fait main quasiment

sct_crop_image -i errsm_25_t1.nii.gz -o errsm_25_t1_crop.nii.gz -dim 2 -start 105 -end 614

sct_straighten_spinalcord.py -i errsm_25_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_25_t1_crop_straight.nii.gz -o errsm_25_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 556

sct_create_cross.py -i errsm_25_t1_crop_straight_crop.nii.gz -x 55 -y 221

sct_push_into_template_space.py -i errsm_25_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

## errsm_30 (30)

sct_crop_image -i errsm_30_t1.nii.gz -o errsm_30_t1_crop.nii.gz -dim 2 -start 37 -end 576

sct_straighten_spinalcord.py -i errsm_30_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i errsm_30_t1_crop_straight.nii.gz -o errsm_30_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 583

sct_create_cross.py -i errsm_30_t1_crop_straight_crop.nii.gz -x 54 -y 221

sct_push_into_template_space.py -i errsm_30_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 



#ALT

sct_crop_image -i ALT_t1.nii.gz -o ALT_t1_crop.nii.gz -start 0 -end 533 -dim 2

sct_straighten_spinalcord -i ALT_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i ALT_t1_crop_straight.nii.gz -o ALT_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 576

sct_create_cross.py -i ALT_t1_crop_straight_crop.nii.gz -x 51 -y 162

sct_push_into_template_space.py -i ALT_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz


#JD

sct_crop_image -i JD_t1.nii.gz -o JD_t1_crop.nii.gz -dim 2 -start 0 -end 545

sct_straighten_spinalcord -i JD_t1_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_crop_image -i JD_t1_crop_straight.nii.gz -o JD_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 585

sct_create_cross.py -i JD_t1_crop_straight_crop.nii.gz -x 54 -y 162

sct_push_into_template_space.py -i JD_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz


#JW

sct_crop_image -i JW_t1.nii.gz -o JW_t1_crop.nii.gz -dim 2 -start 0 -end 516

sct_straighten_spinalcord -i JW_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i JW_t1_crop_straight.nii.gz -o JW_t1_crop_straight_crop.nii.gz -dim 2 -start 29 -end 551

sct_create_cross.py -i JW_t1_crop_straight_crop.nii.gz -x 56 -y 160

sct_push_into_template_space.py -i JW_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz



#MT
sct_crop_image -i MT_t1.nii.gz -o MT_t1_crop.nii.gz -dim 2 -start 0 -end 515

sct_straighten_spinalcord -i MT_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i MT_t1_crop_straight.nii.gz -o MT_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 558

sct_create_cross.py -i MT_t1_crop_straight_crop.nii.gz -x 56 -y 161

sct_push_into_template_space.py -i MT_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


#T047

sct_crop_image -i T047_t1.nii.gz -o T047_t1_crop.nii.gz -start 0 -end 553 -dim 2

sct_straighten_spinalcord -i T047_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i T047_t1_crop_straight.nii.gz -o T047_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 588 

sct_create_cross.py -i T047_t1_crop_straight_crop.nii.gz -x 55 -y 160

sct_push_into_template_space.py -i T047_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


#VG

sct_crop_image -i VG_t1.nii.gz -o VG_t1.nii.gz -dim 1 -start 11 -end 246

sct_crop_image -i VG_t1.nii.gz -o VG_t1_crop.nii.gz -dim 2 -start 71 -end 549

sct_straighten_spinalcord -i VG_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i VG_t1_crop_straight.nii.gz -o VG_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 517 

sct_create_cross.py -i VG_t1_crop_straight_crop.nii.gz -x 55 -y 149

sct_push_into_template_space.py -i VG_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


#VP

sct_crop_image -i VP_t1.nii.gz -o VP_t1_crop.nii.gz -dim 2 -start 0 -end 528 

sct_straighten_spinalcord -i VP_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i VP_t1_crop_straight.nii.gz -o VP_t1_crop_straight_crop.nii.gz -dim 2 -start 29 -end 565

sct_create_cross.py -i VP_t1_crop_straight_crop.nii.gz -x 55 -y 161

sct_push_into_template_space.py -i VP_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


#errsm02

sct_crop_image -i errsm_02_t1.nii.gz -o errsm_02_t1_crop.nii.gz -dim 2 -start 100 -end 620

sct_straighten_spinalcord -i errsm_02_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i errsm_02_t1_crop_straight.nii.gz -o errsm_02_t1_crop_straight_crop.nii.gz -dim 2 -start 29 -end 559

sct_create_cross.py -i errsm_02_t1_crop_straight_crop.nii.gz -x 54 -y 147

sct_push_into_template_space.py -i errsm_02_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


#errsm_04 

sct_crop_image -i errsm_04_t1.nii.gz -o errsm_04_t1_crop.nii.gz -dim 2 -start 73 -end 549

sct_straighten_spinalcord -i errsm_04_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i errsm_04_t1_crop_straight.nii.gz -o errsm_04_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 516

sct_create_cross.py -i errsm_04_t1_crop_straight_crop.nii.gz -x 65 -y 231

sct_push_into_template_space.py -i errsm_04_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


#errsm_31

sct_crop_image -i errsm_31_t1.nii.gz -o errsm_31_t1_crop.nii.gz -dim 2 -start 0 -end 593

sct_straighten_spinalcord -i errsm_31_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i errsm_31_t1_crop_straight.nii.gz -o errsm_31_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 627

sct_create_cross.py -i errsm_31_t1_crop_straight_crop.nii.gz -x 55 -y 223

sct_push_into_template_space.py -i errsm_31_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


# errsm_32

sct_crop_image -i errsm_32_t1.nii.gz -o errsm_32_t1_crop.nii.gz -dim 2 -start 0 -end 552

sct_straighten_spinalcord -i errsm_32_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i errsm_32_t1_crop_straight.nii.gz -o errsm_32_t1_crop_straight_crop.nii.gz -dim 2 -start 29 -end 589

sct_create_cross.py -i errsm_32_t1_crop_straight_crop.nii.gz -x 52 -y 225

sct_push_into_template_space.py -i errsm_32_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


# errsm_33

sct_crop_image -i errsm_33_t1.nii.gz -o errsm_33_t1_crop.nii.gz -dim 2 -start 0 -end 548

sct_straighten_spinalcord -i errsm_33_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i errsm_33_t1_crop_straight.nii.gz -o errsm_33_t1_crop_straight_crop.nii.gz -dim 2 -start 29 -end 587

sct_create_cross.py -i errsm_33_t1_crop_straight_crop.nii.gz -x 55 -y 225

sct_push_into_template_space.py -i errsm_33_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


#MD


sct_crop_image -i MD_t1.nii.gz -o MD_t1_crop.nii.gz -dim 2 -start 8 -end 534

sct_straighten_spinalcord -i MD_t1_crop.nii.gz -c full_centerline.nii.gz 

sct_crop_image -i MD_t1_crop_straight.nii.gz -o MD_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 567 
 
sct_create_cross.py -i  MD_t1_crop_straight_crop.nii.gz -x 57 -y 161

sct_push_into_template_space.py -i  MD_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 




#MLL 

sct_crop_image -i MLL_t1.nii.gz -o MLL_t1.nii.gz -dim 1 -start 12 -end 251 

sct_propseg -i MLL_t1_crop.nii.gz -t t1 -centerline-binary -init-mask init.nii.gz

sct_erase_centerline.py -i MLL_t1_crop_centerline.nii.gz -s 0 -e 63

sct_erase_centerline.py -i centerline_erased.nii.gz -s 467 -e 532

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz

sct_generate_centerline.py -i down.nii.gz -o cdown.nii.gz

sct_straighten_spinalcord -i MLL_t1_crop.nii.gz -c full_centerline.nii.gz 

WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R MLL_t1_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_detect_extrema.py -i centerline_straight.nii.gz

sct_crop_image -i MLL_t1_crop_straight.nii.gz -o MLL_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 568

sct_create_cross.py -i MLL_t1_crop_straight_crop.nii.gz -x 116 -y 152

sct_push_into_template_space.py -i MLL_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

sct_crop_image -i centerline_straight.nii.gz -o centerline_straight_crop.nii.gz -dim 2 -start 30 -end 568

sct_create_cross.py -i centerline_straight_crop.nii.gz -x 116 -y 152

sct_push_into_template_space.py -i centerline_straight_crop.nii.gz -n landmark_native.nii.gz 




# TR
# =================================================================================================
cd marseille_tr
mkdir t1
cd t1
# convert to nii
dcm2nii -o . -r N /Volumes/data_shared/marseille/TR/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5/original-primary-m-norm-dis2d-comp-sp-composed_e01_*.dcm
# change file name
mv *.nii.gz t1.nii.gz
# create mask of spinal cord on T2 image (will be used for registration)
sct_create_mask -i ../t2/t2_RPI_crop.nii.gz -m centerline,../t2/full_centerline.nii.gz -s 50 -f cylinder -o ../t2/mask_spinalcord.nii.gz
# create labels on PMJ (value = 1) and L1 (value = 2) on the cropped_T2 and on the T1 and call it labels_t2_RPI_crop_PMJ-L1.nii.gz and labels_t1_PMJ-L1.nii.gz, respectively. See snapshots. 
# register to T1 to T2: RIGID
sct_ANTSUseLandmarkImagesToGetAffineTransform ../t2/labels_t2_RPI_crop_PMJ-L1.nii.gz labels_t1_PMJ-L1.nii.gz rigid t1_to_t2.txt
sct_apply_transfo -i t1.nii.gz -d ../t2/t2_RPI_crop.nii.gz -w t1_to_t2.txt -o t1_regAffine.nii.gz
# make labels along T1 (every z=50) and then generate centerline
# --> labels_t1_centerline.nii.gz
sct_generate_centerline.py -i labels_t1_centerline.nii.gz -o t1_centerline.nii.gz
# register to T1 to T2: NON-RIGID
sct_register_multimodal -i t1_regAffine.nii.gz -d ../t2/t2_RPI_crop.nii.gz -s t1_centerline.nii.gz -t ../t2/full_centerline.nii.gz -p 0,SyN,0.5,MeanSquares -z 0 -o t1_regAffineWarp.nii.gz
# straighten t1
sct_apply_transfo -i t1_regAffineWarp.nii.gz -d ../t2/t2_RPI_crop_straight.nii.gz -w ../t2/warp_curve2straight.nii.gz -o t1_regAffineWarp_straight.nii.gz
# crop t1
sct_crop_image -i t1_regAffineWarp_straight.nii.gz -o t1_regAffineWarp_straight_crop.nii.gz -dim 2 -start 29 -end 552
# push t1 straight into template space
sct_apply_transfo -i t1_regAffineWarp_straight_crop.nii.gz -o t1_regAffineWarp_straight_crop_to_template.nii.gz -d ${SCT_DIR}/dev/template_creation/template_shape.nii.gz -w ../t2/native2temp.txt -p spline
# align to vertebrae
sct_apply_transfo -i t1_regAffineWarp_straight_crop_to_template.nii.gz -o t1_regAffineWarp_straight_crop_to_template_aligned.nii.gz -d ${SCT_DIR}/dev/template_creation/template_shape-mask.nii.gz -w ../t2/n2t.txt -p spline
# copy centerline
cp ../t2/centerline_straight_crop_2temp_aligned.nii.gz .
# normalize intensity within spinal cord
sct_normalize.py -i t1_regAffineWarp_straight_crop_to_template_aligned.nii.gz -c centerline_straight_crop_2temp_aligned.nii.gz
