#!/bin/bash


 
# #ALT
#
# sct_crop_image -i ALT_t1.nii.gz -o ALT_t1_crop.nii.gz -start 0 -end 533 -dim 2
#
# sct_straighten_spinalcord -i ALT_t1_crop.nii.gz -c full_centerline.nii.gz -v 2
#
# sct_crop_image -i ALT_t1_crop_straight.nii.gz -o ALT_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 576
#
# sct_create_cross.py -i ALT_t1_crop_straight_crop.nii.gz -x 51 -y 162
#
# sct_push_into_template_space.py -i ALT_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz
#
#
# #JD
#
# sct_crop_image -i JD_t1.nii.gz -o JD_t1_crop.nii.gz -dim 2 -start 0 -end 545
#
# sct_straighten_spinalcord -i JD_t1_crop.nii.gz -c full_centerline.nii.gz -v 2
#
# sct_crop_image -i JD_t1_crop_straight.nii.gz -o JD_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 585
#
# sct_create_cross.py -i JD_t1_crop_straight_crop.nii.gz -x 54 -y 162
#
# sct_push_into_template_space.py -i JD_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz
#
#
# #JW
#
# sct_crop_image -i JW_t1.nii.gz -o JW_t1_crop.nii.gz -dim 2 -start 0 -end 516
#
# sct_straighten_spinalcord -i JW_t1_crop.nii.gz -c full_centerline.nii.gz
#
# sct_crop_image -i JW_t1_crop_straight.nii.gz -o JW_t1_crop_straight_crop.nii.gz -dim 2 -start 29 -end 551
#
# sct_create_cross.py -i JW_t1_crop_straight_crop.nii.gz -x 56 -y 160
#
# sct_push_into_template_space.py -i JW_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz
#

cd /home/django/jtouati/data/template_data/new_data/Marseille/MT/T1

#MT
sct_crop_image -i MT_t1.nii.gz -o MT_t1_crop.nii.gz -dim 2 -start 0 -end 515

sct_straighten_spinalcord -i MT_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i MT_t1_crop_straight.nii.gz -o MT_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 558

sct_create_cross.py -i MT_t1_crop_straight_crop.nii.gz -x 56 -y 161

sct_push_into_template_space.py -i MT_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

cd ../../T047/T1

#T047

sct_crop_image -i T047_t1.nii.gz -o T047_t1_crop.nii.gz -start 0 -end 553 -dim 2

sct_straighten_spinalcord -i T047_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i T047_t1_crop_straight.nii.gz -o T047_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 588 

sct_create_cross.py -i T047_t1_crop_straight_crop.nii.gz -x 55 -y 160

sct_push_into_template_space.py -i T047_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

cd ../../VG/T1

#VG

sct_crop_image -i VG_t1.nii.gz -o VG_t1.nii.gz -dim 1 -start 11 -end 246

sct_crop_image -i VG_t1.nii.gz -o VG_t1_crop.nii.gz -dim 2 -start 71 -end 549

sct_straighten_spinalcord -i VG_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i VG_t1_crop_straight.nii.gz -o VG_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 517 

sct_create_cross.py -i VG_t1_crop_straight_crop.nii.gz -x 55 -y 149

sct_push_into_template_space.py -i VG_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

cd ../../VP/T1

#VP

sct_crop_image -i VP_t1.nii.gz -o VP_t1_crop.nii.gz -dim 2 -start 0 -end 528 

sct_straighten_spinalcord -i VP_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i VP_t1_crop_straight.nii.gz -o VP_t1_crop_straight_crop.nii.gz -dim 2 -start 29 -end 565

sct_create_cross.py -i VP_t1_crop_straight_crop.nii.gz -x 55 -y 161

sct_push_into_template_space.py -i VP_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

cd ../../../Montreal/errsm_02/T1

#errsm02

sct_crop_image -i errsm_02_t1.nii.gz -o errsm_02_t1_crop.nii.gz -dim 2 -start 100 -end 620

sct_straighten_spinalcord -i errsm_02_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i errsm_02_t1_crop_straight.nii.gz -o errsm_02_t1_crop_straight_crop.nii.gz -dim 2 -start 29 -end 559

sct_create_cross.py -i errsm_02_t1_crop_straight_crop.nii.gz -x 54 -y 147

sct_push_into_template_space.py -i errsm_02_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

cd ../../errsm_04/T1

#errsm_04 

sct_crop_image -i errsm_04_t1.nii.gz -o errsm_04_t1_crop.nii.gz -dim 2 -start 73 -end 549

sct_straighten_spinalcord -i errsm_04_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i errsm_04_t1_crop_straight.nii.gz -o errsm_04_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 516

sct_create_cross.py -i errsm_04_t1_crop_straight_crop.nii.gz -x 65 -y 231

sct_push_into_template_space.py -i errsm_04_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

cd ../../errsm_31/T1

#errsm_31

sct_crop_image -i errsm_31_t1.nii.gz -o errsm_31_t1_crop.nii.gz -dim 2 -start 0 -end 593

sct_straighten_spinalcord -i errsm_31_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i errsm_31_t1_crop_straight.nii.gz -o errsm_31_t1_crop_straight_crop.nii.gz -dim 2 -start 30 -end 627

sct_create_cross.py -i errsm_31_t1_crop_straight_crop.nii.gz -x 55 -y 223

sct_push_into_template_space.py -i errsm_31_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

cd ../../errsm_32/T1

# errsm_32

sct_crop_image -i errsm_32_t1.nii.gz -o errsm_32_t1_crop.nii.gz -dim 2 -start 0 -end 552

sct_straighten_spinalcord -i errsm_32_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i errsm_32_t1_crop_straight.nii.gz -o errsm_32_t1_crop_straight_crop.nii.gz -dim 2 -start 29 -end 589

sct_create_cross.py -i errsm_32_t1_crop_straight_crop.nii.gz -x 52 -y 225

sct_push_into_template_space.py -i errsm_32_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

cd ../../errsm_33/T1

# errsm_33

sct_crop_image -i errsm_33_t1.nii.gz -o errsm_33_t1_crop.nii.gz -dim 2 -start 0 -end 548

sct_straighten_spinalcord -i errsm_33_t1_crop.nii.gz -c full_centerline.nii.gz

sct_crop_image -i errsm_33_t1_crop_straight.nii.gz -o errsm_33_t1_crop_straight_crop.nii.gz -dim 2 -start 29 -end 587

sct_create_cross.py -i errsm_33_t1_crop_straight_crop.nii.gz -x 55 -y 225

sct_push_into_template_space.py -i errsm_33_t1_crop_straight_crop.nii.gz -n landmark_native.nii.gz 
