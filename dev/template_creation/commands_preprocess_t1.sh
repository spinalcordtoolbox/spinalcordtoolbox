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













  




