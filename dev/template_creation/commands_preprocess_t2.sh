###Marseille :
##1:
#(nurbs 13)

sct_crop_image -i mar_1_t2.nii.gz -o mar_1_t2_crop.nii.gz -dim 2 -start 27 -end 559

sct_segmentation_propagation -i mar_1_t2_crop.nii.gz -t t2 -centerline-binary -init-mask mar_1_t2_crop-mask.nii.gz

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 49 -e 90

sct_erase_centerline.py -i centerline_erased.nii.gz -s 473 -e 532

sct_generate_centerline.py -i mask-up.nii.gz

sct_generate_centerline.py -i mask-down.nii.gz

fslmaths centerline_erased.nii.gz -add generated_centerline-down.nii.gz centerline_down.nii.gz

fslmaths centerline_down.nii.gz -add generated_centerline-up.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i mar_1_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz straight_centerline.nii.gz -R mar_1_t2_crop_straight.nii.gz warp_curve2straight.nii.gz

sct_detect_extrema.py -i straight_centerline.nii.gz 

sct_crop_image -i mar_1_t2_crop_straight.nii.gz -o mar_1_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 570

sct_create_cross.py -i mar_1_t2_crop_straight_crop.nii.gz -x 48 -y 158

sct_push_into_template_space.py -i mar_1_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz -t /home/django/jtouati/code/spinalcordtoolbox/dev/
template_creation/template_landmarks-mm.nii.gz -R /home/django/jtouati/code/spinalcordtoolbox/dev/template_creation/template_shape.nii.gz 

##7 (nurbs 13)

sct_crop_image -i mar_7_t2.nii.gz -o mar_7_t2_crop.nii.gz -start 22 -end 578 -dim 2

sct_segmentation_propagation -i mar_7_t2_crop.nii.gz -t t2 -centerline-binary -init-mask mar_7_t2_crop-mask.nii.gz

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 239

sct_erase_centerline.py -i centerline_erased.nii.gz -s 386 -e 556

sct_generate_centerline.py -i mask-up.nii.gz -o up.nii.gz

sct_generate_centerline.py -i mask-down.nii.gz -o down.nii.gz

fslmaths centerline_erased.nii.gz -add up.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add down.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i mar_7_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz straight_centerline.nii.gz -R mar_7_t2_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_detect_extrema.py -i straight_centerline.nii.gz

sct_crop_image -i mar_7_t2_crop_straight.nii.gz -o mar_7_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 592

sct_create_cross.py -i mar_7_t2_crop_straight_crop.nii.gz -x 53 -y 161

sct_push_into_template_space.py -i mar_7_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz


##errsm_03 (nurbs 20)

sct_crop_image -i errsm_03_t2.nii.gz -o errsm_03_t2_crop.nii.gz -dim 2 -start 23 -end 569

sct_segmentation_propagation -i errsm_03_t2_crop.nii.gz -t t2 -centerline-binary

sct_generate_centerline.py -i mask-up.nii.gz

fslmaths segmentation_binary.nii.gz -add generated_centerline.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_03_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz straight_centerline.nii.gz -R errsm_03_t2_crop_straight.nii.gz warp_curve2straight.nii.gz

sct_detect_extrema.py -i straight_centerline.nii.gz

sct_crop_image -i errsm_03_t2_crop_straight.nii.gz -o errsm_03_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 588

sct_create_cross.py -i errsm_03_t2_crop_straight_crop.nii.gz -x 55 -y 229

sct_push_into_template_space.py -i errsm_03_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz


## errsm 05 (20) 

sct_crop_image -i errsm_05_t2.nii.gz -o errsm_05_t2_crop.nii.gz -start 35 -end 528 -dim 2

sct_segmentation_propagation -i errsm_05_t2_crop.nii.gz -t t2 -centerline-binary

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 401 -e 493

sct_generate_centerline.py -i mask-up.nii.gz 

fslmaths centerline_erased.nii.gz -add generated_centerline.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_05_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_05_t2_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_detect_extrema.py -i centerline_straight.nii.gz

sct_crop_image -i errsm_05_t2_crop_straight.nii.gz -o errsm_05_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 542

sct_create_cross.py -i errsm_05_t2_crop_straight_crop.nii.gz -x 54 -y 221
 
sct_push_into_template_space.py -i errsm_05_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz  


## errsm_09 (20)

sct_crop_image -i errsm_09_t2.nii.gz -o errsm_09_t2_crop.nii.gz -dim 2 -start 52 -end 540

sct_segmentation_propagation -i errsm_09_t2_crop.nii.gz -t t2 -centerline-binary

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 426 -e 488

sct_generate_centerline.py -i mask-up.nii.gz

fslmaths centerline_erased.nii.gz -add generated_centerline.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_09_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz straight_centerline.nii.gz -R errsm_09_t2_crop_straight.nii.gz warp_curve2straight.nii.gz

sct_detect_extrema.py -i straight_centerline.nii.gz 

sct_crop_image -i errsm_09_t2_crop_straight.nii.gz -o errsm_09_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 528

sct_create_cross.py -i errsm_09_t2_crop_straight_crop.nii.gz -x 55 -y 226

sct_push_into_template_space.py -i errsm_09_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz

## errsm_10 (20)

sct_crop_image -i errsm_10_t2.nii.gz -o errsm_10_t2_crop.nii.gz -dim 2 -start 23 -end 568 

sct_segmentation_propagation -i errsm_10_t2_crop.nii.gz -t t2 -centerline-binary

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 76

sct_erase_centerline.py -i centerline_erased.nii.gz -s 390 -e 545

sct_generate_centerline.py -i mask-up.nii.gz -o up.nii.gz

sct_generate_centerline.py -i mask-down.nii.gz -o down.nii.gz

fslmaths centerline_erased.nii.gz -add up.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add down.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_10_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_10_t2_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_detect_extrema.py -i centerline_straight.nii.gz

sct_crop_image -i errsm_10_t2_crop_straight.nii.gz -o errsm_10_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 589

sct_create_cross.py -i errsm_10_t2_crop_straight_crop.nii.gz -x 53 -y 242

sct_push_into_template_space.py -i errsm_10_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

## errsm_11 (20)

sct_crop_image -i errsm_11_t2.nii.gz -o errsm_11_t2_crop.nii.gz -dim 2 -start 17 -end 557

sct_segmentation_propagation -i errsm_11_t2_crop.nii.gz -t t2 -centerline-binary

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 108

sct_erase_centerline.py -i centerline_erased.nii.gz -s 460 -e 540

sct_generate_centerline.py -i mask-up.nii.gz -o upo.nii.gz

sct_generate_centerline.py -i mask-down.nii.gz -o down.nii.gz

fslmaths centerline_erased.nii.gz -add upo.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add down.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_11_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_11_t2_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_detect_extrema.py -i centerline_straight.nii.gz 

sct_crop_image -i errsm_11_t2_crop_straight.nii.gz -o errsm_11_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 590

sct_create_cross.py -i errsm_11_t2_crop_straight_crop.nii.gz -x 55 -y 222

sct_push_into_template_space.py -i errsm_11_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

## errsm_12 (20)

sct_crop_image -i errsm_12_t2.nii.gz -o errsm_12_t2_crop.nii.gz -dim 2 -start 115 -end 610

sct_segmentation_propagation -i errsm_12_t2_crop.nii.gz -t t2 -centerline-binary -init-mask init.nii.gz 

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 437 -e 495

sct_generate_centerline.py -i up.nii.gz 

fslmaths centerline_erased.nii.gz -add generated_centerline.nii.gz full_centerline.nii.gz

sct_erase_centerline.py -i full_centerline.nii.gz -s 0 -e 25

fslmaths centerline_erased.nii.gz -add generated_centerline.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_12_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_12_t2_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_crop_image -i errsm_12_t2_crop_straight.nii.gz -o errsm_12_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 540

sct_create_cross.py -i errsm_12_t2_crop_straight_crop.nii.gz -x 55 -y 221

sct_push_into_template_space.py -i errsm_12_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

## errsm_13 (20)

sct_crop_image -i errsm_13_t2.nii.gz -o errsm_13_t2_crop.nii.gz -dim 2 -start 21 -end 618 

sct_segmentation_propagation -i errsm_13_t2_crop.nii.gz -t t2 -centerline-binary

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 117

sct_generate_centerline.py -i up.nii.gz -o center_up.nii.gz

sct_generate_centerline.py -i down.nii.gz -o center_down.nii.gz

fslmaths centerline_erased.nii.gz -add center_up.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add center_down.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_13_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_13_t2_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_detect_extrema.py -i centerline_straight.nii.gz 

sct_crop_image -i errsm_13_t2_crop_straight.nii.gz -o errsm_13_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 637

sct_create_cross.py -i errsm_13_t2_crop_straight_crop.nii.gz -x 54 -y 221

sct_push_into_template_space.py -i errsm_13_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz


## errsm_14 (20)

sct_crop_image -i errsm_14_t2.nii.gz -o errsm_14_t2_crop.nii.gz -dim 2 -start 42 -end 515

sct_segmentation_propagation -i errsm_14_t2_crop.nii.gz -t t2 -centerline-binary -init-mask init.nii.gz

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 74

sct_generate_centerline.py -i up.nii.gz -o centerup.nii.gz

sct_generate_centerline.py -i down.nii.gz -o centerdown.nii.gz

fslmaths centerline_erased.nii.gz -add centerup.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add centerdown.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_14_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_14_t2_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_detect_extrema.py -i centerline_straight.nii.gz 

sct_crop_image -i errsm_14_t2_crop_straight.nii.gz -o errsm_14_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 516

sct_create_cross.py -i errsm_14_t2_crop_straight_crop.nii.gz -x 54 -y 221

sct_push_into_template_space.py -i errsm_14_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


## errsm_16 (20)

sct_crop_image -i errsm_16_t2.nii.gz -o errsm_16_t2_crop.nii.gz -dim 2 -start 114 -end 618

sct_segmentation_propagation -i errsm_16_t2_crop.nii.gz -t t2 -centerline-binary -init-mask init.nii.gz

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 36

sct_erase_centerline.py -i centerline_erased.nii.gz -s 442 -e 504

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz

sct_generate_centerline.py -i down.nii.gz -o cdown.nii.gz

fslmaths centerline_erased.nii.gz -add cup.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add cdown.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_16_t2_crop.nii.gz -c full_centerline.nii.gz -v 2 

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_16_t2_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_detect_extrema.py -i centerline_straight.nii.gz 

sct_crop_image -i errsm_16_t2_crop_straight.nii.gz -o errsm_16_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 546 

sct_create_cross.py -i errsm_16_t2_crop_straight_crop.nii.gz -x 54 -y 221 

sct_push_into_template_space.py -i errsm_16_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz 



## errsm_17 (20)
sct_crop_image -i errsm_17_t2.nii.gz -o errsm_17_t2_crop.nii.gz -dim 2 -start 51 -end 595

sct_segmentation_propagation -i errsm_17_t2_crop.nii.gz -t t2 -centerline-binary

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz

sct_generate_centerline.py -i down.nii.gz -o cdown.nii.gz

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 127

sct_erase_centerline.py -i centerline_erased.nii.gz -s 476 -e 544

sct_straighten_spinalcord.py -i errsm_17_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_17_t2_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_detect_extrema.py -i centerline_straight.nii.gz 

sct_crop_image -i errsm_17_t2_crop_straight.nii.gz -o errsm_17_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 585

sct_create_cross.py -i errsm_17_t2_crop_straight_crop.nii.gz -x 54 -y 221

sct_push_into_template_space.py -i errsm_17_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

## errsm_17 BIS (30)

sct_crop_image -i errsm_17_t2.nii.gz -o errsm_17_t2_crop.nii.gz -dim 2 -start 86 -end 595

sct_segmentation_propagation -i errsm_17_t2_crop.nii.gz -t t2 -centerline-binary

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 90

sct_erase_centerline.py -i centerline_erased.nii.gz -s 445 -e 509

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz

sct_generate_centerline.py -i down.nii.gz -o cdown.nii.gz

fslmaths centerline_erased.nii.gz -add cup.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add cdown.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_17_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_17_t2_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_detect_extrema.py -i centerline_straight.nii.gz 

sct_crop_image -i errsm_17_t2_crop_straight.nii.gz -o errsm_17_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 550

sct_create_cross.py -i errsm_17_t2_crop_straight_crop.nii.gz -x 55 -y 222


## errsm_18 (20)

sct_crop_image -i errsm_18_t2.nii.gz -o errsm_18_t2_crop.nii.gz -dim 2 -start 16 -end 567

sct_segmentation_propagation -i errsm_18_t2_crop.nii.gz -t t2 -centerline-binary

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 104

sct_erase_centerline.py -i centerline_erased.nii.gz -s 481 -e 551

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz

sct_generate_centerline.py -i down.nii.gz -o cdown.nii.gz

fslmaths centerline_erased.nii.gz -add cup.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add cdown.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_18_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_18_t2_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_detect_extrema.py -i centerline_straight.nii.gz

sct_crop_image -i errsm_18_t2_crop_straight.nii.gz -o errsm_18_t2_crop_straight_straight.nii.gz -start 30 -end 591 -dim 2

sct_create_cross.py -i errsm_18_t2_crop_straight_straight.nii.gz -x 54 -y 222

sct_push_into_template_space.py -i errsm_18_t2_crop_straight_straight.nii.gz -n landmark_native.nii.gz 


## errsm_18 BIS (30)

sct_crop_image -i errsm_18_t2.nii.gz -o errsm_18_t2_crop.nii.gz -dim 2 -start 15 -end 568

sct_segmentation_propagation -i errsm_18_t2_crop.nii.gz -t t2 -centerline-binary

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 122

sct_erase_centerline.py -i centerline_erased.nii.gz -s 471 -e 553

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz

sct_generate_centerline.py -i down.nii.gz -o cdown.nii.gz

fslmaths centerline_erased.nii.gz -add cup.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add cdown.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_18_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_18_t2_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_crop_image -i errsm_18_t2_crop_straight.nii.gz -o errsm_18_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 593

sct_create_cross.py -i errsm_18_t2_crop_straight_crop.nii.gz -x 24 -y 222

sct_push_into_template_space.py -i errsm_18_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz


## errsm_20 (30)

sct_crop_image -i errsm_20_t2.nii.gz -o errsm_20_t2_crop.nii.gz -dim 2 -start 39 -end 604 

sct_segmentation_propagation -i errsm_20_t2_crop.nii.gz -t t2 -centerline-binary -init-mask init.nii.gz

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 117

sct_erase_centerline.py -i centerline_erased.nii.gz -s 338 -e 565

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz

sct_generate_centerline.py -i down.nii.gz -o cdown.nii.gz

fslmaths semi.nii.gz -add cup.nii.gz full_centerline.nii.gz

fslmaths semi.nii.gz -add cup.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_20_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_20_t2_crop_straight.nii.gz warp_curve2straight.nii.gz 

sct_detect_extrema.py -i centerline_straight.nii.gz

sct_crop_image -i errsm_20_t2_crop_straight.nii.gz -o errsm_20_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 624 

sct_create_cross.py -i errsm_20_t2_crop_straight_crop.nii.gz -x 55 -y 223

sct_push_into_template_space.py -i errsm_20_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


## errsm_21 (30)

sct_crop_image -i errsm_21_t2.nii.gz -o errsm_21_t2_crop.nii.gz -dim 2 -start 48 -end 591

sct_segmentation_propagation -i errsm_21_t2_crop.nii.gz -t t2 -centerline-binary 

sct_erase_centerline.py -i centerline_erased.nii.gz -s 465 -e 543

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz

sct_generate_centerline.py -i down.nii.gz -o cdown.nii.gz

fslmaths centerline_erased.nii.gz -add cdown.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add cup.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_21_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_21_t2_crop_straight.nii.gz warp_curve2straight.nii.gz

sct_detect_extrema.py -i centerline_straight.nii.gz 

sct_crop_image -i errsm_21_t2_crop_straight.nii.gz -o errsm_21_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 585 

sct_create_cross.py -i errsm_21_t2_crop_straight_crop.nii.gz -x 51 -y 221
 
sct_push_into_template_space.py -i errsm_21_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz



## errsm_22 (30)

sct_crop_image -i errrsm_22_t2.nii.gz -o errsm_22_t2_crop.nii.gz -dim 2 -start 65 -end 582

sct_segmentation_propagation -i errsm_22_t2_crop.nii.gz -t t2 -centerline-binary -init-mask init.nii.gz

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 72

sct_erase_centerline.py -i centerline_erased.nii.gz -s 376 -e 517

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz

sct_generate_centerline.py -i down.nii.gz -o cdown.nii.gz

fslmaths centerline_erased.nii.gz -add cup.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add cdown.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_22_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_22_t2_crop_straight.nii.gz warp_curve2straight.nii.gz

sct_detect_extrema.py -i centerline_straight.nii.gz

sct_crop_image -i errsm_22_t2_crop_straight.nii.gz -o errsm_22_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 558 

sct_create_cross.py -i errsm_22_t2_crop_straight_crop.nii.gz -x 54 -y 222 

sct_push_into_template_space.py -i errsm_22_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


# errsm_23 (30)

sct_crop_image -i errsm_23_t2.nii.gz -o errsm_23_t2_crop.nii.gz -dim 2 -start 62 -end 582

sct_segmentation_propagation -i errsm_23_t2_crop.nii.gz -t t2 -centerline-binary
 
sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 113

sct_erase_centerline.py -i centerline_erased.nii.gz -s 429 -e 520

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz

sct_generate_centerline.py -i down.nii.gz -o cdown.nii.gz

fslmaths centerline_erased.nii.gz -add cup.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add cdown.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_23_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_23_t2_crop_straight.nii.gz warp_curve2straight.nii.gz

sct_detect_extrema.py -i centerline_straight.nii.gz 

sct_crop_image -i errsm_23_t2_crop_straight.nii.gz -o errsm_23_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 557

sct_create_cross.py -i errsm_23_t2_crop_straight_crop.nii.gz -x 55 -y 222

sct_push_into_template_space.py -i errsm_23_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


## errsm_24 (30)

sct_crop_image -i errsm_24_t2.nii.gz -o errsm_24_t2_crop.nii.gz -dim 2 -start 26 -end 579

sct_segmentation_propagation -i errsm_24_t2_crop.nii.gz -t t2 -centerline-binary

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 486 -e 553 

sct_generate_centerline.py -i up.nii.gz

fslmaths centerline_erased.nii.gz -add generated_centerline.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_24_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_24_t2_crop_straight.nii.gz warp_curve2straight.nii.gz

sct_detect_extrema.py -i centerline_straight.nii.gz

sct_crop_image -i errsm_24_t2_crop_straight.nii.gz -o errsm_24_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 597

sct_create_cross.py -i errsm_24_t2_crop_straight_crop.nii.gz -x 55 -y 221

sct_push_into_template_space.py -i errsm_24_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz 


## errsm_25 (30)

## centerline fait main quasiment

sct_crop_image -i errsm_25_t2.nii.gz -o errsm_25_t2_crop.nii.gz -dim 2 -start 105 -end 614

sct_segmentation_propagation -i errsm_25_t2_crop.nii.gz -t t2 -centerline-binary -init-mask init.nii.gz

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 0 -e 92sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz -v 2

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz -v 2

sct_generate_centerline.py -i down.nii.gz -o cdown.nii.gz -v 2

fslmaths centerline_erased.nii.gz -add cup.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add cdown.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_25_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_25_t2_crop_straight.nii.gz warp_curve2straight.nii.gz

sct_detect_extrema.py -i centerline_straight.nii.gz 

sct_crop_image -i errsm_25_t2_crop_straight.nii.gz -o errsm_25_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 556

sct_create_cross.py -i errsm_25_t2_crop_straight_crop.nii.gz -x 55 -y 221

sct_push_into_template_space.py -i errsm_25_t2_crop_straight_crop.nii.gz -n landmark_native.nii.gz 

## errsm_30 (30)

sct_crop_image -i errsm_30_t2.nii.gz -o errsm_30_t2_crop.nii.gz -dim 2 -start 37 -end 576

sct_segmentation_propagation -i errsm_30_t2_crop.nii.gz -t t2 -centerline-binary

sct_erase_centerline.py -i segmentation_centerline_binary.nii.gz -s 409 -e 539

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz

sct_generate_centerline.py -i up.nii.gz -o cup.nii.gz

sct_generate_centerline.py -i down.nii.gz -o cdown.nii.gz

fslmaths centerline_erased.nii.gz -add cup.nii.gz semi.nii.gz

fslmaths semi.nii.gz -add cdown.nii.gz full_centerline.nii.gz

sct_straighten_spinalcord.py -i errsm_30_t2_crop.nii.gz -c full_centerline.nii.gz -v 2

sct_WarpImageMultiTransform 3 full_centerline.nii.gz centerline_straight.nii.gz -R errsm_30_t2_crop_straight.nii.gz warp_curve2straight.nii.gz

sct_detect_extrema.py -i centerline_straight.nii.gz 

sct_crop_image -i errsm_30_t2_crop_straight.nii.gz -o errsm_30_t2_crop_straight_crop.nii.gz -dim 2 -start 30 -end 583

sct_create_cross.py -i errsm_30_t2_crop_straight_crop.nii.gz -x 54 -y 221

sct_push_into_template_space.py -i errsm_30_t2_crop_straight_crop.nii.gz -n landmark_native.ni.gz 













  




