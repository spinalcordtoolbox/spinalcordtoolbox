#!/bin/bash
#
# Example of commands to process multi-parametric data of the spinal cord using the PAM50 template.
# For information about acquisition parameters, see: https://dl.dropboxusercontent.com/u/20592661/publications/Fonov_NIMG14_MNI-Poly-AMU.pdf
# N.B. The parameters are set for these type of data. With your data, parameters might be slightly different.
#
# tested with v2.1_beta20

# download the PAM50 template
sct_download_data -d PAM50

# create variable to template
SCT_PAM50=`pwd`/PAM50

# download example data (errsm_30)
sct_download_data -d sct_example_data

# display starting time:
echo "Started at: $(date +%x_%r)"

# go in folder
cd sct_example_data

# t2
# ===========================================================================================
cd t2
# spinal cord segmentation
sct_propseg -i t2.nii.gz -c t2 -mesh -max-deformation 4 -init 130
# tips: we use "-max-deformation 4" otherwise the segmentation does not cover the whole spinal cord
# tips: we use "-init 130" to start propagation closer to a region which would otherwise give poor segmentation (try it with and without the parameter).
# tips: we use "-mesh" to get the mesh of the segmentation, which can be viewed using MITKWORKBENCH
# check your results:
fslview t2 -b 0,800 t2_seg -l Red -t 0.5 &
# vertebral labeling. Here we use the fact that the FOV is centered at C7.
sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -initcenter 7
# create labels at C2 and T2 vertebral levels
sct_label_utils -i t2_seg_labeled.nii.gz -p label-vertebrae -vert 2,9
# register to template
# tips: here we used only iter=1 for the third step for faster processing. 
sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -l labels.nii.gz -t $SCT_PAM50/template -param step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=seg,algo=bsplinesyn,iter=3,shrink=1,metric=MeanSquares:step=3,type=im,algo=syn,metric=CC,iter=1 -r 0
# warp template and white matter atlas
sct_warp_template -d t2.nii.gz -w warp_template2anat.nii.gz -t $SCT_PAM50 -a 0
# check results
fslview t2.nii.gz -b 0,800 label/template/MNI-Poly-AMU_T2.nii.gz -b 0,4000 label/template/MNI-Poly-AMU_level.nii.gz -l MGH-Cortical -t 0.5 label/template/MNI-Poly-AMU_GM.nii.gz -l Red-Yellow -b 0.5,1 label/template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.5,1 &
# compute average cross-sectional area and volume between C3 and C4 levels
sct_process_segmentation -i t2_seg.nii.gz -p csa -t label/template/MNI-Poly-AMU_level.nii.gz -vert 3:4
# --> Mean CSA: 78.2927207039 +/- 0.770570121273 mm^2
# --> Volume (in volume.txt): 2596.0 mm^3

# go back to root folder
cd ..


#  t1
# ----------
cd t1
# spinal cord segmentation
sct_propseg -i t1.nii.gz -c t1
# check results
fslview t1 -b 0,800 t1_seg -l Red -t 0.5 &
# create mask around spinal cord (for cropping)
sct_create_mask -i t1.nii.gz -p centerline,t1_seg.nii.gz -size 61 -f box -o mask_t1.nii.gz
# crop t1 and t1_seg (for faster registration)
sct_crop_image -i t1.nii.gz -m mask_t1.nii.gz -o t1_crop.nii.gz
sct_crop_image -i t1_seg.nii.gz -m mask_t1.nii.gz -o t1_seg_crop.nii.gz
# register to template (which was previously registered to the t2).
sct_register_multimodal -i ../t2/label/template/MNI-Poly-AMU_T1.nii.gz -iseg ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -d t1_crop.nii.gz -dseg t1_seg_crop.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=im,algo=syn,iter=3,gradStep=0.2,metric=CC
# concatenate transformations
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_MNI-Poly-AMU_T12t1_crop.nii.gz -d t1.nii.gz -o warp_template2t1.nii.gz
sct_concat_transfo -w warp_t1_crop2MNI-Poly-AMU_T1.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_PAM50/template/MNI-Poly-AMU_T1.nii.gz -o warp_t12template.nii.gz
# warp template
sct_warp_template -d t1.nii.gz -w warp_template2t1.nii.gz -t $SCT_PAM50 -a 0
# check results
fslview t1.nii.gz label/template/MNI-Poly-AMU_T1.nii.gz -b 10,1500 label/template/MNI-Poly-AMU_level.nii.gz -l MGH-Cortical -t 0.5 label/template/MNI-Poly-AMU_GM.nii.gz -l Red-Yellow -b 0.5,1 label/template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.5,1 &
cd ..

# display ending time:
echo "Ended at: $(date +%x_%r)"
