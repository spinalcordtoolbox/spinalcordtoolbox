#!/bin/bash
#
# Example of commands to process multi-parametric data of the spinal cord
# For information about acquisition parameters, see: https://dl.dropboxusercontent.com/u/20592661/publications/Fonov_NIMG14_MNI-Poly-AMU.pdf
# N.B. The parameters are set for these type of data. With your data, parameters might be slightly different.


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
sct_propseg -i t2.nii.gz -t t2 -mesh -max-deformation 4 -init 130
# tips: we use "-max-deformation 4" otherwise the segmentation does not cover the whole spinal cord
# tips: we use "-init 130" to start propagation closer to a region which would otherwise give poor segmentation (try it with and without the parameter).
# tips: we use "-mesh" to get the mesh of the segmentation, which can be viewed using MITKWORKBENCH
# check your results:
fslview t2 -b 0,800 t2_seg -l Red -t 0.5 &
# vertebral labeling. Here we use the fact that the FOV is centered at C7.
sct_label_vertebrae -i t2.nii.gz -seg t2_seg.nii.gz -initcenter 7
# create labels at C2 and T2 vertebral levels
sct_label_utils -i t2_seg_labeled.nii.gz -t label-vertebrae -level 2,9
# register to template
sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -l labels.nii.gz -p step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=seg,algo=bsplinesyn,iter=5,shrink=2,metric=MeanSquares:step=3,type=im,algo=syn,iter=3,shrink=1,metric=CC
# warp template and white matter atlas
sct_warp_template -d t2.nii.gz -w warp_template2anat.nii.gz
# check results
fslview t2.nii.gz -b 0,800 label/template/MNI-Poly-AMU_T2.nii.gz -b 0,4000 label/template/MNI-Poly-AMU_level.nii.gz -l MGH-Cortical -t 0.5 label/template/MNI-Poly-AMU_GM.nii.gz -l Red-Yellow -b 0.5,1 label/template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.5,1 &
# compute average cross-sectional area between C2 and C4 levels
sct_process_segmentation -i t2_seg.nii.gz -p csa -t label/template -l 2:4
# --> mean CSA: 77.1679472754 +/- 1.08332424568
# go back to root folder
cd ..


#  t1
# ----------
cd t1
# spinal cord segmentation
sct_propseg -i t1.nii.gz -t t1
# check results
fslview t1 -b 0,800 t1_seg -l Red -t 0.5 &
# adjust segmentation (it was not perfect)
# --> t1_seg_modif.nii.gz
# register to template (which was previously registered to the t2).
sct_register_multimodal -i ../t2/label/template/MNI-Poly-AMU_T2.nii.gz -iseg ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -d t1.nii.gz -dseg t1_seg.nii.gz -p step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=im,algo=syn,iter=3,gradStep=0.2,metric=CC
# concatenate transformations
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_MNI-Poly-AMU_T22t1.nii.gz -d t1.nii.gz -o warp_template2t1.nii.gz
sct_concat_transfo -w warp_t12MNI-Poly-AMU_T2.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_t12template.nii.gz
# warp template
sct_warp_template -d t1.nii.gz -w warp_template2t1.nii.gz -a 0
# check results
fslview t1.nii.gz label/template/MNI-Poly-AMU_T2.nii.gz -b 0,4000 label/template/MNI-Poly-AMU_level.nii.gz -l MGH-Cortical -t 0.5 label/template/MNI-Poly-AMU_GM.nii.gz -l Red-Yellow -b 0.5,1 label/template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.5,1 &
# warp T1 to template space
sct_apply_transfo -i t1.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -w warp_t12template.nii.gz
# check registration of T1 to template
fslview $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -b 0,4000 t1_reg.nii.gz -b 0,800 &
# go back to root folder
cd ..


# mt
# ----------
cd mt
# bring T2 segmentation in MT space to help segmentation (no optimization)
sct_register_multimodal -i ../t2/t2_seg.nii.gz -d mt1.nii.gz -p step=1,iter=0 -x nn
# extract centerline
sct_process_segmentation -i t2_seg_reg.nii.gz -p centerline -o t2_seg_reg_centerline.nii.gz
# segment mt1
sct_propseg -i mt1.nii.gz -t t2 -init-centerline t2_seg_reg_centerline.nii.gz
# check results
fslview mt1 -b 0,800 mt1_seg.nii.gz -l Red -t 0.5 &
# use centerline to create mask encompassing the spinal cord (will be used for improved registration of mt0 on mt1)
sct_create_mask -i mt1.nii.gz -m centerline,mt1_seg.nii.gz -s 60 -f cylinder
# register mt0 on mt1
sct_register_multimodal -i mt0.nii.gz -d mt1.nii.gz -z 3 -m mask_mt1.nii.gz -p step=1,type=im,algo=slicereg,metric=MI:step=2,type=im,algo=bsplinesyn,metric=CC,iter=3,gradStep=0.2
# compute mtr
sct_compute_mtr -i mt0_reg.nii.gz -j mt1.nii.gz
# register template (in T2 space) to mt1
sct_register_multimodal -i ../t2/template2anat.nii.gz -d mt1.nii.gz -iseg ../t2/t2_seg.nii.gz -dseg mt1_seg.nii.gz -p step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3:step=3,type=im,algo=syn,metric=CC,iter=1,shrink=1
# concat transfo
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_template2anat2mt1.nii.gz -d mt1.nii.gz -o warp_template2mt.nii.gz
# warp template (to get vertebral labeling)
sct_warp_template -d mt1.nii.gz -w warp_template2mt.nii.gz -a 0
# OPTIONAL PART: SEGMENT GRAY MATTER:
# -----------------------------------
# add mt1 and mt0 to increase GM/WM contrast
sct_maths -i mt0_reg.nii.gz -add mt1.nii.gz -o mt0mt1.nii.gz
# segment GM
sct_segment_graymatter -i mt0mt1.nii.gz -s mt1_seg.nii.gz -l label/template/MNI-Poly-AMU_level.nii.gz
# create mask around spinal cord for faster registration
sct_create_mask -i mt1.nii.gz -m centerline,mt1_seg.nii.gz -s 51 -f box
# register WM template to WMseg
sct_register_multimodal -i label/template/MNI-Poly-AMU_WM.nii.gz -d mt0mt1_wmseg.nii.gz -p step=1,algo=slicereg,metric=MeanSquares:step=2,algo=bsplinesyn,metric=MeanSquares,iter=3
# concat transfo
sct_concat_transfo -w warp_template2mt.nii.gz,warp_MNI-Poly-AMU_WM2mt1_wmseg.nii.gz -d mt1.nii.gz -o warp_template2mt1_corrected.nii.gz
# warp template (final warp template for mt1)
sct_warp_template -d mt1.nii.gz -w warp_template2mt1_corrected.nii.gz
# check registration result
fslview mt1.nii.gz label/template/MNI-Poly-AMU_T2.nii.gz -b 0,4000 label/template/MNI-Poly-AMU_level.nii.gz -l MGH-Cortical -t 0.5 label/template/MNI-Poly-AMU_GM.nii.gz -l Red-Yellow -b 0.5,1 label/template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.5,1 &
# extract MTR within the white matter
sct_extract_metric -i mtr.nii.gz -f label/atlas/ -l wm -m map
# --> MTR = 34.4617644116
# go back to root folder
cd ..


# dmri
# ----------
cd dmri
# create mask to help moco
sct_create_mask -i dmri.nii.gz -m coord,110x20 -s 60 -f cylinder
# motion correction
sct_dmri_moco -i dmri.nii.gz -b bvecs.txt -g 3 -m mask_dmri.nii.gz -p 2,2,1,MeanSquares -t 0
# detect approximate spinal cord centerline
sct_get_centerline -method auto -i dwi_moco_mean.nii.gz -t t1 -sym 1
# fine segmentation with propseg
sct_propseg -i dwi_moco_mean.nii.gz -t t1 -init-centerline dwi_moco_mean_centerline.nii.gz
# check segmentation
fslview dwi_moco_mean dwi_moco_mean_seg -l Red -t 0.5 & 
# register template to dwi: here we use the template register to the MT to get the correction of the internal structure
sct_register_multimodal -i ../mt/template2anat_reg.nii.gz -d dwi_moco_mean.nii.gz -iseg ../mt/mt1_seg.nii.gz -dseg dwi_moco_mean_seg.nii.gz -p step=1,type=seg,algo=slicereg,metric=MeanSquares,smooth=2:step=2,type=im,algo=bsplinesyn,metric=MeanSquares,iter=5,gradStep=0.5
# concatenate transfo: the first warping field contains : warp template -> anat ; warp anat --> MT ; warp correction of the internal structure 
sct_concat_transfo -w ../mt/warp_template2mt1_corrected.nii.gz,warp_template2anat_reg2dwi_moco_mean.nii.gz -d dwi_moco_mean.nii.gz -o warp_template2dmri.nii.gz
# warp template and white matter atlas
sct_warp_template -d dwi_moco_mean.nii.gz -w warp_template2dmri.nii.gz
# visualize white matter template and lateral CST on DWI
fslview dwi_moco_mean label/template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.2,1 -t 0.5 label/atlas/WMtract__02.nii.gz -b 0.2,1 -l Red label/atlas/WMtract__17.nii.gz -b 0.2,1 -l Yellow &
# compute DTI metrics
sct_dmri_compute_dti -i dmri_moco.nii.gz -bvals bvals.txt -bvecs bvecs.txt
# compute FA within right and left lateral corticospinal tracts from slices 1 to 3 using maximum a posteriori
sct_extract_metric -i dti_FA.nii.gz -f label/atlas/ -l 2,17 -z 1:3 -m map
# --> 17, right lateral corticospinal tract:    0.771558633821 +/- 0
# --> 2, left lateral corticospinal tract:    0.794731973648 +/- 0
# go back to root folder
cd ..


# fmri
# ----------
cd fmri
# create mask at the center of the FOV (will be used for moco)
sct_create_mask -i fmri.nii.gz -m center -s 30 -f cylinder
# moco
sct_fmri_moco -i fmri.nii.gz -m mask_fmri.nii.gz
# tips: if you have low SNR you can group consecutive images with "-g"
# put T2 segmentation into fmri space
sct_register_multimodal -i ../t2/t2_seg.nii.gz -d fmri_moco_mean.nii.gz -p step=1,iter=0
# extract centerline
sct_process_segmentation -i t2_seg_reg.nii.gz -p centerline
# segment mean fMRI volume
sct_propseg -i fmri_moco_mean.nii.gz -t t2 -init-centerline t2_seg_reg_centerline.nii.gz -radius 5 -max-deformation 4
# tips: we use the T2 segmentation to help with fMRI segmentation
# tips: we use "-radius 5" otherwise the segmentation is too small
# tips: we use "-max-deformation 4" to prevent the propagation from stopping at the edge
# check segmentation
fslview fmri_moco_mean fmri_moco_mean_seg -l Red -t 0.5 &
# here segmentation slightly failed due to the close proximity of susceptibility artifact --> use file "fmri_moco_mean_seg_modif.nii.gz"
# register template to fmri: here we use the template register to the MT to get the correction of the internal structure
sct_register_multimodal -i ../mt/template2anat_reg.nii.gz -d fmri_moco_mean.nii.gz -iseg ../mt/mt1_seg.nii.gz -dseg fmri_moco_mean_seg_modif.nii.gz -p step=1,type=seg,algo=slicereg,metric=MeanSquares,smooth=2:step=2,type=im,algo=bsplinesyn,metric=MeanSquares,iter=5,gradStep=0.5
# concatenate transfo: the first warping field contains : warp template -> anat ; warp anat --> MT ; warp correction of the internal structure
sct_concat_transfo -w ../mt/warp_template2mt1_corrected.nii.gz,warp_template2anat_reg2fmri_moco_mean.nii.gz -d fmri_moco_mean.nii.gz -o warp_template2fmri.nii.gz
# warp template, atlas and spinal levels
sct_warp_template -d fmri_moco_mean.nii.gz -w warp_template2fmri.nii.gz -a 0 -s 1
# check results
fslview fmri_moco_mean -b 0,1300 label/spinal_levels/spinal_level_C3.nii.gz -l Red -b 0,0.05 label/spinal_levels/spinal_level_C4.nii.gz -l Blue -b 0,0.05 label/spinal_levels/spinal_level_C5.nii.gz -l Green -b 0,0.05 label/spinal_levels/spinal_level_C6.nii.gz -l Yellow -b 0,0.05 label/spinal_levels/spinal_level_C7.nii.gz -l Pink -b 0,0.05 &
# also see: https://dl.dropboxusercontent.com/u/20592661/spinalcordtoolbox/result_batch_processing_fmri.png

# display ending time:
echo "Ended at: $(date +%x_%r)"
