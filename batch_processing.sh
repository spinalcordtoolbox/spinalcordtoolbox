#!/bin/bash
#
# Example of commands to process multi-parametric data of the spinal cord
# For information about acquisition parameters, see: https://dl.dropboxusercontent.com/u/20592661/publications/Fonov_NIMG14_MNI-Poly-AMU.pdf
# N.B. The parameters are set for these type of data. With your data, parameters might be slightly different.
#
# To run without fslview output, type:
#   ./batch_processing.sh -nodisplay
#
# tested with v2.2.3

# Check if display is on or off
if [[ $@ == *"-nodisplay"* ]]; then
   DISPLAY=false
   echo "Display mode turned off."
else
   DISPLAY=true
fi

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
sct_propseg -i t2.nii.gz -c t2
# check your results:
if [ $DISPLAY = true ]; then
   fslview t2 -b 0,800 t2_seg -l Red -t 0.5 &
fi
# vertebral labeling. Here we use the fact that the FOV is centered at C7.
sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -initcenter 7
# create labels at C2 and T2 vertebral levels
sct_label_utils -i t2_seg_labeled.nii.gz -p label-vertebrae -vert 2,9
# register to template
# tips: here we used only iter=1 for the third step for faster processing. 
sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -l labels.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=seg,algo=bsplinesyn,iter=3,shrink=1,metric=MeanSquares:step=3,type=im,algo=syn,metric=CC,iter=1
# warp template and white matter atlas
sct_warp_template -d t2.nii.gz -w warp_template2anat.nii.gz
# check results
if [ $DISPLAY = true ]; then
   fslview t2.nii.gz -b 0,800 label/template/MNI-Poly-AMU_T2.nii.gz -b 0,4000 label/template/MNI-Poly-AMU_level.nii.gz -l MGH-Cortical -t 0.5 label/template/MNI-Poly-AMU_GM.nii.gz -l Red-Yellow -b 0.5,1 label/template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.5,1 &
fi
# compute average cross-sectional area and volume between C3 and C4 levels
sct_process_segmentation -i t2_seg.nii.gz -p csa -t label/template/MNI-Poly-AMU_level.nii.gz -vert 3:4
# --> Mean CSA: 77.4289770106 +/- 2.00647224442 mm^2
# --> Volume (in volume.txt): 2402.0 mm^3

# go back to root folder
cd ..


#  t1
# ----------
cd t1
# spinal cord segmentation
sct_propseg -i t1.nii.gz -c t1
# check results
if [ $DISPLAY = true ]; then
   fslview t1 -b 0,800 t1_seg -l Red -t 0.5 &
fi
# create mask around spinal cord (for cropping)
sct_create_mask -i t1.nii.gz -p centerline,t1_seg.nii.gz -size 61 -f box -o mask_t1.nii.gz
# crop t1 and t1_seg (for faster registration)
sct_crop_image -i t1.nii.gz -m mask_t1.nii.gz -o t1_crop.nii.gz
sct_crop_image -i t1_seg.nii.gz -m mask_t1.nii.gz -o t1_seg_crop.nii.gz
# register to template (which was previously registered to the t2).
sct_register_multimodal -i ../t2/label/template/MNI-Poly-AMU_T2.nii.gz -iseg ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -d t1_crop.nii.gz -dseg t1_seg_crop.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=im,algo=syn,iter=3,gradStep=0.2,metric=CC
# concatenate transformations
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_MNI-Poly-AMU_T22t1_crop.nii.gz -d t1.nii.gz -o warp_template2t1.nii.gz
sct_concat_transfo -w warp_t1_crop2MNI-Poly-AMU_T2.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_t12template.nii.gz
# warp template
sct_warp_template -d t1.nii.gz -w warp_template2t1.nii.gz -a 0
# check results
if [ $DISPLAY = true ]; then
   fslview t1.nii.gz label/template/MNI-Poly-AMU_T2.nii.gz -b 0,4000 label/template/MNI-Poly-AMU_level.nii.gz -l MGH-Cortical -t 0.5 label/template/MNI-Poly-AMU_GM.nii.gz -l Red-Yellow -b 0.5,1 label/template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.5,1 &
fi
# warp T1 to template space
sct_apply_transfo -i t1.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -w warp_t12template.nii.gz
# check registration of T1 to template
if [ $DISPLAY = true ]; then
   fslview $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -b 0,4000 t1_reg.nii.gz -b 0,800 &
fi
cd ..


# mt
# ----------
cd mt
# bring T2 segmentation in MT space to help segmentation (no optimization)
sct_register_multimodal -i ../t2/t2_seg.nii.gz -d mt1.nii.gz -identity 1 -x nn
# needs to add short pause to make sure the output from previous line is generated (in case this batch script is launched at once)
# extract centerline
sct_process_segmentation -i t2_seg_reg.nii.gz -p centerline
# segment mt1
sct_propseg -i mt1.nii.gz -c t2 -init-centerline t2_seg_reg_centerline.nii.gz
# check results
if [ $DISPLAY = true ]; then
   fslview mt1 -b 0,800 mt1_seg.nii.gz -l Red -t 0.5 &
fi
# create mask around spinal cord for faster registration
sct_create_mask -i mt1.nii.gz -p centerline,mt1_seg.nii.gz -size 51 -f box -o mask_mt1.nii.gz
# crop data
sct_crop_image -i mt1.nii.gz -m mask_mt1.nii.gz -o mt1_crop.nii.gz
sct_crop_image -i mt1_seg.nii.gz -m mask_mt1.nii.gz -o mt1_seg_crop.nii.gz
# register mt0 on mt1
sct_register_multimodal -i mt0.nii.gz -d mt1_crop.nii.gz -param step=1,type=im,algo=slicereg,metric=MI:step=2,type=im,algo=bsplinesyn,metric=CC,iter=3,gradStep=0.2
# compute mtr
sct_compute_mtr -mt0 mt0_reg.nii.gz -mt1 mt1_crop.nii.gz
# register template (in T2 space) to mt1
# tips: here we only rely on the segmentation (not the image), because the close proximity of the cord with the spine can induce inaccuracies in the registration on some slices.
sct_register_multimodal -i ../t2/template2anat.nii.gz -d mt1_crop.nii.gz -iseg ../t2/t2_seg.nii.gz -dseg mt1_seg_crop.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3 -z 3
# concat transfo
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_template2anat2mt1_crop.nii.gz -d mtr.nii.gz -o warp_template2mt.nii.gz
# warp template (to get vertebral labeling)
sct_warp_template -d mtr.nii.gz -w warp_template2mt.nii.gz -a 0
# OPTIONAL PART: SEGMENT GRAY MATTER:
# <<<<<<<<<<
# add mt1 and mt0 to increase GM/WM contrast
sct_maths -i mt0_reg.nii.gz -add mt1_crop.nii.gz -o mt0mt1.nii.gz
# segment GM
sct_segment_graymatter -i mt0mt1.nii.gz -s mt1_seg_crop.nii.gz -vert label/template/MNI-Poly-AMU_level.nii.gz
# register WM template to WMseg
sct_register_multimodal -i label/template/MNI-Poly-AMU_WM.nii.gz -d mt0mt1_wmseg.nii.gz -param step=1,algo=slicereg,metric=MeanSquares:step=2,algo=bsplinesyn,metric=MeanSquares,iter=3
# concat transfo
sct_concat_transfo -w warp_template2mt.nii.gz,warp_MNI-Poly-AMU_WM2mt0mt1_wmseg.nii.gz -d mt0mt1.nii.gz -o warp_template2mt_corrected.nii.gz
# warp template (final warp template for mt1)
sct_warp_template -d mt0mt1.nii.gz -w warp_template2mt_corrected.nii.gz
# check registration result
if [ $DISPLAY = true ]; then
   fslview mtr.nii.gz -b 0,100 mt0mt1.nii.gz -b 0,1200 label/template/MNI-Poly-AMU_T2.nii.gz -b 0,4000 label/template/MNI-Poly-AMU_level.nii.gz -l MGH-Cortical -t 0.5 label/template/MNI-Poly-AMU_GM.nii.gz -l Red-Yellow -b 0.3,1 label/template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.3,1 &
fi
# >>>>>>>>>>
# extract MTR within the white matter
sct_extract_metric -i mtr.nii.gz -f label/atlas/ -l wm -m map
# --> MTR = 34.7067535734
# Once we have register the WM atlas to the subject, we can compute the cross-sectional area (CSA) of specific pathways.
# For example, we can compare the CSA of the left corticospinal tract (CST) to the right CST averaged across the vertebral levels C2 to C5:
sct_process_segmentation -i label/atlas/WMtract__02.nii.gz -p csa -vert 2:5 -t label/template/MNI-Poly-AMU_level.nii.gz
# --> Mean CSA of left CST: 4.97641126008 +/- 0.512628334474 mm^2
sct_process_segmentation -i label/atlas/WMtract__17.nii.gz -p csa -vert 2:5 -t label/template/MNI-Poly-AMU_level.nii.gz
# --> Mean CSA of right CST: 4.77218674544 +/- 0.472737313312 mm^2
# Get CSA of the left dorsal column (fasciculus cuneatus + fasciculus gracilis)
sct_maths -i label/atlas/WMtract__00.nii.gz -add label/atlas/WMtract__01.nii.gz -o left_dorsal_column.nii.gz
sct_process_segmentation -i left_dorsal_column.nii.gz -p csa -l 2:5 -t label/template/MNI-Poly-AMU_level.nii.gz
# --> Mean CSA of the left dorsal column: 9.44132531044 +/- 0.462686426095 mm^2
cd ..


# dmri
# ----------
cd dmri
# create mask to help moco
sct_create_mask -i dmri.nii.gz -p coord,110x20 -size 60 -f cylinder
# motion correction
sct_dmri_moco -i dmri.nii.gz -bvec bvecs.txt -g 3 -m mask_dmri.nii.gz -param 2,2,1,MeanSquares -thr 0
# detect approximate spinal cord centerline
sct_get_centerline -p auto -i dwi_moco_mean.nii.gz -c t1
# fine segmentation with propseg
sct_propseg -i dwi_moco_mean.nii.gz -c t1 -init-centerline dwi_moco_mean_centerline.nii.gz
# check segmentation
if [ $DISPLAY = true ]; then
   fslview dwi_moco_mean -b 0,300 dwi_moco_mean_seg -l Red -t 0.5 & 
fi
# register template to dwi: here we use the template registered to the MT data in order to account for gray matter segmentation
sct_register_multimodal -i ../mt/template2anat_reg.nii.gz -d dwi_moco_mean.nii.gz -iseg ../mt/mt1_seg.nii.gz -dseg dwi_moco_mean_seg.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares,smooth=2:step=2,type=im,algo=bsplinesyn,metric=MeanSquares,iter=5,gradStep=0.5
# concatenate transfo: the first warping field contains : warp template -> anat ; warp anat --> MT ; warp correction of the internal structure 
sct_concat_transfo -w ../mt/warp_template2mt_corrected.nii.gz,warp_template2anat_reg2dwi_moco_mean.nii.gz -d dwi_moco_mean.nii.gz -o warp_template2dmri.nii.gz
# warp template and white matter atlas
sct_warp_template -d dwi_moco_mean.nii.gz -w warp_template2dmri.nii.gz
# visualize white matter template and lateral CST on DWI
if [ $DISPLAY = true ]; then
   fslview dwi_moco_mean -b 0,300 label/template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.2,1 -t 0.5 label/atlas/WMtract__02.nii.gz -b 0.2,1 -l Red label/atlas/WMtract__17.nii.gz -b 0.2,1 -l Yellow &
fi
# compute DTI metrics
sct_dmri_compute_dti -i dmri_moco.nii.gz -bval bvals.txt -bvec bvecs.txt
# compute FA within right and left lateral corticospinal tracts from slices 1 to 3 using maximum a posteriori
sct_extract_metric -i dti_FA.nii.gz -f label/atlas/ -l 2,17 -z 1:3 -method map
# --> 17, right lateral corticospinal tract:    0.787807890652
# --> 2, left lateral corticospinal tract:    0.76589414129
cd ..


# fmri
# ----------
cd fmri
# create mask at the center of the FOV (will be used for moco)
sct_create_mask -i fmri.nii.gz -p center -size 30 -f cylinder
# moco
sct_fmri_moco -i fmri.nii.gz -m mask_fmri.nii.gz
# tips: if you have low SNR you can group consecutive images with "-g"
# put T2 segmentation into fmri space
sct_register_multimodal -i ../t2/t2_seg.nii.gz -d fmri_moco_mean.nii.gz -identity 1 -x nn
# extract centerline
sct_process_segmentation -i t2_seg_reg.nii.gz -p centerline
# segment mean fMRI volume
sct_propseg -i fmri_moco_mean.nii.gz -c t2 -init-centerline t2_seg_reg_centerline.nii.gz -radius 5 -max-deformation 4
# tips: we use the T2 segmentation to help with fMRI segmentation
# tips: we use "-radius 5" otherwise the segmentation is too small
# tips: we use "-max-deformation 4" to prevent the propagation from stopping at the edge
# check segmentation
if [ $DISPLAY = true ]; then
   fslview fmri_moco_mean -b 0,1000 fmri_moco_mean_seg -l Red -t 0.5 &
fi
# here segmentation slightly failed due to the close proximity of susceptibility artifact --> use file "fmri_moco_mean_seg_modif.nii.gz"
# register template to fmri: here we use the template register to the MT to get the correction of the internal structure
sct_register_multimodal -i ../mt/template2anat_reg.nii.gz -d fmri_moco_mean.nii.gz -iseg ../mt/mt1_seg.nii.gz -dseg fmri_moco_mean_seg_modif.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares,smooth=2:step=2,type=im,algo=bsplinesyn,metric=MeanSquares,iter=5,gradStep=0.5
# concatenate transfo: the first warping field contains : warp template -> anat ; warp anat --> MT ; warp correction of the internal structure
sct_concat_transfo -w ../mt/warp_template2mt_corrected.nii.gz,warp_template2anat_reg2fmri_moco_mean.nii.gz -d fmri_moco_mean.nii.gz -o warp_template2fmri.nii.gz
# warp template, atlas and spinal levels
sct_warp_template -d fmri_moco_mean.nii.gz -w warp_template2fmri.nii.gz -a 0 -s 1
# check results
if [ $DISPLAY = true ]; then
   fslview fmri_moco_mean -b 0,1300 label/spinal_levels/spinal_level_C3.nii.gz -l Red -b 0,0.05 label/spinal_levels/spinal_level_C4.nii.gz -l Blue -b 0,0.05 label/spinal_levels/spinal_level_C5.nii.gz -l Green -b 0,0.05 label/spinal_levels/spinal_level_C6.nii.gz -l Yellow -b 0,0.05 label/spinal_levels/spinal_level_C7.nii.gz -l Pink -b 0,0.05 &
fi
# also see: https://dl.dropboxusercontent.com/u/20592661/spinalcordtoolbox/result_batch_processing_fmri.png

# display ending time:
echo "Ended at: $(date +%x_%r)"
