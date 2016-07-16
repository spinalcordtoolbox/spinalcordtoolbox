#!/bin/bash
#
# Example of commands to process multi-parametric data of the spinal cord
# For information about acquisition parameters, see: https://dl.dropboxusercontent.com/u/20592661/publications/Fonov_NIMG14_MNI-Poly-AMU.pdf
# N.B. The parameters are set for these type of data. With your data, parameters might be slightly different.
#
# To run without fslview output, type:
#   ./batch_processing.sh -nodisplay
#
# tested with v3.0_beta14 on 2016-07-16

# Check if display is on or off
if [[ $@ == *"-nodisplay"* ]]; then
  DISPLAY=false
  echo "Display mode turned off."
else
  DISPLAY=true
fi

# get SCT_DIR
source sct_env

# download example data (errsm_30)
sct_download_data -d sct_example_data

# display starting time:
echo "Started at: $(date +%x_%r)"

# go in folder
cd sct_example_data


# t2
# ===========================================================================================
cd t2
# Spinal cord segmentation
sct_propseg -i t2.nii.gz -c t2
# Check results:
if [ $DISPLAY = true ]; then
  fslview t2 -b 0,800 t2_seg -l Red -t 0.5 &
fi
# Vertebral labeling. Here we use the fact that the FOV is centered at C7.
sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2 -initcenter 7
# Create labels at C3 and T2 vertebral levels
sct_label_utils -i t2_seg_labeled.nii.gz -vert-body 3,9
# Register to template
sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -l labels.nii.gz -c t2
# Warp template without the white matter atlas (we don't need it at this point)
sct_warp_template -d t2.nii.gz -w warp_template2anat.nii.gz -a 0
# check results
if [ $DISPLAY = true ]; then
  fslview t2.nii.gz -b 0,800 label/template/PAM50_t2.nii.gz -b 0,4000 label/template/PAM50_levels.nii.gz -l MGH-Cortical -t 0.5 label/template/PAM50_gm.nii.gz -l Red-Yellow -b 0.5,1 label/template/PAM50_wm.nii.gz -l Blue-Lightblue -b 0.5,1 &
fi
# compute average cross-sectional area and volume between C3 and C4 levels
sct_process_segmentation -i t2_seg.nii.gz -p csa -vert 3:4
# go back to root folder
cd ..


# t1
# ===========================================================================================
cd t1
# Spinal cord segmentation
sct_propseg -i t1.nii.gz -c t1
# Check results:
if [ $DISPLAY = true ]; then
  fslview t1 -b 0,800 t2_seg -l Red -t 0.5 &
fi
# Tips: the cord segmentation is "leaking". To obtain better segmentation, you can smooth along centerline and re-run propseg:
sct_smooth_spinalcord -i t1.nii.gz -s t1_seg.nii.gz
sct_propseg -i t1_smooth.nii.gz -c t1
mv t1_smooth_seg.nii.gz t1_seg.nii.gz
# Vertebral labeling. Here we use the fact that the FOV is centered at C7.
sct_label_vertebrae -i t1.nii.gz -s t1_seg.nii.gz -c t1
if [ $DISPLAY = true ]; then
  fslview t1.nii.gz t1_seg_labeled.nii.gz -l Random-Rainbow -t 0.5 &
fi
# Create labels at C3 and T2 vertebral levels
sct_label_utils -i t1_seg_labeled.nii.gz -vert-body 3,9
# Register to template
sct_register_to_template -i t1.nii.gz -s t1_seg.nii.gz -l labels.nii.gz -c t1
# Warp template without the white matter atlas (we don't need it at this point)
sct_warp_template -d t1.nii.gz -w warp_template2anat.nii.gz -a 0
# go back to root folder
cd ..


# mt
# ----------
cd mt
# bring T2 segmentation in MT space to help segmentation (no optimization)
sct_register_multimodal -i ../t2/t2_seg.nii.gz -d mt1.nii.gz -identity 1 -x nn
# segment mt1
sct_propseg -i mt1.nii.gz -c t2 -init-centerline t2_seg_reg.nii.gz
# Check results
if [ $DISPLAY = true ]; then
   fslview mt1 -b 0,800 mt1_seg.nii.gz -l Red -t 0.5 &
fi
# Create close mask around spinal cord (for more accurate registration results)
sct_create_mask -i mt1.nii.gz -p centerline,mt1_seg.nii.gz -size 35 -f cylinder -o mask_mt1.nii.gz
# Register mt0 on mt1
# Tips: here we only use rigid transformation because both images have very similar sequence parameters. We don't want to use SyN/BSplineSyN to avoid introducing spurious deformations.
sct_register_multimodal -i mt0.nii.gz -d mt1.nii.gz -param step=1,type=im,algo=rigid,slicewise=1,metric=CC -m mask_mt1.nii.gz -x spline
# Check results
if [ $DISPLAY = true ]; then
   fslview mt1 -b 0,700 mt0 -b 0,800 mt0_reg -b 0,800 &
fi
# Compute mtr
sct_compute_mtr -mt0 mt0_reg.nii.gz -mt1 mt1.nii.gz
# Register template (in T2 space) to mt1
# Tips: here we only use the segmentations due to poor SC/CSF contrast at the bottom slice.
# Tips: First step: slicereg based on images, with large smoothing to capture potential motion between anat and mt, then at second step: bpslinesyn in order to adapt the shape of the cord to the mt modality (in case there are distortions between anat and mt).
sct_register_multimodal -i ../t2/template2anat.nii.gz -d mt1.nii.gz -iseg ../t2/t2_seg.nii.gz -dseg mt1_seg.nii.gz -param step=1,type=seg,algo=slicereg,smooth=5:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3 -m mask_mt1.nii.gz
# Concat transfo
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_template2anat2mt1.nii.gz -d mtr.nii.gz -o warp_template2mt.nii.gz
# Warp template (to get vertebral labeling)
sct_warp_template -d mt1.nii.gz -w warp_template2mt.nii.gz -a 0
# Segment gray matter
sct_segment_graymatter -i mt1.nii.gz -s mt1_seg.nii.gz
# Register WM/GM template to WM/GM seg
sct_register_graymatter -gm mt1_gmseg.nii.gz -wm mt1_wmseg.nii.gz -w warp_template2mt.nii.gz
# warp template (this time corrected for internal structure)
sct_warp_template -d mt1.nii.gz -w warp_template2mt1_gmseg.nii.gz
# Check registration result
if [ $DISPLAY = true ]; then
   fslview mt1.nii.gz -b 0,800 label/template/PAM50_t2.nii.gz -b 0,4000 label/template/PAM50_levels.nii.gz -l MGH-Cortical -t 0.5 label/template/PAM50_gm.nii.gz -l Red-Yellow -b 0.5,1 label/template/PAM50_wm.nii.gz -l Blue-Lightblue -b 0.5,1 &
fi
# extract MTR within the white matter
sct_extract_metric -i mtr.nii.gz -method map -o mtr_in_wm.txt -l 51
# Once we have register the WM atlas to the subject, we can compute the cross-sectional area (CSA) of specific pathways.
# For example, we can compare the CSA of the left corticospinal tract (CST) to the right CST averaged across the vertebral levels C2 to C5:
sct_process_segmentation -i label/atlas/PAM50_atlas_04.nii.gz -p csa -vert 2:5 -o mt_cst_left_
sct_process_segmentation -i label/atlas/PAM50_atlas_05.nii.gz -p csa -vert 2:5 -o mt_cst_right_
# Get CSA of the dorsal column (fasciculus cuneatus + fasciculus gracilis)
sct_maths -i label/atlas/PAM50_atlas_00.nii.gz -add label/atlas/PAM50_atlas_01.nii.gz,label/atlas/PAM50_atlas_02.nii.gz,label/atlas/PAM50_atlas_03.nii.gz -o dorsal_column.nii.gz
sct_process_segmentation -i dorsal_column.nii.gz -p csa -l 2:5 -o mt_cst_dorsal_
# --> Mean CSA of the left dorsal column: 11.26572434 +/- 0.785786800121 mm^2
cd ..


# dmri
# ----------
cd dmri
# bring T2 segmentation in dmri space to create mask (no optimization)
sct_maths -i dmri.nii.gz -mean t -o dmri_mean.nii.gz
sct_register_multimodal -i ../t2/t2_seg.nii.gz -d dmri_mean.nii.gz -identity 1 -x nn
# create mask to help moco and for faster processing
sct_create_mask -i dmri_mean.nii.gz -p centerline,t2_seg_reg.nii.gz -size 51
# crop data
sct_crop_image -i dmri.nii.gz -m mask_dmri_mean.nii.gz -o dmri_crop.nii.gz
# motion correction
sct_dmri_moco -i dmri_crop.nii.gz -bvec bvecs.txt -x spline
# segmentation with propseg
sct_propseg -i dwi_moco_mean.nii.gz -c t1 -init-centerline t2_seg_reg.nii.gz
# check segmentation
if [ $DISPLAY = true ]; then
  fslview dwi_moco_mean -b 0,300 dwi_moco_mean_seg -l Red -t 0.5 &
fi
# Register template to dwi
# Tips: We use the template registered to the MT data in order to account for gray matter segmentation
# Tips: again, here, we prefer no stick to rigid registration on segmentation following by slicereg to realign center of mass. If there are susceptibility distortions in your EPI, then you might consider adding a third step with bsplinesyn or syn transformation for local adjustment.
sct_register_multimodal -i ../mt/label/template/PAM50_t2.nii.gz -d dwi_moco_mean.nii.gz -iseg ../mt/label/template/PAM50_cord.nii.gz -dseg dwi_moco_mean_seg.nii.gz -param step=1,type=seg,algo=slicereg,smooth=5:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,smooth=1,iter=10
# Concatenate transfo: (1) template -> anat -> MT -> MT_gmreg ; (2) MT_gmreg -> DWI
sct_concat_transfo -w ../mt/warp_template2mt1_gmseg.nii.gz,warp_PAM50_t22dwi_moco_mean.nii.gz -d dwi_moco_mean.nii.gz -o warp_template2dmri.nii.gz
# Warp template and white matter atlas
sct_warp_template -d dwi_moco_mean.nii.gz -w warp_template2dmri.nii.gz
# Visualize white matter template and lateral CST on DWI
if [ $DISPLAY = true ]; then
  fslview dwi_moco_mean -b 0,300 label/template/PAM50_wm.nii.gz -l Blue-Lightblue -b 0.2,1 -t 0.5 label/atlas/PAM50_atlas_04.nii.gz -b 0.2,1 -l Red label/atlas/PAM50_atlas_05.nii.gz -b 0.2,1 -l Yellow &
fi
# Compute DTI metrics
# Tips: the flag -method "restore" allows you to estimate the tensor with robust fit (see help)
sct_dmri_compute_dti -i dmri_crop_moco.nii.gz -bval bvals.txt -bvec bvecs.txt
# Compute FA within right and left lateral corticospinal tracts from slices 1 to 3 using maximum a posteriori
sct_extract_metric -i dti_FA.nii.gz -z 1:3 -method map -l 4,5 -o fa_in_cst.txt
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
# segment mean fMRI volume
# tips: we use the T2 segmentation to help with fMRI segmentation
# tips: we use "-radius 5" otherwise the segmentation is too small
# tips: we use "-max-deformation 4" to prevent the propagation from stopping at the edge
sct_propseg -i fmri_moco_mean.nii.gz -c t2 -init-centerline t2_seg_reg.nii.gz -radius 5 -max-deformation 4
# check segmentation
if [ $DISPLAY = true ]; then
  fslview fmri_moco_mean -b 0,1000 fmri_moco_mean_seg -l Red -t 0.5 &
fi
# here segmentation slightly failed due to the close proximity of susceptibility artifact --> use file "fmri_moco_mean_seg_modif.nii.gz"
# register template to fmri: here we use the template register to the MT to get the correction of the internal structure
sct_register_multimodal -i ../mt/label/template/PAM50_t2.nii.gz -d fmri_moco_mean.nii.gz -iseg ../mt/label/template/PAM50_cord.nii.gz -dseg fmri_moco_mean_seg_modif.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares,smooth=2:step=2,type=im,algo=bsplinesyn,metric=MeanSquares,iter=5,gradStep=0.5
# concatenate transfo: (1) template -> anat -> MT -> MT_gm ; (2) MT_gm -> fmri
sct_concat_transfo -w ../mt/warp_template2mt1_gmseg.nii.gz,warp_PAM50_t22fmri_moco_mean.nii.gz -d fmri_moco_mean.nii.gz -o warp_template2fmri.nii.gz
# warp template and spinal levels (here we don't need the WM atlas)
# N.B. SPINAL LEVEL CURRENTLY NOT AVAILABLE FOR PAM50. WORK IN PROGRESS.
# sct_warp_template -d fmri_moco_mean.nii.gz -w warp_template2fmri.nii.gz -a 0 -s 1
# check results
#if [ $DISPLAY = true ]; then
#  fslview fmri_moco_mean -b 0,1300 label/spinal_levels/spinal_level_C3.nii.gz -l Red -b 0,0.05 label/spinal_levels/spinal_level_C4.nii.gz -l Blue -b 0,0.05 label/spinal_levels/spinal_level_C5.nii.gz -l Green -b 0,0.05 label/spinal_levels/spinal_level_C6.nii.gz -l Yellow -b 0,0.05 label/spinal_levels/spinal_level_C7.nii.gz -l Pink -b 0,0.05 &
#fi
cd ..


# display results (to easily compare integrity across SCT versions)
# ----------
echo "Ended at: $(date +%x_%r)"
echo
echo "t2/CSA:  " `grep -v '^#' t2/csa_mean.txt | grep -v '^$'`
echo "mt/MTR:  " `grep -v '^#' mt/mtr_in_wm.txt | grep -v '^$'`
echo "mt/CSA:  " `grep -v '^#' mt/mt_cst_dorsal_csa_mean.txt | grep -v '^$'`
echo "dmri/FA: " `grep -v '^#' dmri/fa_in_cst.txt | grep -v 'right'`
echo "dmri/FA: " `grep -v '^#' dmri/fa_in_cst.txt | grep -v 'left'`
echo
# results from version dev-ea5287897849623dbda1c25b069d33806a1338c3
#t2/CSA:   /Users/julien/sct_example_data/t2/t2_seg, 76.656727, 2.366052
#mt/MTR:   51, white matter, 397.071102, 33.834860, 0.000000
#mt/CSA:   /Users/julien/sct_example_data/mt/dorsal_column, 17.574594, 2.156033
#dmri/FA:  4, WM left lateral corticospinal tract, 25.650313, 0.709122, 0.000000
#dmri/FA:  5, WM right lateral corticospinal tract, 25.646853, 0.717024, 0.000000
#fMRI results: https://dl.dropboxusercontent.com/u/20592661/sct/result_batch_processing_fmri.png
