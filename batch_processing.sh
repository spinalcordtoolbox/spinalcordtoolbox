#!/bin/bash
#
# Example of commands to process multi-parametric data of the spinal cord.
# 
# Please note that this batch script has a lot of redundancy and should not
# be used as a pipeline for regular processing. For example, there is no need
# to process both t1 and t2 to extract CSA values. 
#
# For information about acquisition parameters, see: https://osf.io/wkdym/
# N.B. The parameters are set for these type of data. With your data, parameters 
# might be slightly different.
#
# Usage:
# 
#   [option] $SCT_DIR/batch_processing.sh
# 
#   Prevent (re-)downloading sct_example_data:
#   SCT_BP_DOWNLOAD=0 $SCT_DIR/batch_processing.sh
# 
#   Specify quality control (QC) folder (Default is ~/qc_batch_processing):
#   SCT_BP_QC_FOLDER=/user/toto/my_qc_folder $SCT_DIR/batch_processing.sh

# Abort on error
set -e

# For full verbose, uncomment the next line
# set -x

# Fetch OS type
if uname -a | grep -i  darwin > /dev/null 2>&1; then
  # OSX
  open_command="open"
elif uname -a | grep -i  linux > /dev/null 2>&1; then
  # Linux
  open_command="xdg-open"
fi

# Check if users wants to use his own data
if [ -z "$SCT_BP_DOWNLOAD" ]; then
	SCT_BP_DOWNLOAD=1
fi

# QC folder
if [ -z "$SCT_BP_QC_FOLDER" ]; then
	SCT_BP_QC_FOLDER=~/qc_batch_processing
fi

# Remove QC folder
if [ -z "$SCT_BP_NO_REMOVE_QC" -a -d "$SCT_BP_QC_FOLDER" ]; then
  echo "Removing $SCT_BP_QC_FOLDER folder."
  rm -rf "$SCT_BP_QC_FOLDER"
fi

# display starting time:
echo "Started at: $(date +%x_%r)"

# download example data
if [ "$SCT_BP_DOWNLOAD" == "1" ]; then
  sct_download_data -d sct_example_data
fi
cd sct_example_data


# t2
# ===========================================================================================
cd t2
# Segment spinal cord
sct_propseg -i t2.nii.gz -c t2 -qc "$SCT_BP_QC_FOLDER"
# Tips: If you are not satisfied with the results you can try with another algorithm:
# sct_deepseg_sc -i t2.nii.gz -c t2 -qc "$SCT_BP_QC_FOLDER"
# Vertebral labeling
# Tips: for manual initialization of labeling by clicking at disc C2-C3, use flag -initc2
sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2 -qc "$SCT_BP_QC_FOLDER"
# Create labels at in the cord at C2 and C5 mid-vertebral levels
sct_label_utils -i t2_seg_labeled.nii.gz -vert-body 2,5 -o labels_vert.nii.gz
# Tips: you can also create labels manually using:
# sct_label_utils -i t2.nii.gz -create-viewer 2,5 -o labels_vert.nii.gz
# Register to template
sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -l labels_vert.nii.gz -c t2 -qc "$SCT_BP_QC_FOLDER"
# Tips: If you are not satisfied with the results, you can tweak registration parameters. 
# For example here, we would like to take into account the rotation of the cord, as well as
# adding a 3rd registration step that uses the image intensity (not only cord segmentations).
# so we could do something like this:
# sct_register_multimodal -i $SCT_DIR/data/PAM50/template/PAM50_t2s.nii.gz -iseg $SCT_DIR/data/PAM50/template/PAM50_cord.nii.gz -d t2s.nii.gz -dseg t2s_seg.nii.gz -param step=1,type=seg,algo=slicereg,smooth=3:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3 -initwarp ../t2/warp_template2anat.nii.gz
# Warp template without the white matter atlas (we don't need it at this point)
sct_warp_template -d t2.nii.gz -w warp_template2anat.nii.gz -a 0
# Compute average cross-sectional area and volume between C2 and C3 levels
sct_process_segmentation -i t2_seg.nii.gz -p csa -vert 2:3
# Compute spinal cord shape information at each slice (e.g. AP/RL diameter, eccentricity, etc.)
sct_process_segmentation -i t2_seg.nii.gz -p shape
# Go back to root folder
cd ..


# t2s (stands for t2-star)
# ===========================================================================================
cd t2s
# Spinal cord segmentation
sct_deepseg_sc -i t2s.nii.gz -c t2s -qc "$SCT_BP_QC_FOLDER"
# Segment gray matter
sct_deepseg_gm -i t2s.nii.gz -qc "$SCT_BP_QC_FOLDER"
# Register template->t2s (using warping field generated from template<->t2 registration)
sct_register_multimodal -i $SCT_DIR/data/PAM50/template/PAM50_t2s.nii.gz -iseg $SCT_DIR/data/PAM50/template/PAM50_cord.nii.gz -d t2s.nii.gz -dseg t2s_seg.nii.gz -param step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3:step=3,type=im,algo=syn,slicewise=1,iter=1,metric=CC -initwarp ../t2/warp_template2anat.nii.gz -initwarpinv ../t2/warp_anat2template.nii.gz
# rename warping fields for clarity
mv warp_PAM50_t2s2t2s.nii.gz warp_template2t2s.nii.gz
mv warp_t2s2PAM50_t2s.nii.gz warp_t2s2template.nii.gz
# Warp template
sct_warp_template -d t2s.nii.gz -w warp_template2t2s.nii.gz
# Subtract GM segmentation from cord segmentation to obtain WM segmentation
sct_maths -i t2s_seg.nii.gz -sub t2s_gmseg.nii.gz -o t2s_wmseg.nii.gz
# Compute cross-sectional area (CSA) of the gray and white matter between C2 and C5
sct_process_segmentation -i t2s_wmseg.nii.gz -p csa -vert 2:5 -perlevel 1 -o csa_wm.csv
sct_process_segmentation -i t2s_gmseg.nii.gz -p csa -vert 2:5 -perlevel 1 -o csa_gm.csv
# OPTIONAL: Update template registration using information from gray matter segmentation
# # <<<
# # Register WM/GM template to WM/GM seg
# sct_register_graymatter -gm t2s_gmseg.nii.gz -wm t2s_wmseg.nii.gz -w warp_template2t2s.nii.gz -winv warp_t2s2template.nii.gz
# # Rename warping fields for clarity
# mv warp_template2t2s_reg_gm.nii.gz warp_template2t2s.nii.gz
# mv warp_t2s2template_reg_gm.nii.gz warp_t2s2template.nii.gz
# # Warp template (this time corrected for internal structure)
# sct_warp_template -d t2s.nii.gz -w warp_template2t2s.nii.gz
# # >>>
cd ..


# t1
# ===========================================================================================
cd t1
# Segment spinal cord
sct_propseg -i t1.nii.gz -c t1
# Smooth spinal cord along superior-inferior axis
sct_smooth_spinalcord -i t1.nii.gz -s t1_seg.nii.gz
# Flatten cord in the right-left direction (to make nice figure)
sct_flatten_sagittal -i t1.nii.gz -s t1_seg.nii.gz
# Go back to root folder
cd ..


# mt
# ===========================================================================================
cd mt
# Get centerline from mt1 data
sct_get_centerline -i mt1.nii.gz -c t2
# sct_get_centerline -i mt1.nii.gz -c t2 -qc "$SCT_BP_QC_FOLDER"
# Create mask
sct_create_mask -i mt1.nii.gz -p centerline,mt1_centerline.nii.gz -size 45mm
# Crop data for faster processing
sct_crop_image -i mt1.nii.gz -m mask_mt1.nii.gz -o mt1_crop.nii.gz
# Segment spinal cord
sct_propseg -i mt1_crop.nii.gz -c t2 -qc "$SCT_BP_QC_FOLDER"
# Register mt0->mt1
# Tips: here we only use rigid transformation because both images have very similar sequence parameters. We don't want to use SyN/BSplineSyN to avoid introducing spurious deformations.
# Tips: here we input -dseg because it is needed by the QC report
sct_register_multimodal -i mt0.nii.gz -d mt1_crop.nii.gz -dseg mt1_crop_seg.nii.gz -param step=1,type=im,algo=rigid,slicewise=1,metric=CC -x spline -qc "$SCT_BP_QC_FOLDER"
# Register template->mt1
# Tips: here we only use the segmentations due to poor SC/CSF contrast at the bottom slice.
# Tips: First step: slicereg based on images, with large smoothing to capture potential motion between anat and mt, then at second step: bpslinesyn in order to adapt the shape of the cord to the mt modality (in case there are distortions between anat and mt).
sct_register_multimodal -i $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz -iseg $SCT_DIR/data/PAM50/template/PAM50_cord.nii.gz -d mt1_crop.nii.gz -dseg mt1_crop_seg.nii.gz -param step=1,type=seg,algo=slicereg,smooth=3:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3 -initwarp ../t2/warp_template2anat.nii.gz -initwarpinv ../t2/warp_anat2template.nii.gz
# Rename warping fields for clarity
mv warp_PAM50_t22mt1_crop.nii.gz warp_template2mt.nii.gz
mv warp_mt1_crop2PAM50_t2.nii.gz warp_mt2template.nii.gz
# Warp template
sct_warp_template -d mt1_crop.nii.gz -w warp_template2mt.nii.gz -qc "$SCT_BP_QC_FOLDER"
# Compute mtr
sct_compute_mtr -mt0 mt0_reg.nii.gz -mt1 mt1_crop.nii.gz
# Register t1w->mt1
# Tips: We do not need to crop the t1w image before registration because step=0 of the registration is to put the source image in the space of the destination image (equivalent to cropping the t1w)
sct_register_multimodal -i t1w.nii.gz -d mt1_crop.nii.gz -dseg mt1_crop_seg.nii.gz -param step=1,type=im,algo=rigid,slicewise=1,metric=CC -x spline -qc "$SCT_BP_QC_FOLDER"
# Compute MTsat
# Tips: Check your TR and Flip Angle from the Dicom data
sct_compute_mtsat -mt mt1_crop.nii.gz -pd mt0_reg.nii.gz -t1 t1w_reg.nii.gz -trmt 30 -trpd 30 -trt1 15 -famt 9 -fapd 9 -fat1 15
# Extract MTR, T1 and MTsat within the white matter between C2 and C5.
# Tips: Here we use "-discard-neg-val 1" to discard inconsistent negative values in MTR calculation which are caused by noise.
sct_extract_metric -i mtr.nii.gz -method map -o mtr_in_wm.csv -l 51 -vert 2:5
sct_extract_metric -i mtsat.nii.gz -method map -o mtsat_in_wm.csv -l 51 -vert 2:5
sct_extract_metric -i t1map.nii.gz -method map -o t1_in_wm.csv -l 51 -vert 2:5
# Bring MTR to template space (e.g. for group mapping)
sct_apply_transfo -i mtr.nii.gz -d $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz -w warp_mt2template.nii.gz
# Go back to root folder
cd ..


# dmri
# ===========================================================================================
cd dmri
# bring t2 segmentation in dmri space to create mask (no optimization)
sct_maths -i dmri.nii.gz -mean t -o dmri_mean.nii.gz
sct_register_multimodal -i ../t2/t2_seg.nii.gz -d dmri_mean.nii.gz -identity 1 -x nn
# create mask to help moco and for faster processing
sct_create_mask -i dmri_mean.nii.gz -p centerline,t2_seg_reg.nii.gz -size 35mm
# crop data
sct_crop_image -i dmri.nii.gz -m mask_dmri_mean.nii.gz -o dmri_crop.nii.gz
# motion correction
# Tips: if data have very low SNR you can increase the number of successive images that are averaged into group with "-g". Also see: sct_dmri_moco -h
sct_dmri_moco -i dmri_crop.nii.gz -bvec bvecs.txt
# segmentation with propseg
sct_propseg -i dmri_crop_moco_dwi_mean.nii.gz -c dwi -qc "$SCT_BP_QC_FOLDER"
# Register template to dwi
# Tips: Again, here, we prefer to stick to segmentation-based registration. If there are susceptibility distortions in your EPI, then you might consider adding a third step with bsplinesyn or syn transformation for local adjustment.
sct_register_multimodal -i $SCT_DIR/data/PAM50/template/PAM50_t1.nii.gz -iseg $SCT_DIR/data/PAM50/template/PAM50_cord.nii.gz -d dmri_crop_moco_dwi_mean.nii.gz -dseg dmri_crop_moco_dwi_mean_seg.nii.gz -param step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,smooth=1,iter=3 -initwarp ../t2/warp_template2anat.nii.gz -initwarpinv ../t2/warp_anat2template.nii.gz -qc "$SCT_BP_QC_FOLDER"
# Rename warping fields for clarity
mv warp_PAM50_t12dmri_crop_moco_dwi_mean.nii.gz warp_template2dmri.nii.gz
mv warp_dmri_crop_moco_dwi_mean2PAM50_t1.nii.gz warp_dmri2template.nii.gz
# Warp template and white matter atlas
sct_warp_template -d dmri_crop_moco_dwi_mean.nii.gz -w warp_template2dmri.nii.gz -qc "$SCT_BP_QC_FOLDER"
# Compute DTI metrics
# Tips: The flag -method "restore" allows you to estimate the tensor with robust fit (see: sct_dmri_compute_dti -h)
sct_dmri_compute_dti -i dmri_crop_moco.nii.gz -bval bvals.txt -bvec bvecs.txt
# Compute FA within right and left lateral corticospinal tracts from slices 2 to 14 using weighted average method
sct_extract_metric -i dti_FA.nii.gz -z 2:14 -method wa -l 4,5 -o fa_in_cst.csv
# Bring metric to template space (e.g. for group mapping)
sct_apply_transfo -i dti_FA.nii.gz -d $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz -w warp_dmri2template.nii.gz
# Go back to root folder
cd ..


# fmri
# ===========================================================================================
cd fmri
# Average all fMRI time series (to be able to do the next step)
sct_maths -i fmri.nii.gz -mean t -o fmri_mean.nii.gz
# Get cord centerline
sct_get_centerline -i fmri_mean.nii.gz -c t2s
# Create mask around the cord to help motion correction and for faster processing
sct_create_mask -i fmri_mean.nii.gz -p centerline,fmri_mean_centerline.nii.gz -size 35mm
# Crop data
sct_crop_image -i fmri.nii.gz -m mask_fmri_mean.nii.gz -o fmri_crop.nii.gz
# Motion correction
# Tips: Here data have sufficient SNR and there is visible motion between two consecutive scans, so motion correction is more efficient with -g 1 (i.e. not average consecutive scans)
sct_fmri_moco -i fmri_crop.nii.gz -g 1
# Segment spinal cord manually
# Since these data have very poor cord/CSF contrast, it is difficult to segment the cord properly
# and hence in this case we do it manually. The file is called: fmri_crop_moco_mean_seg_manual.nii.gz
# Register template->fmri
sct_register_multimodal -i $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz -iseg $SCT_DIR/data/PAM50/template/PAM50_cord.nii.gz -d fmri_crop_moco_mean.nii.gz -dseg fmri_crop_moco_mean_seg_manual.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares,smooth=2:step=2,type=im,algo=bsplinesyn,metric=MeanSquares,iter=5,gradStep=0.5 -initwarp ../t2/warp_template2anat.nii.gz -initwarpinv ../t2/warp_anat2template.nii.gz -qc "$SCT_BP_QC_FOLDER"
# Rename warping fields for clarity
mv warp_PAM50_t22fmri_crop_moco_mean.nii.gz warp_template2fmri.nii.gz
mv warp_fmri_crop_moco_mean2PAM50_t2.nii.gz warp_fmri2template.nii.gz
# Warp template and spinal levels (here we don't need the WM atlas)
sct_warp_template -d fmri_crop_moco_mean.nii.gz -w warp_template2fmri.nii.gz -a 0 -s 1
# Note, once you have computed fMRI statistics in the subject's space, you can use
# warp_fmri2template.nii.gz to bring the statistical maps on the template space, for group analysis.
cd ..


# Display results (to easily compare integrity across SCT versions)
# ===========================================================================================
echo "Ended at: $(date +%x_%r)"
echo
echo "t2/CSA:         " `awk -F"," ' {print $6}' t2/csa.csv | tail -1`
echo "mt/MTR(WM):     " `awk -F"," ' {print $8}' mt/mtr_in_wm.csv | tail -1`
echo "t2s/CSA_GM:     " `awk -F"," ' {print $6}' t2s/csa_gm.csv | tail -1`
echo "t2s/CSA_WM:     " `awk -F"," ' {print $6}' t2s/csa_wm.csv | tail -1`
echo "dmri/FA(CST_r): " `awk -F"," ' {print $7}' dmri/fa_in_cst.csv | tail -1`
echo "dmri/FA(CST_l): " `awk -F"," ' {print $7}' dmri/fa_in_cst.csv | tail -2 | head -1`
echo

# Display syntax to open QC report on web browser
echo "To open Quality Control (QC) report on a web-browser, run the following:"
echo "${open_command} ${SCT_BP_QC_FOLDER}/index.html"
