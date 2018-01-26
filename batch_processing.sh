#!/bin/bash
#
# Example of commands to process multi-parametric data of the spinal cord
# For information about acquisition parameters, see: https://osf.io/wkdym/
# N.B. The parameters are set for these type of data. With your data, parameters might be slightly different.
#
# To prevent downloading sct_example_data and run from local folder, run:
#   ./batch_processing.sh -nodownload

# Abort on error
set -e

# Check if users wants to use his own data
if [[ $@ == *"-nodownload"* ]]; then
  echo "Use local folder."
else
  # download example data
  sct_download_data -d sct_example_data
  # go in folder
fi
cd sct_example_data

# Remove QC folder
if [ -d ~/qc_batch_processing ]; then
  echo "Removing ~/qc_batch_processing folder folder."
  rm -rf ~/qc_batch_processing
fi

# display starting time:
echo "Started at: $(date +%x_%r)"

# t2
# ===========================================================================================
cd t2
# Spinal cord segmentation
sct_propseg -i t2.nii.gz -c t2 -qc ~/qc_batch_processing
# Vertebral labeling
# tips: for manual initialization of labeling by clicking at disc C2-C3, use flag -initc2
sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2 -v 2 -qc ~/qc_batch_processing
# Create labels at C2 and C5 vertebral levels
sct_label_utils -i t2_seg_labeled.nii.gz -vert-body 2,5
# Register to template
sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -l labels.nii.gz -c t2 -qc ~/qc_batch_processing
# Warp template without the white matter atlas (we don't need it at this point)
sct_warp_template -d t2.nii.gz -w warp_template2anat.nii.gz -a 0
# Compute average cross-sectional area and volume between C2 and C3 levels
sct_process_segmentation -i t2_seg.nii.gz -p csa -vert 2:3
# Compute spinal cord shape information at each slice (e.g. AP/RL diameter, eccentricity, etc.)
sct_process_segmentation -i t2_seg.nii.gz -p shape
# Go back to root folder
cd -


# mt
# ===========================================================================================
cd mt
# bring T2 segmentation in MT space to help segmentation (no optimization)
sct_register_multimodal -i ../t2/t2_seg.nii.gz -d mt1.nii.gz -identity 1 -x nn
# create mask for faster processing
sct_create_mask -i mt1.nii.gz -p centerline,t2_seg_reg.nii.gz -size 45mm
# crop data
sct_crop_image -i mt1.nii.gz -m mask_mt1.nii.gz -o mt1_crop.nii.gz
sct_crop_image -i mt0.nii.gz -m mask_mt1.nii.gz -o mt0_crop.nii.gz
# segment mt1
sct_propseg -i mt1_crop.nii.gz -c t2 -init-centerline t2_seg_reg.nii.gz -qc ~/qc_batch_processing
# Create close mask around spinal cord (for more accurate registration results)
sct_create_mask -i mt1_crop.nii.gz -p centerline,mt1_crop_seg.nii.gz -size 35mm -f cylinder
# Register mt0 on mt1
# Tips: here we only use rigid transformation because both images have very similar sequence parameters. We don't want to use SyN/BSplineSyN to avoid introducing spurious deformations.
sct_register_multimodal -i mt0_crop.nii.gz -d mt1_crop.nii.gz -param step=1,type=im,algo=rigid,slicewise=1,metric=CC -m mask_mt1_crop.nii.gz -x spline
# Register template to mt1
# Tips: here we only use the segmentations due to poor SC/CSF contrast at the bottom slice.
# Tips: First step: slicereg based on images, with large smoothing to capture potential motion between anat and mt, then at second step: bpslinesyn in order to adapt the shape of the cord to the mt modality (in case there are distortions between anat and mt).
sct_register_multimodal -i $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz -d mt1_crop.nii.gz -iseg $SCT_DIR/data/PAM50/template/PAM50_cord.nii.gz -dseg mt1_crop_seg.nii.gz -param step=1,type=seg,algo=slicereg,smooth=3:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3 -m mask_mt1_crop.nii.gz -initwarp ../t2/warp_template2anat.nii.gz -initwarpinv ../t2/warp_anat2template.nii.gz
# rename warping fields for clarity
mv warp_PAM50_t22mt1_crop.nii.gz warp_template2mt.nii.gz
mv warp_mt1_crop2PAM50_t2.nii.gz warp_mt2template.nii.gz
# Warp template
sct_warp_template -d mt1_crop.nii.gz -w warp_template2mt.nii.gz

# OPTIONAL: SEGMENT GRAY MATTER AND USE IT TO IMPROVE TEMPLATE REGISTRATION
# <<<
# Segment gray matter
sct_segment_graymatter -i mt0_crop_reg.nii.gz -s mt1_crop_seg.nii.gz
# Register WM/GM template to WM/GM seg
sct_register_graymatter -gm mt0_crop_reg_gmseg.nii.gz -wm mt0_crop_reg_wmseg.nii.gz -w warp_template2mt.nii.gz -winv warp_mt2template.nii.gz
# rename warping fields for clarity
mv warp_template2mt_reg_gm.nii.gz warp_template2mt.nii.gz
mv warp_mt2template_reg_gm.nii.gz warp_mt2template.nii.gz
# warp template (this time corrected for internal structure)
sct_warp_template -d mt1_crop.nii.gz -w warp_template2mt.nii.gz
# >>>

# Compute mtr
sct_compute_mtr -mt0 mt0_crop_reg.nii.gz -mt1 mt1_crop.nii.gz
# Extract MTR within the white matter between C2 and C5
sct_extract_metric -i mtr.nii.gz -method map -o mtr_in_wm.txt -l 51 -vert 2:5
# Once we have register the WM atlas to the subject, we can compute the cross-sectional area (CSA) of the gray and white matter
sct_process_segmentation -i label/template/PAM50_wm.nii.gz -p csa -vert 2:5 -ofolder csa_wm
sct_process_segmentation -i label/template/PAM50_gm.nii.gz -p csa -vert 2:5 -ofolder csa_gm
# Bring metric to template space (e.g. for group mapping)
sct_apply_transfo -i mtr.nii.gz -d $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz -w warp_mt2template.nii.gz
# Go back to root folder
cd -


# dmri
# ===========================================================================================
cd dmri
# bring T2 segmentation in dmri space to create mask (no optimization)
sct_maths -i dmri.nii.gz -mean t -o dmri_mean.nii.gz
sct_register_multimodal -i ../t2/t2_seg.nii.gz -d dmri_mean.nii.gz -identity 1 -x nn
# create mask to help moco and for faster processing
sct_create_mask -i dmri_mean.nii.gz -p centerline,t2_seg_reg.nii.gz -size 35mm
# crop data
sct_crop_image -i dmri.nii.gz -m mask_dmri_mean.nii.gz -o dmri_crop.nii.gz
# motion correction
sct_dmri_moco -i dmri_crop.nii.gz -bvec bvecs.txt
# segmentation with propseg
sct_propseg -i dwi_moco_mean.nii.gz -c dwi -init-centerline t2_seg_reg.nii.gz -qc ~/qc_batch_processing
# Register template to dwi
# Tips: We use the template registered to the MT data in order to account for gray matter segmentation
# Tips: again, here, we prefer no stick to rigid registration on segmentation following by slicereg to realign center of mass. If there are susceptibility distortions in your EPI, then you might consider adding a third step with bsplinesyn or syn transformation for local adjustment.
sct_register_multimodal -i $SCT_DIR/data/PAM50/template/PAM50_t1.nii.gz -d dwi_moco_mean.nii.gz -iseg $SCT_DIR/data/PAM50/template/PAM50_cord.nii.gz -dseg dwi_moco_mean_seg.nii.gz -param step=1,type=seg,algo=slicereg,smooth=5:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,smooth=1,iter=3 -initwarp ../mt/warp_template2mt.nii.gz -initwarpinv ../mt/warp_mt2template.nii.gz
# rename warping field for clarity
mv warp_PAM50_t12dwi_moco_mean.nii.gz warp_template2dmri.nii.gz
mv warp_dwi_moco_mean2PAM50_t1.nii.gz warp_dmri2template.nii.gz
# Warp template and white matter atlas
sct_warp_template -d dwi_moco_mean.nii.gz -w warp_template2dmri.nii.gz
# Compute DTI metrics
# Tips: the flag -method "restore" allows you to estimate the tensor with robust fit (see help)
sct_dmri_compute_dti -i dmri_crop_moco.nii.gz -bval bvals.txt -bvec bvecs.txt
# Compute FA within right and left lateral corticospinal tracts from slices 2 to 14 using weighted average method
sct_extract_metric -i dti_FA.nii.gz -z 2:14 -method wa -l 4,5 -o fa_in_cst.txt
# Bring metric to template space (e.g. for group mapping)
sct_apply_transfo -i dti_FA.nii.gz -d $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz -w warp_dmri2template.nii.gz
# Go back to root folder
cd -


# Display results (to easily compare integrity across SCT versions)
# ===========================================================================================
echo "Ended at: $(date +%x_%r)"
echo
echo "t2/CSA:  " `grep -v '^#' t2/csa_mean.txt | grep -v '^$'`
echo "mt/MTR:  " `grep -v '^#' mt/mtr_in_wm.txt | grep -v '^$'`
echo "mt/CSA_GM:  " `grep -v '^#' mt/csa_gm/csa_mean.txt | grep -v '^$'`
echo "mt/CSA_WM:  " `grep -v '^#' mt/csa_wm/csa_mean.txt | grep -v '^$'`
echo "dmri/FA: " `grep -v '^#' dmri/fa_in_cst.txt | grep -v 'right'`
echo "dmri/FA: " `grep -v '^#' dmri/fa_in_cst.txt | grep -v 'left'`
echo

# Generate QC report
sct_qc -folder ~/qc_batch_processing
