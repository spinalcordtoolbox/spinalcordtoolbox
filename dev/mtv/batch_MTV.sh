#!/bin/bash
# hc_sc_003
# mtv processing

# -----------------------------------------Parameters to be updated-----------------------------------------------------

# Path to the MTV folder
#path_to_folder=/home/django/slevy/data/boston/hc_sc_003/mtv  # No need if the script is run from this folder

# Path to SPGR raw data
path_to_spgr4=/Volumes/data_shared/boston/connectome/HC_SC_003/dicom/8_fl3d_mtv_4/20140724_092151fl3dmtv4s008a1001.nii.gz
path_to_spgr10=/Volumes/data_shared/boston/connectome/HC_SC_003/dicom/7_fl3d_mtv_10/20140724_092151fl3dmtv10s007a1001.nii.gz
path_to_spgr20=/Volumes/data_shared/boston/connectome/HC_SC_003/dicom/6_fl3d_mtv_20/20140724_092151fl3dmtv20s006a1001.nii.gz
path_to_spgr30=/Volumes/data_shared/boston/connectome/HC_SC_003/dicom/5_fl3d_mtv_30/20140724_092151fl3dmtv30s005a1001.nii.gz

# Path to raw EPI for B1 mapping
path_to_ep60=/Volumes/data_shared/boston/connectome/HC_SC_003/dicom/9_ep_seg_se_FA60/20140724_092151epsegseFA60s009a1001.nii.gz
path_to_ep120=/Volumes/data_shared/boston/connectome/HC_SC_003/dicom/10_ep_seg_se_FA120/20140724_092151epsegseFA120s010a1001.nii.gz

# Coordinates to crop data
xmin=50
xsize=80
ymin=60
ysize=80
zmin=4
zsize=13

# ----------------------------------------------------------------------------------------------------------------------

# Go to mtv folder
#cd ${path_to_folder}  # No need if the script is run from this folder

# copy the data from the raw data
# SPGR
cp ${path_to_spgr4} spgr4.nii.gz
cp ${path_to_spgr10} spgr10.nii.gz
cp ${path_to_spgr20} spgr20.nii.gz
cp ${path_to_spgr30} spgr30.nii.gz
# B1 mapping
if ! [ -e "b1" ]; then mkdir b1; fi  # make folder B1 if doesn't exist
cp ${path_to_ep60} b1/ep60.nii.gz
cp ${path_to_ep120} b1/ep120.nii.gz

# List of SPGR images
SPGR_LIST="spgr4 spgr10 spgr20 spgr30"

# Crop SPGR images
for spgr in $SPGR_LIST; do

	fslroi ${spgr} ${spgr}_crop ${xmin} ${xsize} ${ymin} ${ysize} ${zmin} ${zsize}

done

# Compute B1 profile
sct_smooth_b1.py -i b1/ep60.nii.gz,b1/ep120.nii.gz -d spgr10.nii.gz -o b1_smoothed

# Crop B1 profile as were the SPGR data
fslroi b1/b1_smoothed_in_spgr10_space.nii.gz b1/b1_smoothed_in_spgr10_space_crop.nii.gz ${xmin} ${xsize} ${ymin} ${ysize} ${zmin} ${zsize}

# SPGR segmentation
if [ -e "spgr10_crop_seg_modif.nii.gz" ]; then
    fname_spgr10_seg=spgr10_crop_seg_modif.nii.gz
else
    fname_spgr10_seg=spgr10_crop_seg.nii.gz
    sct_propseg -i spgr10_crop.nii.gz -t t1 -radius 6
fi

# Create mask used for SPGR images registration
sct_create_mask -i spgr10_crop.nii.gz -m centerline,${fname_spgr10_seg} -f cylinder -s 40

# Register all SPGR images to the SPGR image with flip angle=10
FA_LIST="20 30"
for fa in $FA_LIST; do
    sct_register_multimodal -i spgr${fa}_crop.nii.gz -d spgr10_crop.nii.gz -m mask_spgr10_crop.nii.gz -z 0 -p 5,sliceReg,1,MeanSquares -o spgr${fa}to10.nii.gz
done

# remove useless outputs
rm spgr10_crop_reg.nii.gz warp_dest2src.nii.gz warp_src2dest.nii.gz
#rm mask_spgr10_crop.nii.gz tmp_InverseWarp.nii.gz tmp_Warp.nii.gz tmp_TxTy_poly.csv

# Compute PD, T1 and MTVF maps
sct_compute_mtvf.py -a 60 -b b1/b1_smoothed_in_spgr10_space_crop.nii.gz -c spgr10_crop_csf_mask.nii.gz -f 4,10,20,30 -i spgr4_crop.nii.gz,spgr10_crop.nii.gz,spgr20to10.nii.gz,spgr30to10.nii.gz -o T1_map,PD_map,MTVF_map -p mean-PD-in-CSF-from-mean-SPGR -s ${fname_spgr10_seg} -t 0.02

# create folder mtv/template if doesn't exist
if ! [ -e "template" ]; then mkdir template; fi

# Estimate warping field from template in anat space to spgr10_crop
sct_register_multimodal -i ../t2/label/template/MNI-Poly-AMU_T2.nii.gz -d spgr10_crop.nii.gz -m mask_spgr10_crop.nii.gz -z 0 -p 5,sliceReg,1,MI
mv warp_src2dest.nii.gz ../warping_fields/warp_template2mtv_step1.nii.gz
mv warp_dest2src.nii.gz ../warping_fields/warp_mtv2template_step2.nii.gz
rm MNI-Poly-AMU_T2_reg.nii.gz spgr10_crop_reg.nii.gz

# Warp white matter to intermediate step to improve the registration afterwards
sct_apply_transfo -i ../t2/label/template/MNI-Poly-AMU_WM.nii.gz -d spgr10_crop.nii.gz -w ../warping_fields/warp_template2mtv_step1.nii.gz -o template/wm2mtv_step1.nii.gz

# Improve the registration
sct_register_multimodal -i template/wm2mtv_step1.nii.gz -d T1_map.nii.gz -m mask_spgr10_crop.nii.gz -z 5 -p 5,BSplineSyN,0.1,MI -o template/wm2mtv_step2.nii.gz
mv template/warp_src2dest.nii.gz ../warping_fields/warp_template2mtv_step2.nii.gz
mv template/warp_dest2src.nii.gz ../warping_fields/warp_mtv2template_step1.nii.gz
rm template/T1_map_reg.nii.gz

# Concatenate warping fields to get:
#   - warp_template2mtv_final = warp_template2anat_final + warp_template2anat_step1 + warp_template2anat_step2
#   - warp_mtv2template_final = warp_mtv2template_step1 + warp_mtv2template_step2 + warp_anat2template_final
sct_concat_transfo -w ../warping_fields/warp_template2anat_final.nii.gz,../warping_fields/warp_template2mtv_step1.nii.gz,../warping_fields/warp_template2mtv_step2.nii.gz -d spgr10_crop.nii.gz -o ../warping_fields/warp_template2mtv_final.nii.gz
sct_concat_transfo -w ../warping_fields/warp_mtv2template_step1.nii.gz,../warping_fields/warp_mtv2template_step2.nii.gz,../warping_fields/warp_anat2template_final.nii.gz -d ${SCT_DIR}/data/template/MNI-Poly-AMU_T2.nii.gz -o ../warping_fields/warp_mtv2template_final.nii.gz

# Warp atlas to MTV space
sct_warp_template -d spgr10_crop.nii.gz -w ../warping_fields/warp_template2mtv_final.nii.gz

# Warp MTVF map to template
sct_apply_transfo -i MTVF_map.nii.gz -o MTVF_map_to_template.nii.gz -d ${SCT_DIR}/data/template/MNI-Poly-AMU_T2.nii.gz -w ../warping_fields/warp_mtv2template_final.nii.gz -p spline