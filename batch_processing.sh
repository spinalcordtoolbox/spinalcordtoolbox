# !/etc/bash
# example batch to process multi-parametric data of the spinal cord
# specific to errsm_30

# set default FSL output to be nii.gz
export FSLOUTPUTTYPE=NIFTI_GZ


# t2
# ----------
mkdir t2
cd t2
dcm2nii -o . /Volumes/data_shared/montreal_criugm/errsm_30/02-tse_spc_1mm_p2_FOV384__top/errsm_30-0001.dcm
mv *.nii.gz t2.nii.gz
sct_propseg -t t2 -i t2.nii.gz -o . -centerline-binary -mesh
# check results and crop image (identify min and max slices)
fslview t2 t2_seg &
sct_crop_image -i t2.nii.gz -o t2.nii.gz -start 6 -end 200 -dim 1
sct_crop_image -i t2_seg.nii.gz -o t2_seg.nii.gz -start 6 -end 200 -dim 1
# make landmarks at C3 (3) and T4 (11)
# register to template
sct_register_to_template -i t2.nii.gz -l labels.nii.gz -m t2_seg.nii.gz -o 1 -s normal -r 1
# warp template and atlas
sct_warp_template -d t2.nii.gz -w warp_template2anat.nii.gz


#  t1
# ----------
mkdir t1
cd t1
dcm2nii -o . /Volumes/data_shared/montreal_criugm/errsm_30/48-MEMPRAGE_3e_p2_1mm__top_ND\ RMS_S4_DIS3D/echo_2.09/errsm_30-0001.dcm
rm c*.nii.gz
rm o*.nii.gz
mv *.nii.gz t1.nii.gz
# segmentation (for registration to template)
sct_propseg -i t1.nii.gz -t t1
# check results and crop image (identify min and max slices)
fslview t1 t1_seg &
sct_crop_image -i t1.nii.gz -o t1.nii.gz -start 10 -end 210
sct_crop_image -i t1_seg.nii.gz -o t1_seg.nii.gz -start 10 -end 210
# register to template (template registered to t2). N.B. only uses segmentation (more accurate)
sct_register_multimodal -i ../t2/template2anat.nii.gz -d t1.nii.gz -x 1 -v 1 -n 15x3 -y 0 -g 0.2,0.5 -s ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -t t1_seg.nii.gz
# concatenate transfo
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d t1.nii.gz -o warp_template2t1.nii.gz
sct_concat_transfo -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_t12template.nii.gz
# warp template and atlas
sct_warp_template -d t1.nii.gz -w warp_template2t1.nii.gz


# dmri
# ----------
mkdir dmri
cd dmri
dcm2nii -o . /Volumes/data_shared/montreal_criugm/errsm_30/26-ep2d_diff_2drf_0.8mm_24dir_AC_allCoil/errsm_30-0001.dcm
mv *.nii.gz dmri.nii.gz
mv *.bval bvals.txt
mv *.bvec bvecs.txt
# moco
sct_dmri_moco -i dmri.nii.gz -b bvecs.txt -d 3
# create "init-mask.nii.gz" on mean_dwi_moco (will be used for segmentation). Three points in middle of the cord.
fslview dwi_moco_mean &
# segment mean_dwi
sct_propseg -i dwi_moco_mean.nii.gz -t t1 -init-mask init-mask.nii.gz
# register to template (template registered to t2).
sct_register_multimodal -i ../t2/template2anat.nii.gz -d dwi_moco_mean.nii.gz -x 1 -v 1 -n 15x3 -y 5 -g 0.1,0.5 -s ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -t dwi_moco_mean_seg.nii.gz
# concatenate transfo
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d dwi_moco_mean.nii.gz -o warp_template2dmri.nii.gz
sct_concat_transfo -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_dmri2template.nii.gz
# warp template and atlas
sct_warp_template -d dwi_moco_mean.nii.gz -w warp_template2dmri.nii.gz


# mt
# ----------
mkdir mt
cd mt
dcm2nii -o . /Volumes/data_shared/montreal_criugm/errsm_30/19-gre_t1_MTC1/errsm_30-0001.dcm
mv *.nii.gz mt1.nii.gz
# === open fslview and create file init.nii.gz otherwise segmentation will not work ===
sct_segmentation_propagation -i mt1.nii.gz -t t2 -init-mask init.nii.gz
# register to template (template registered to t2). N.B. only uses segmentation (more accurate)
sct_register_multimodal.py -i ../t2/template2anat.nii.gz -d mt1.nii.gz -x 1 -v 1 -n 15x3 -y 5 -g 0.1,0.5 -s ../t2/templateseg2anat.nii.gz -t segmentation_binary.nii.gz
# concatenate transfo
sct_concat_transfo.py -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d mt1.nii.gz -o warp_template2mt.nii.gz
sct_concat_transfo.py -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_mt2template.nii.gz
# warp template+atlas
sct_warp_template.py -d mt1.nii.gz -w warp_template2mt.nii.gz
# warp mt to template
sct_apply_transfo.py -i mt1.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -w warp_mt2template.nii.gz


# fmri
# ----------
mkdir fmri
cd fmri
dcm2nii -o . /Volumes/data_shared/montreal_criugm/errsm_30/46-rp_ep2d_bold_phcorr__test_1x1x3/errsm_30-0001.dcm
mv *.nii.gz fmri.nii.gz
# mean volume
fslmaths fmri.nii.gz -Tmean fmri_mean
# create "fmri_mean-mask.nii.gz" on fmri_mean (will be used for segmentation). Three points in middle of the cord.
fslview fmri_mean &
# segment mean volume
sct_propseg -i fmri_mean.nii.gz -t t2 -init-mask fmri_mean-mask.nii.gz -detect-radius 6 -max-deformation 5
# register to template (template registered to t2). Only uses segmentation (more accurate)
sct_register_multimodal -i ../t2/template2anat.nii.gz -d fmri_mean.nii.gz -x 1 -v 1 -n 15x3 -y 0 -s ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -t fmri_mean_seg.nii.gz
# concatenate transfo
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d fmri_mean.nii.gz -o warp_template2fmri.nii.gz
sct_concat_transfo -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_fmri2template.nii.gz
# warp template and atlas
sct_warp_template -d fmri_mean.nii.gz -w warp_template2fmri.nii.gz
