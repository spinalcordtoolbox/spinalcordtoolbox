#!/etc/bash
# example batch to process multi-parametric data of the spinal cord

# t2
# ----------
mkdir t2
cd t2
dcm2nii -o . /Volumes/data_shared/montreal_criugm/errsm_30/02-tse_spc_1mm_p2_FOV384__top/errsm_30-0001.dcm
mv *.nii.gz t2.nii.gz
sct_segmentation_propagation -t t2 -i t2.nii.gz -o . -centerline-binary -mesh
# check results and crop image (identify min and max slices)
fslview t2 segmentation_binary.nii.gz &
sct_crop_image -i t2.nii.gz -o t2.nii.gz -start 6 -end 224 -dim 1
sct_crop_image -i segmentation_binary.nii.gz -o segmentation_binary.nii.gz -start 6 -end 224 -dim 1
# make landmarks at C3 (3) and T4 (11)
# register to template
sct_register_to_template.py -i t2.nii.gz -l labels.nii.gz -m segmentation_binary.nii.gz -o 1 -s normal -r 1
# warp cord segmentation from template
WarpImageMultiTransform 3 $SCT_DIR/data/template/MNI-Poly-AMU_cord.nii.gz templateseg2anat.nii.gz -R t2.nii.gz warp_template2anat.nii.gz
# warp template+atlas
sct_warp_template.py -d t2.nii.gz -w warp_template2anat.nii.gz


#  t1
# ----------
mkdir t1
cd t1
dcm2nii -o . /Volumes/data_shared/montreal_criugm/errsm_30/48-MEMPRAGE_3e_p2_1mm__top_ND\ RMS_S4_DIS3D/echo_2.09/errsm_30-0001.dcm
rm c*.nii.gz
rm o*.nii.gz
mv *.nii.gz t1.nii.gz
# segmentation (for registration to template)
sct_segmentation_propagation -t t1 -i t1.nii.gz -o .
# crop T1
# --- use fslview to identify start and end points ---
sct_crop_image -i t1.nii.gz -o t1_crop.nii.gz -start 10 -end 210
# move segmentation into cropped space
c3d t1_crop.nii.gz segmentation_binary.nii.gz -reslice-identity -o segmentation_binary_crop.nii.gz
# adjust segmentation
# --- manually edit using fslview --- took 2 minutes
# smooth spinal cord
sct_smooth_spinalcord.py -i t1_crop.nii.gz -c segmentation_binary_crop.nii.gz
# register to template (template registered to t2). N.B. only uses segmentation (more accurate)
sct_register_multimodal.py -i ../t2/template2anat.nii.gz -d t1_crop_smooth.nii.gz -x 1 -v 1 -n 15x3 -y 0 -g 0.2,0.5 -s ../t2/templateseg2anat.nii.gz -t segmentation_binary_crop.nii.gz
# concatenate transfo
sct_concat_transfo.py -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d t1_crop.nii.gz -o warp_template2t1.nii.gz
sct_concat_transfo.py -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_t12template.nii.gz
# warp template+atlas
sct_warp_template.py -d t1_crop.nii.gz -w warp_template2t1.nii.gz
# warp t1 to template
sct_apply_transfo.py -i t1_crop.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -w warp_t12template.nii.gz

# dmri
# ----------
mkdir dmri
cd dmri
dcm2nii -o . /Volumes/data_shared/montreal_criugm/errsm_30/26-ep2d_diff_2drf_0.8mm_24dir_AC_allCoil/errsm_30-0001.dcm
mv *.nii.gz dmri.nii.gz
mv *.bval bvals.txt
mv *.bvec bvecs.txt
# moco
sct_dmri_moco.py -i dmri.nii.gz -b bvecs.txt -d 3 -s 30 -p spline -f 1
# segment mean_dwi
sct_segmentation_propagation -i dwi_mean.nii -t t1
# register to template (template registered to t2). N.B. only uses segmentation (more accurate)
sct_register_multimodal.py -i ../t2/template2anat.nii.gz -d dwi_mean.nii -x 1 -v 1 -n 15x3 -y 5 -g 0.1,0.5 -s ../t2/templateseg2anat.nii.gz -t segmentation_binary.nii
# concatenate transfo
sct_concat_transfo.py -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d dwi_mean.nii -o warp_template2dmri.nii.gz
sct_concat_transfo.py -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_dmri2template.nii.gz
# warp template+atlas
sct_warp_template.py -d dwi_mean.nii -w warp_template2dmri.nii.gz
# warp dwi to template
sct_apply_transfo.py -i dwi_mean.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -w warp_dmri2template.nii.gz

# fmri
# ----------
mkdir fmri
cd fmri
dcm2nii -o . /Volumes/data_shared/montreal_criugm/errsm_30/46-rp_ep2d_bold_phcorr__test_1x1x3/errsm_30-0001.dcm
mv *.nii.gz fmri.nii.gz
# mean volume
fslmaths fmri.nii.gz -Tmean fmri_mean
# create initialization points for segmentation (otherwise does not work)
# === open fslview and create file fmri_init.nii.gz ===
# segment mean volume
sct_segmentation_propagation -i fmri_mean.nii.gz -t t2 -init-mask fmri_init.nii.gz -verbose
# === edit segmentation manually because not working well ===
# register to template (template registered to t2). N.B. only uses segmentation (more accurate)
sct_register_multimodal.py -i ../t2/template2anat.nii.gz -d fmri_mean.nii.gz -x 1 -v 1 -n 15x3 -y 5 -g 0.1,0.5 -s ../t2/templateseg2anat.nii.gz -t segmentation_binary.nii.gz
# concatenate transfo
sct_concat_transfo.py -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d fmri_mean.nii.gz -o warp_template2fmri.nii.gz
sct_concat_transfo.py -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_fmri2template.nii.gz
# warp template+atlas
sct_warp_template.py -d fmri_mean.nii.gz -w warp_template2fmri.nii.gz
# warp fmri to template
sct_apply_transfo.py -i fmri_mean.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -w warp_fmri2template.nii.gz

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

