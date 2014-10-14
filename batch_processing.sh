# !/etc/bash
# example batch to process multi-parametric data of the spinal cord
# specific to errsm_30

# set default FSL output to be nii.gz
export FSLOUTPUTTYPE=NIFTI_GZ

# download example data
git clone https://github.com/neuropoly/sct_example_data.git

# go in folder
cd sct_example_data

# t2
# ===========================================================================================
cd t2
# orient data to RPI
sct_orientation -i t2.nii.gz -o t2.nii.gz -orientation RPI
# spinal cord segmentation
# tips: we use "-max-deformation 3" otherwise the segmentation does not cover the whole spinal cord
# tips: we use "-init 130" to start propagation closer to a region which would otherwise give poor segmentation (try it with and without the parameter).
sct_propseg -i t2.nii.gz -t t2 -centerline-binary -mesh -max-deformation 4 -init 130
# you can check results with "fslview". You can also use MITKWORKBENCH to view the mesh.
fslview t2 -b 0,800 t2_seg -l Red -t 0.5 &
# At this point you should make labels. Here we can use the file labels.nii.gz, which contains labels at C3 (value=3) and T4 (value=11).
# register to template
sct_register_to_template -i t2.nii.gz -l labels.nii.gz -m t2_seg.nii.gz -o 1 -s normal -r 1
# warp template and white matter atlas
sct_warp_template -d t2.nii.gz -w warp_template2anat.nii.gz
# compute cross-sectional area
# tips: flag "-b 1" will output a volume of CSA along the spinal cord. You can overlay it to the T2 volume.
sct_process_segmentation -i t2_seg.nii.gz -p compute_csa -b 1
# get average cross-sectional area between C2 and C4 levels
sct_extract_metric -i csa_volume.nii.gz -f label/template -l 0 -m wa -v 2:4
# go back to root folder
cd ..


#  t1
# ----------
cd t1
# crop data using graphical user interface (put two points)
sct_crop -i t1.nii.gz
# segmentation (used for registration to template)
sct_propseg -i t1.nii.gz -t t1
# check results
fslview t1 -b 0,800 t1_seg -l Red -t 0.5 &
# adjust segmentation (it was not perfect)
# --> t1_seg_modif.nii.gz
# register to template (template registered to t2).
sct_register_multimodal -i ../t2/template2anat.nii.gz -d t1.nii.gz -n 3 -g 0.2 -s ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -t t1_seg_modif.nii.gz -v 1
# check results
fslview t1 -b 0,800 template2anat_reg -b 10,4000 &

# concatenate transfo -- FIX ISSUE BEFORE DOING IT
#sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d t1.nii.gz -o warp_template2t1.nii.gz
#sct_concat_transfo -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_t12template.nii.gz
#sct_warp_template -d t1.nii.gz -w warp_template2t1.nii.gz -a 0
# warp T1 to template space
#sct_apply_transfo -i t1.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -w warp_t12template.nii.gz
# check registration of T1 to template
#fslview t1_reg.nii.gz -b 0,800 $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -b 0,4000 &

# ALTERNATIVE THAT WORKS:
# warp template
sct_warp_template -d t1.nii.gz -w warp_src2dest.nii.gz -p ../t2/label -a 0
# go back to root folder
cd ..


# dmri
# ----------
cd dmri
# moco option #1: volume-wise using flirt 2D, without grouping 
sct_dmri_moco -i dmri.nii.gz -b bvecs.txt -m flirt
# moco option #2: slice-wise using ants_affine (keep temporary folders)
# tips: flag "-s 10" creates a gaussian mask of 10mm FWHM to disregard motion from other structures (e.g. muscles)
# tips: flag "-d 5" improves robustness towards diffusion images with very low signal by averaging 5 adjacent images and doing a block-wise registration
sct_dmri_moco -i dmri.nii.gz -b bvecs.txt -m ants_affine -z 1 -r 0 -s 10 -d 5
# check moco
fslview -m ortho,ortho dmri_moco dmri &
# create "init-mask.nii.gz" on mean_dwi_moco (will be used for segmentation). Three points in middle of the cord.
fslview dwi_moco_mean &
# segment mean_dwi
# tips: use flag "-init" to start propagation from another slice, otherwise results are not good.
sct_propseg -i dwi_moco_mean.nii.gz -t t1 -init 3
# check segmentation
fslview dwi_moco_mean dwi_moco_mean_seg & 
# register to template (template registered to t2).
sct_register_multimodal -i ../t2/template2anat.nii.gz -d dwi_moco_mean.nii.gz -x 1 -v 1 -n 15x3 -y 5 -g 0.1,0.5 -s ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -t dwi_moco_mean_seg.nii.gz
# concatenate transfo
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d dwi_moco_mean.nii.gz -o warp_template2dmri.nii.gz
sct_concat_transfo -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_dmri2template.nii.gz
# warp template and white matter atlas
sct_warp_template -d dwi_moco_mean.nii.gz -w warp_template2dmri.nii.gz
# visualize white matter template on DWI
fslview dwi_moco_mean label/template/MNI-Poly-AMU_WM.nii.gz &
# compute tensors (using FSL)
dtifit -k dmri_moco -o dti -m dwi_moco_mean -r bvecs.txt -b bvals.txt
# compute FA within lateral cortico-spinal tracts on slice #1
sct_extract_metric -i dti_FA.nii.gz -f label/atlas/ -l 2,17 -z 1:1


# mt
# ----------
mkdir mt
cd mt
dcm2nii -o . /Volumes/data_shared/montreal_criugm/errsm_30/19-gre_t1_MTC1/errsm_30-0001.dcm
mv *.nii.gz mt1.nii.gz
dcm2nii -o . /Volumes/data_shared/montreal_criugm/errsm_30/20-gre_t1_MTC0/errsm_30-0001.dcm
mv 2*.nii.gz mt0.nii.gz
# compute MTR
sct_compute_mtr -i mt0.nii.gz -j mt1.nii.gz
# create "mt1-mask.nii.gz" on mt1 (will be used for segmentation). Three points in middle of the cord.
fslview mt1 &
# segment mt1
sct_propseg -i mt1.nii.gz -t t2 -init-mask mt1-mask.nii.gz -detect-radius 5 -max-deformation 5
# register to template (template registered to t2).
sct_register_multimodal -i ../t2/template2anat.nii.gz -d mt1.nii.gz -x 1 -v 1 -n 15x3 -y 5 -g 0.1,0.5 -s ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -t mt1_seg.nii.gz
# concatenate transfo
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d mt1.nii.gz -o warp_template2mt.nii.gz
sct_concat_transfo -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_mt2template.nii.gz
# warp template and atlas
sct_warp_template -d mt1.nii.gz -w warp_template2mt.nii.gz
# extract MTR within the whole white matter
sct_extract_metric -i mtr.nii.gz -f label/atlas/ -a


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
sct_propseg -i fmri_mean.nii.gz -t t2 -init-mask fmri_mean-mask.nii.gz -detect-radius 5
# register to template (template registered to t2). Only uses segmentation (more accurate)
sct_register_multimodal -i ../t2/template2anat.nii.gz -d fmri_mean.nii.gz -x 1 -v 1 -n 15x3 -y 0 -s ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -t fmri_mean_seg.nii.gz
# concatenate transfo
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d fmri_mean.nii.gz -o warp_template2fmri.nii.gz
sct_concat_transfo -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_fmri2template.nii.gz
# warp template and atlas
sct_warp_template -d fmri_mean.nii.gz -w warp_template2fmri.nii.gz
