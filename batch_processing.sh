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
# spinal cord segmentation
# tips: we use "-max-deformation 3" otherwise the segmentation does not cover the whole spinal cord
# tips: we use "-init 130" to start propagation closer to a region which would otherwise give poor segmentation (try it with and without the parameter).
# tips: we use "-centerline-binary" to get the centerline, which can be used to initialize segmentation on other contrasts.
sct_propseg -i t2.nii.gz -t t2 -centerline-binary -mesh -max-deformation 4 -init 130
# check your results:
# >> fslview t2 -b 0,800 t2_seg -l Red -t 0.5 &
# tips: You can also use MITKWORKBENCH to view the mesh.
# At this point you should make labels. Here we can use the file labels.nii.gz, which contains labels at C3 (value=3) and T4 (value=11).
# register to template
sct_register_to_template -i t2.nii.gz -l labels.nii.gz -m t2_seg.nii.gz -o 1 -s normal -r 0
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
# >> sct_crop -i t1.nii.gz
# segmentation (used for registration to template)
sct_propseg -i t1.nii.gz -t t1
# check results
# >> fslview t1 -b 0,800 t1_seg -l Red -t 0.5 &
# adjust segmentation (it was not perfect)
# --> t1_seg_modif.nii.gz
# register to template (template registered to t2).
sct_register_multimodal -i ../t2/template2anat.nii.gz -d t1.nii.gz -s ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -t t1_seg_modif.nii.gz -r 0 -p 1,SyN,0.2,MI
# concatenate transfo
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d t1.nii.gz -o warp_template2t1.nii.gz
sct_concat_transfo -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_t12template.nii.gz
sct_warp_template -d t1.nii.gz -w warp_template2t1.nii.gz -a 0
# check results
# >> fslview t1.nii.gz label/template/MNI-Poly-AMU_T2.nii.gz -b 0,4000 label/template/MNI-Poly-AMU_level.nii.gz -l MGH-Cortical -t 0.5 label/template/MNI-Poly-AMU_GM.nii.gz -l Red-Yellow -b 0.5,1 label/template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.5,1 &
# warp T1 to template space
sct_apply_transfo -i t1.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -w warp_t12template.nii.gz
# check registration of T1 to template
# >> fslview t1_reg.nii.gz -b 0,800 $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -b 0,4000 &
# go back to root folder
cd ..


# dmri
# ----------
cd dmri
# create mask to help moco
sct_create_mask -i dmri.nii.gz -m coord,110x20 -s 60 -f cylinder
# motion correction
sct_dmri_moco -i dmri.nii.gz -b bvecs.txt -g 3 -m mask_dmri.nii.gz -p 2,2,1,MeanSquares -t 0
# check moco
# >> fslview -m ortho,ortho dmri_moco dmri &
# segment mean_dwi
# tips: use flag "-init" to start propagation from another slice, otherwise results are not good.
sct_propseg -i dwi_moco_mean.nii.gz -t t1 -init 3
# check segmentation
# >> fslview dwi_moco_mean dwi_moco_mean_seg -l Red -t 0.5 & 
# register to template (template registered to t2).
# tips: here, we register the spinal cord segmentation to the mean DWI image because the contrasts are similar
sct_register_multimodal -i ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -d dwi_moco_mean.nii.gz -s ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -t dwi_moco_mean_seg.nii.gz -p 30,SyN,0.1,MI -x linear
# concatenate transfo
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d dwi_moco_mean.nii.gz -o warp_template2dmri.nii.gz
sct_concat_transfo -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_dmri2template.nii.gz
# warp template and white matter atlas
sct_warp_template -d dwi_moco_mean.nii.gz -w warp_template2dmri.nii.gz
# visualize white matter template on DWI
# >> fslview dwi_moco_mean label/template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.2,1 &
# compute tensors (using FSL)
dtifit -k dmri_moco -o dti -m dwi_moco_mean -r bvecs.txt -b bvals.txt
# compute FA within lateral cortico-spinal tracts from slices 1 to 3
sct_extract_metric -i dti_FA.nii.gz -f label/atlas/ -l 2,17 -z 1:3
# go back to root folder
cd ..


# mt
# ----------
cd mt
# register mt0 on mt1
sct_register_multimodal -i mt0.nii.gz -d mt1.nii.gz -z 3 -p 5,SyN,0.2,MI
# compute mtr
sct_compute_mtr -i mt0_reg.nii.gz -j mt1.nii.gz
# create initialization points on mt1 to help segmentation. Three points in middle of the cord.
echo -8.50 0.84 24.17 1 > landmarks.txt
echo -8.84 0.41 7.14 1 >> landmarks.txt
echo -8.34 0 -9.87 1 >> landmarks.txt
sct_c3d mt1.nii.gz -scale 0 -landmarks-to-spheres landmarks.txt 0.5 -o mt1_init.nii.gz
# segment mt1
sct_propseg -i mt1.nii.gz -t t2 -init-mask mt1-mask.nii.gz -detect-radius 5 -max-deformation 5
# check results
# >> fslview mt1 -b 0,800 mt1_seg.nii.gz -l Red -t 0.5 &
# register to template (template registered to t2).
sct_register_multimodal -i ../t2/template2anat.nii.gz -d mt1.nii.gz -p3,SyN,0.1,MI -s ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -t mt1_seg.nii.gz
# concatenate transfo
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d mt1.nii.gz -o warp_template2mt.nii.gz
sct_concat_transfo -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_mt2template.nii.gz
# warp template and atlas
sct_warp_template -d mt1.nii.gz -w warp_template2mt.nii.gz
# extract MTR within the whole white matter
sct_extract_metric -i mtr.nii.gz -f label/atlas/ -a
# go back to root folder
cd ..


# fmri
# ----------
cd fmri
# moco
sct_fmri_moco -i fmri.nii.gz
# put T2 centerline into fmri space
sct_c3d fmri_moco_mean.nii.gz ../t2/t2_centerline.nii.gz -reslice-identity -interpolation NearestNeighbor -o t2_centerline.nii.gz
# segment mean volume
# tips: we use the T2 centerline to help initialize the segmentation
# tips: we use "-radius 6" otherwise the segmentation is too small
sct_propseg -i fmri_moco_mean.nii.gz -t t2 -init-centerline t2_centerline.nii.gz -radius 6
# check segmentation
# >> fslview fmri_moco_mean fmri_moco_mean_seg -l Red -t 0.5 &
# here segmentation slightly failed due to the close proximity of susceptibility artifact --> use file "fmri_moco_mean_seg_modif.nii.gz"
# register to template (template registered to t2). Only uses segmentation (more accurate)
sct_register_multimodal -i ../t2/label/template/MNI-Poly-AMU_T2.nii.gz -d fmri_moco_mean.nii.gz -s ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -t fmri_moco_mean_seg_modif.nii.gz -n 3
# concatenate transfo
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d fmri_moco_mean.nii.gz -o warp_template2fmri.nii.gz
sct_concat_transfo -w warp_dest2src.nii.gz,../t2/warp_anat2template.nii.gz -d $SCT_DIR/data/template/MNI-Poly-AMU_T2.nii.gz -o warp_fmri2template.nii.gz
# warp template, atlas and spinal levels
sct_warp_template -d fmri_moco_mean.nii.gz -w warp_template2fmri.nii.gz -a 0 -s 1
