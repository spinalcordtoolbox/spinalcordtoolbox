#!/bin/bash
#
# Example of commands to process multi-parametric data of the spinal cord using the PAM50 template.
# For information about acquisition parameters, see: https://dl.dropboxusercontent.com/u/20592661/publications/Fonov_NIMG14_MNI-Poly-AMU.pdf
# N.B. The parameters are set for these type of data. With your data, parameters might be slightly different.
#
# tested with v2.1_beta20

# download the PAM50 template
sct_download_data -d PAM50

# create temporary variable to template
SCT_PAM50=`pwd`/PAM50
# N.B. To create permanent variable, add the following line to your .bashrc:
# SCT_PAM50="path_to_template"

# download example data (errsm_30)
sct_download_data -d sct_example_data

# display starting time:
echo "Started at: $(date +%x_%r)"

# go in folder
cd sct_example_data

#  t1
# ----------
cd t1
# spinal cord segmentation
sct_propseg -i t1.nii.gz -c t1
# check results
fslview t1 -b 0,800 t1_seg -l Red -t 0.5 &
# vertebral labeling. Here we use the fact that the axial slice #146 is located at the C4/C5 disc
sct_label_vertebrae -i t1.nii.gz -s t1_seg.nii.gz -initz 146,4
# create labels at C2 and T2 vertebral levels
sct_label_utils -i t1_seg_labeled.nii.gz -label-vert 2,9
# register to template
# tips: here we used only iter=1 for the third step for faster processing.
sct_register_to_template -i t1.nii.gz -s t1_seg.nii.gz -l labels.nii.gz -t $SCT_PAM50 -c t1 -param step=1,type=seg,algo=slicereg,metric=MeanSquares:step=2,type=seg,algo=bsplinesyn,iter=3,shrink=1,metric=MeanSquares:step=3,type=im,algo=syn,metric=CC,iter=1 -r 0
# warp template and white matter atlas
sct_warp_template -d t1.nii.gz -w warp_template2anat.nii.gz -t $SCT_PAM50 -a 0 -qc 0
# check results
fslview t1.nii.gz label/template/PAM50_T1.nii.gz -b 10,1500 label/template/PAM50_level.nii.gz -l MGH-Cortical -t 0.5 label/template/PAM50_GM.nii.gz -l Red-Yellow -b 0.5,1 label/template/PAM50_WM.nii.gz -l Blue-Lightblue -b 0.5,1 &
cd ..

# display ending time:
echo "Ended at: $(date +%x_%r)"
