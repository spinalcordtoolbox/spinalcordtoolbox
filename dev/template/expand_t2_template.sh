#!/bin/bash
#
# This script registers the MNI-Poly T2 template to the AMU atlas

# copy files locally
cp ../../data/template/MNI-Poly-AMU_T2.nii.gz tmp.MNI-Poly-AMU_T2.nii.gz
cp ../../data/template/MNI-Poly-AMU_WM.nii.gz tmp.MNI-Poly-AMU_WM.nii.gz
cp ../../data/template/MNI-Poly-AMU_GM.nii.gz tmp.MNI-Poly-AMU_GM.nii.gz
cp ../../data/template/MNI-Poly-AMU_cord.nii.gz tmp.MNI-Poly-AMU_cord.nii.gz
cp ../../data/template/MNI-Poly-AMU_level.nii.gz tmp.MNI-Poly-AMU_level.nii.gz
cp ../../data/template/MNI-Poly-AMU_CSF.nii.gz tmp.MNI-Poly-AMU_CSF.nii.gz
cp ../../data/template/MNI-Poly-AMU_seg.nii.gz tmp.MNI-Poly-AMU_seg.nii.gz

# create AMU WMGM image
fslmaths tmp.MNI-Poly-AMU_WM -add tmp.MNI-Poly-AMU_GM tmp.MNI-Poly-AMU_WMGM

# threshold AMU
sct_c3d tmp.MNI-Poly-AMU_WMGM.nii.gz -threshold 0.2 10 1 0 -o tmp.MNI-Poly-AMU_WMGM_th.nii.gz

# smooth AMU
fslmaths tmp.MNI-Poly-AMU_WMGM_th -s 0.5 tmp.MNI-Poly-AMU_WMGM_th_smooth

# registration
isct_antsRegistration --dimensionality 3 --transform SyN[0.1,3,0] --metric CC[tmp.MNI-Poly-AMU_WMGM_th_smooth.nii.gz,tmp.MNI-Poly-AMU_T2.nii.gz,1,4] --convergence 30x10 --shrink-factors 2x1 --smoothing-sigmas 0x0mm --Restrict-Deformation 1x1x0 --output [tmp.reg,MNI-Poly-AMU_WMGM_T2_reg.nii.gz] --collapse-output-transforms 1 --interpolation BSpline[3] --winsorize-image-intensities [0.005,0.995] -x [tmp.MNI-Poly-AMU_WMGM_th.nii.gz]

# apply transformation to other files
WarpImageMultiTransform 3 tmp.MNI-Poly-AMU_cord.nii.gz MNI-Poly-AMU_cord.nii.gz -R tmp.MNI-Poly-AMU_cord.nii.gz --use-NN tmp.reg0Warp.nii.gz
WarpImageMultiTransform 3 tmp.MNI-Poly-AMU_level.nii.gz MNI-Poly-AMU_level.nii.gz -R tmp.MNI-Poly-AMU_level.nii.gz --use-NN tmp.reg0Warp.nii.gz
WarpImageMultiTransform 3 tmp.MNI-Poly-AMU_CSF.nii.gz MNI-Poly-AMU_CSF.nii.gz -R tmp.MNI-Poly-AMU_CSF.nii.gz --use-NN tmp.reg0Warp.nii.gz
WarpImageMultiTransform 3 tmp.MNI-Poly-AMU_seg.nii.gz MNI-Poly-AMU_seg.nii.gz -R tmp.MNI-Poly-AMU_seg.nii.gz --use-NN tmp.reg0Warp.nii.gz
