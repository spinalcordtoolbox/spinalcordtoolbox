# !/etc/bash
# 
# issue: https://sourceforge.net/p/spinalcordtoolbox/discussion/help/thread/37809790/
# author: julien cohen-adad
# 2015-02-25

# WARNING: script should be started in a folder that contains the folder "Raw_Data"

# check if data are in the same space
mkdir data
cd data/
mkdir diff
cd diff/
dcm2nii -o . ../../Raw_Data/Diffusion/*.*
mv *.nii.gz dmri.nii.gz
mv *.bval bvals.txt
mv *.bvec bvecs.txt
cd ..
mkdir t2
cd t2/
dcm2nii -o . ../../Raw_Data/T2/*.*
mv *.nii.gz t2.nii.gz
cd ../diff/
sct_dmri_separate_b0_and_dwi -i dmri.nii.gz -b bvecs.txt
fslmaths dwi.nii.gz -Tmean dwi_mean.nii.gz
sct_c3d dwi_mean.nii.gz ../t2/t2.nii.gz -reslice-identity t2_to_dmri_resliceId.nii.gz
fslview t2_to_dmri_resliceId.nii.gz dwi_mean.nii.gz &

# register template to t2
cd ../t2/
sct_resample -i t2.nii.gz -f 0.5x0.5x4
sct_label_utils -i t2r.nii.gz -t create -x 128,100,32,1:132,54,32,1:133,192,28,1:146,0,30,1 -o labels_propseg.nii.gz
sct_propseg -i t2r.nii.gz -t t2 -init-centerline labels_propseg.nii.gz
sct_label_utils -i t2r.nii.gz -t create -x 129,149,30,3:140,23,31,9 -o labels_vert.nii.gz
sct_register_to_template -i t2r.nii.gz -m t2r_seg.nii.gz -l labels_vert.nii.gz
sct_warp_template -d t2r.nii.gz -w warp_template2anat.nii.gz

# register template to dmri
cd ../diff/
sct_orientation -i dmri.nii.gz -o dmri_rpi.nii.gz -s RPI
sct_resample -i dmri_rpi.nii.gz -f 3x1x1
sct_dmri_separate_b0_and_dwi -i dmri_rpir.nii.gz -b bvecs.txt
fslmaths dwi.nii.gz -Tmean dwi_mean.nii.gz
sct_c3d dwi_mean.nii.gz ../t2/t2r.nii.gz -reslice-identity t2r_to_dmri_resliceId.nii.gz
sct_label_utils -i dwi_mean.nii.gz -t create -x 18,123,150,1:19,109,60,1:19,119,80,1:19,122,100,1:20,95,30,1:21,127,180,1:23,130,203,1 -o mask_centerline.nii.gz
sct_propseg -i dwi_mean.nii.gz -t t1 -verbose -init-centerline mask_centerline.nii.gz -min-contrast 10 -radius 5
sct_crop_image -i dwi_mean_seg.nii.gz -o dwi_mean_seg_crop.nii.gz -dim 2 -start 45 -end 195
sct_register_multimodal -i ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -d dwi_mean_seg_crop.nii.gz -s ../t2/label/template/MNI-Poly-AMU_cord.nii.gz -t dwi_mean_seg_crop.nii.gz -p 10,SyN,0.2,MeanSquares -x linear
sct_concat_transfo -w ../t2/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d dwi_mean.nii.gz -o warp_template2dmri.nii.gz
sct_warp_template -d dwi_mean.nii.gz -w warp_template2dmri.nii.gz
