# !/etc/bash
# 
# issue: http://sourceforge.net/p/spinalcordtoolbox/discussion/help/thread/37809790/
# author: julien cohen-adad
# 2015-02-16

# resample in R-L direction
sct_resample -i RPI_Oriented_Dif.nii.gz -f 3x1x1

# separate b=0 and DWI data
sct_dmri_separate_b0_and_dwi -i RPI_Oriented_Difr.nii.gz -b Diffusion_Test_Bvec.txt

# average across time
fslmaths dwi.nii.gz -Tmean dwi_mean

# SOLUTION #1: USING sct_get_centerline
# ==============================

# create a point in the spinal cord
sct_label_utils -i dwi_mean.nii.gz -t create -x 18,121,127,1 -o mask_point.nii.gz

# launch sct_get_centerline
sct_get_centerline -i dwi_mean.nii.gz -p mask_point.nii.gz

# SOLUTION #2: USING sct_propseg
# ==============================
# create points along centerline to generate spline
sct_label_utils -i dwi_mean.nii.gz -t create -x 18,123,150,1:19,109,60,1:19,119,80,1:19,122,100,1:20,95,30,1:21,127,180,1:23,130,203,1 -o mask_centerline.nii.gz

# launch propseg
sct_propseg -i dwi_mean.nii.gz -t t1 -verbose -init-centerline mask_centerline.nii.gz -min-contrast 10 -radius 5

