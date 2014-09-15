#! /bin/bash -f
#
# Compute TSNR. The anat.nii.gz and the fmri.nii.gz files should be there locally.
#
# julien cohen-adad
# 2013-07-29


# motion correct the fmri data
echo Motion correct the fMRI data...
mcflirt -in fmri -out fmri_moco

# compute tsnr
echo Compute the tSNR...
fslmaths fmri_moco -Tmean fmri_moco_mean
fslmaths fmri_moco -Tstd fmri_moco_std
fslmaths fmri_moco_mean -div fmri_moco_std fmri_tsnr

# register tsnr to anatomic
echo Register tSNR to anatomic...
sct_c3d  anat.nii.gz fmri_tsnr.nii.gz -reslice-identity -o fmri_tsnr_reslice.nii.gz

# Remove temp files
echo Remove temporary files...
rm fmri_moco_std.nii.gz

echo Done!

