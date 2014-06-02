load('anat/anat_info.mat');
M_dmri=load('anat/dmri_vox2real_matrix.txt');

sct_apply_transfo('dmri/dwi_mean_total.nii',['anat/dwi_mean_registrated.nii'],anat_info.dim,anat_info.mat,M_dmri,0);