function sct_fit_exp_nifti(fname,params)
% sct_fit_exp_nifti(fname,params)
% sct_fit_exp_nifti('b0_merged.nii.gz',[65 69 70 85]);
%------------------

img = read_avw(fname);
coeffvals = sct_fit_exp(img,params,1);

T2 = coeffvals(:,:,:,2);
save_nii_v2(T2,'T2',fname)

Signal = coeffvals(:,:,:,1);
save_nii_v2(Signal,'Signal',fname)