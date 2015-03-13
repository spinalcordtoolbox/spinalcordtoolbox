function sct_fit_exp_nifti(fname,params)
% sct_fit_exp_nifti(fname,params)
% sct_fit_exp_nifti('b0_merged.nii.gz',[65 69 70 85]);
%------------------

img = read_avw(fname);
coeffvals = sct_fit_exp(img,params,1);

T2 = coeffvals(:,:,:,2);
save_avw_v2(T2,'T2','f',[1 1 1 3], fname,1)

Signal = coeffvals(:,:,:,1);
save_avw_v2(Signal,'Signal','f',[1 1 1 3], fname,1)