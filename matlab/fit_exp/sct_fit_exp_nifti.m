fname = 'v4d.nii.gz';
load('param.mat')
TE=param.te;

%------------------

img = read_avw(fname);
coeffvals = sct_fit_exp(img,TE,1);

T2 = coeffvals(:,:,:,2);
save_avw_v2(T2,'T2','f',[1 1 1 3], fname,1)

Signal = coeffvals(:,:,:,1);
save_avw_v2(Signal,'Signal','f',[1 1 1 3], fname,1)