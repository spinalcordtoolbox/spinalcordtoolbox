%% parameters
fname_t1map='T1_map.nii.gz';
fname_pd='PD_map.nii.gz';
fname_mtv='MTVF_map.nii.gz';

%% load images
t1map=load_nii(fname_t1map);
pd=load_nii(fname_t1map);
mtv=load_nii(fname_mtv);

%% 