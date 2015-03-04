% =========================================================================
% FUNCTION
% j_registerDTI.m
%
% Register diffusion MRI series, using first DW volume as the reference.
%
% INPUT
% (-)
% 
% OUTPUT
% (-)
%
% DEPENDENCES
% spm_coreg, spm_reslice, spm_get_space, j_writeAnalyze2, j_analyze_read,
% spm_vol
%
% COMMENTS
% julien cohen-adad 2007-07-26
% =========================================================================
function j_registerDTI()


% initialization
path_read  = 'D:\data_irm\analyses_irm\2007-07-18_exvivo_2007-07-17\38-DTI_dir100_b1000_register';

% get source files
file_read = ls(strcat(path_read,filesep,'*.img'));
fname_read = strcat(path_read,filesep,file_read);
clear file_read

hdr_target = spm_vol(fname_read(2,:));
nb_files = size(fname_read,1);

% estimate transformation parameters by registering B0 value from 1+nth DTI
% series to B0 value from 1st DTI series: (1x6) matrix
hdr_source      = spm_vol(fname_read);
opt.cost_fun    = 'nmi';
opt.sep         = [4 2];
opt.tol         = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
opt.fwhm        = [7 7];
x  = spm_coreg(hdr_target,hdr_source(51),opt);

% estimate matrix rigid transformation: (4x4) matrix
M  = inv(spm_matrix(x));
MM = zeros(4,4,nb_files);
for j=1:nb_files
    MM(:,:,j) = spm_get_space(deblank(hdr_source(j).fname));
end

% write .mat file for source image
for j=1:nb_files
    spm_get_space(deblank(hdr_source(j).fname), M*MM(:,:,j));
end

% reslice
opt.mean  = 0;
opt.which = 1;
opt.mean  = 0;
for j=1:nb_files
    P = str2mat(hdr_target.fname,hdr_source(j).fname);
    spm_reslice(P,opt);
end

% delete mat files
for j=1:nb_files
    [path_src file_src ext_src] = fileparts(hdr_source(j).fname);
    fname_src_mat = strcat(path_src,filesep,file_src,'.mat');
    delete(fname_src_mat);
    file_reg{j} = strcat('r',file_src);
    fname_reg_mat = strcat(path_src,filesep,file_reg{j},'.mat');
    delete(fname_reg_mat);
end

% change origin
for j=1:nb_files
    fname_reg = strcat(path_src,filesep,file_reg{j},ext_src);
    [data_reg hdr_reg] = j_analyze_read(fname_reg);
    opt.origin = [0 0 0];
    j_writeAnalyze2(data_reg,hdr_reg,'',opt);
end

