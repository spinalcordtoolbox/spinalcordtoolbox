function sct = sct_register_2_anat(sct)
% =========================================================================
% Module to coregistrate and generate anat
% 
% INPUT
% anat				structure
%
% MANDATORY
%   path_anat
%   path_b0
%   sct.anat.file
%
% OUTPUT
% none
% 
% 
% =========================================================================

% debug if error
dbstop if error
if isfield(sct.anat,'cost_spm_coreg'), cost_spm_coreg = sct.anat.cost_spm_coreg; else cost_spm_coreg = 'ncc'; end
options_spm_reslice = struct(...
                          'mask',1,... % don't mask anything
                          'mean',0,... % write mean image
                          'which',1,... % don't reslice the first image
                          'wrap',[0 0 0]',...
                          'interp',5); % the B-spline interpolation method

j_disp(sct.log,['\n\n   ANATOMICAL COREGISTRATION:'])
j_disp(sct.log,['-----------------------------------------------'])


% =========================================================================
%	GET ANATOMICAL INFOS
% =========================================================================
path_anat = [sct.output_path,'anat/'];
anat_file = [path_anat,sct.anat.file,'.nii'];
anat_info = spm_vol(anat_file);
% write anat infos
save([path_anat,'anat_info'],'anat_info');


% =========================================================================
%	DMRI2ANAT
% =========================================================================
j_disp(sct.log,['\n\n       DMRI2ANAT COREGISTRATION:'])
j_disp(sct.log,['       -----------------------------------------------'])
if isfield(sct.dmri,'folder')
    % initialise files names
    
    path_b0 = [sct.output_path,'dmri/'];
    if exist([path_b0,'b0_crop_mean.nii'])
        b0_file = [path_b0,'b0_crop_mean.nii'];
    else b0_file = [path_b0,'b0_mean.nii'];
    end
    
    b0_info = spm_vol(b0_file);
    
    if sct.anat.estimate
        j_disp(sct.log,['estimate transformation..'])
        % estimate a better transformation matrix
        options_spm_coreg = struct('cost_fun',cost_spm_coreg);
        transfo=spm_coreg2(anat_info,b0_info,options_spm_coreg);
        M_transfo=spm_matrix(transfo);

        % save estimation matrix for futur use
        fid = fopen([sct.output_path,'dmri/dmri_vox2real_matrix.txt'],'w');
        for L=1:4
            fprintf(fid,'%f %f %f %f\n',M_transfo(L,1),M_transfo(L,2),M_transfo(L,3),M_transfo(L,4));
        end
        fclose(fid);
        j_disp(sct.log,['.. File written: ','dmri/dmri_vox2real_matrix.txt (for futur use)'])
    else j_disp(sct.log,['no estimations..'])
    end
    

    
    
    % registrate b0 for demo
        % change voxel to world matrix in b0 header
        if sct.anat.estimate, spm_get_space(b0_file, M_transfo^(-1)*b0_info.mat); end
        % reslice
    options_spm_reslice.output = [path_anat,'b0_mean_anat_coreg.nii'];  
    spm_reslice2({anat_file, b0_file}, options_spm_reslice);
    
    j_disp(sct.log,['.. File written: ','anat/b0_mean_anat_coreg.nii (for demo)'])
else
    j_disp(sct.log,['..no diffusion files'])
end




% =========================================================================
%	MTR2ANAT
% =========================================================================

j_disp(sct.log,['\n\n       MTR2ANAT COREGISTRATION:'])
j_disp(sct.log,['       -----------------------------------------------'])
if isfield(sct.mtr,'folder')
    % initialise files names
    
    path_mtr = [sct.output_path,'mtr/'];
    mtr_file = [path_mtr,'MTR.nii'];
    mtr_info = spm_vol(mtr_file);
    
    if sct.anat.estimate
        j_disp(sct.log,['.. estimate transformation'])
        % estimate a better transformation matrix
        options_spm_coreg = struct('cost_fun','nmi');
        transfo=spm_coreg2(anat_info,mtr_info,options_spm_coreg);
        M_transfo=spm_matrix(transfo);
        
        % save estimation matrix for futur use
        fid = fopen([sct.output_path,'mtr/mtr_vox2real_matrix.txt'],'w');
        for L=1:4
            fprintf(fid,'%f %f %f %f\n',M_transfo(L,1),M_transfo(L,2),M_transfo(L,3),M_transfo(L,4));
        end
        fclose(fid);
        j_disp(sct.log,['.. File written: ','mtr/mtr_vox2real_matrix.txt (for futur use)'])
    
    else j_disp(sct.log,['.. no estimations'])
    end
    
    
    
    % registrate mtr
        % change voxel to world matrix in b0 header
    if sct.anat.estimate, spm_get_space(mtr_file, M_transfo^(-1)*mtr_info.mat); end
        % reslice
    options_spm_reslice.output = [path_mtr,'mtr_anat_coreg.nii'];  
    spm_reslice2({anat_file, mtr_file}, options_spm_reslice);
    
    j_disp(sct.log,['.. File written: ','mtr/mtr_anat_coreg.nii'])
else
    j_disp(sct.log,['..no mtr files'])
end

