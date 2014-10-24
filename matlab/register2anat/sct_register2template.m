log='log_applytransfo';
uppest_level = 1;
levels=5:-1:2;
warp_transfo = 1;

%-------------------------- FILES TO REGISTER -----------------------------------
file_reg = {'data_highQ_mean_masked'}; % file to register
%--------------------------------------------------------------------------

%-----------------------------REFERENCE (DESTINATION)------------------------------------
ref_fname = 'fulltemplate';%'/home/django/tanguy/matlab/spinalcordtoolbox/data/template/MNI-Poly-AMU_WM.nii.gz';
levels_fname='/home/django/tanguy/matlab/spinalcordtoolbox/data/template/MNI-Poly-AMU_level.nii.gz';
%--------------------------------------------------------------------------


%--------------------------SOURCE FILE--------------------------------------
% data = 'KS_HCP35_crop_eddy_moco_lpca'; 
% scheme = 'KS_HCP.scheme';
% % Generate good source image (White Matter image)
% if ~exist([data '_ordered.nii'])
%     opt.fname_log = log;
%     sct_dmri_OrderByBvals(data,scheme,opt)
% end
% scd_generateWM([data '_ordered'],scheme,log);
% param.maskname='mask_spinal_cord';
% if ~exist(param.maskname)
%     param.file = 'data_highQ_mean'; 
%     scd_GenerateMask(param);
% end
% if ~exist('data_highQ_mean_masked.nii'), unix(['fslmaths data_highQ_mean -mul ' param.maskname ' data_highQ_mean_masked']), end
% file_src = 'data_highQ_mean_masked';
%----------------------------OR--------------------------------------------
file_src = 'data_highQ_mean_masked';
%--------------------------------------------------------------------------


ants_param = ' -r Gauss[3,1] -o reg_ -i 1x50 --number-of-affine-iterations 1000x1000x1000 --rigid-affine true --ignore-void-origin true';

%--------------------------------------------------------------------------
%---------------------------DON'T CHANGE BELOW-----------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
ext = '.nii.gz'; % do not change

% read file_reg dim
[~,dim] = read_avw(file_reg{1});
[~,dim_ref] = read_avw(ref_fname);


% read template files
    % read levels
    levels_template=read_avw(levels_fname);
    z_lev=[];
    for i=levels
        [~,~,z]=find3d(levels_template==i); z_lev(end+1)=floor(mean(z));
    end
        
    % choose only good slices of the template
    template=read_avw(ref_fname);
    template_roi=template(:,:,z_lev);
    save_avw_v2(template_roi(:,end:-1:1,:),'template_roi','f',[0.5 0.5 0.5 1])
    ref_fname = 'template_roi';

    % apply sqrt
    unix('fslmaths template_roi -sqrt -sqrt template_roi_sqrt');
    ref_fname = 'template_roi_sqrt';
    
files_ref = sct_sliceandrename(ref_fname);

% splitZ source
files_src = sct_sliceandrename(file_src);

%--------------------------------------------------------------------------
% Estimate transfo between source and GW template
%--------------------------------------------------------------------------

for level = 1:dim(3)
    cmd = ['ants 2 -m CC[' files_ref{level} ',' files_src{level} ext ',1,4] -t SyN ' ants_param];
    j_disp(log,['>> ',cmd]); [status result] = unix(cmd);
    
     % copy and rename matrix
     mat_folder{level} = ['mat_level' num2str(level)];
     if ~exist(mat_folder{level},'dir'), mkdir(mat_folder{level}); end
     unix(['mv reg_Warp.nii.gz ' mat_folder{level} '/reg_Warp.nii.gz']);
     unix(['mv reg_InverseWarp.nii.gz ' mat_folder{level} '/reg_InverseWarp.nii.gz']);
     unix(['mv reg_Affine.txt ' mat_folder{level} '/reg_Affine.txt']);
     unix(['rm ' files_src{level} ext]);

end




%--------------------------------------------------------------------------
% apply transfo
%--------------------------------------------------------------------------

for i_file_reg = 1:length(file_reg)
files_reg = sct_sliceandrename(file_reg{i_file_reg});
for level = 1:dim(3)
    if warp_transfo, warp_mat = [mat_folder{level} '/reg_Warp.nii.gz ']; else warp_mat = ' '; end
    % split
    files_tmp = sct_splitTandrename(files_reg{level});
    for iT=1:dim(4)
        % register reg file
        cmd = ['WarpImageMultiTransform 2 ' files_tmp{iT} ext ' ' files_tmp{iT} '_reg.nii.gz  -R ' files_ref{level} ' ' warp_mat  mat_folder{level} '/reg_Affine.txt --use-BSpline'];
        j_disp(log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    end
    cmd = ['fslmerge -t ' files_reg{level} '_reg.nii.gz ' file_reg{i_file_reg} '*T*_reg*'];
    j_disp(log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    unix(['rm ' file_reg{i_file_reg} '*T*']);
    unix(['rm ' files_reg{level} ext]);
end

% merge files
    %reg
cmd = ['fslmerge -z ' file_reg{i_file_reg} '_reg ' file_reg{i_file_reg} 'C*_reg*'];
j_disp(log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
unix(['rm ' file_reg{i_file_reg} 'C*_reg*']);


% swap
    %reg
cmd = ['fslswapdim ' file_reg{i_file_reg} '_reg x y -z ' file_reg{i_file_reg} '_reg' ];
j_disp(log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end

end


% remove matrix
unix('rm -rf mat_level*');
% remove template
for level = 1:dim(3), delete([files_ref{level} '*']); end
%delete([ref_fname '*']);