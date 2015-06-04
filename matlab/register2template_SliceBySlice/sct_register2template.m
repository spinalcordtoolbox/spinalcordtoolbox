function sct_register2template(file_reg,file_src,levels,file_ref,verbose)
% sct_register2template(file_reg,file_src,levels [,file_ref,verbose])
%-------------------------- FILES TO REGISTER -----------------------------------
% file_reg = {'data_highQ_mean_masked'}; % file to register
%--------------------------------------------------------------------------
% file_src = 'data_highQ_mean_masked';
%--------------------------------------------------------------------------
% %-----------------------------REFERENCE (DESTINATION)------------------------------------
% ref_fname = '/Volumes/users_hd2/tanguy/data/Boston/2014-07/Connectome/template/PD_template.nii.gz';%'/home/django/tanguy/matlab/spinalcordtoolbox/data/template/MNI-Poly-AMU_WM.nii.gz';
% levels_fname='/home/django/tanguy/matlab/spinalcordtoolbox/data/template/MNI-Poly-AMU_level.nii.gz';
% %--------------------------------------------------------------------------
dbstop if error

if ~exist('verbose','var')
    verbose=false;
end
log='log_applytransfo';
% levels=5:-1:2;
warp_transfo = 1;
[~,sct_dir] = unix('echo $SCT_DIR'); sct_dir(end)=[];

%-------------------------- FILES TO REGISTER -----------------------------------
% file_reg = {'data_highQ_mean_masked'}; % file to register
%--------------------------------------------------------------------------

%-----------------------------REFERENCE (DESTINATION)------------------------------------
if ~exist('file_ref','var'), file_ref=[sct_dir '/data/template/MNI-Poly-AMU_T2.nii.gz']; end;
levels_fname=[sct_dir '/data/template/MNI-Poly-AMU_level.nii.gz'];
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
% file_src = 'data_highQ_mean_masked';
%--------------------------------------------------------------------------


%--------------------------------------------------------------------------
%---------------------------DON'T CHANGE BELOW-----------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------


% read template files
% read levels
levels_template=load_nii(levels_fname);
z_lev=[];
for i=levels
    [~,~,z]=find3d(levels_template.img==i); z_lev(end+1)=floor(mean(z));
end

% choose only good slices of the template
template=load_nii(file_ref);
template_roi=template.img(:,:,z_lev);
template_roi=make_nii(double(template_roi),[0.5 0.5 0.5],[],[]);
save_nii(template_roi,'template_roi.nii')
file_ref = 'template_roi';

%     % apply sqrt
%     unix('fslmaths template_roi -sqrt -sqrt template_roi_sqrt');
%     ref_fname = 'template_roi_sqrt';

files_ref = sct_sliceandrename(file_ref, 'z');
% splitZ source
files_src = sct_sliceandrename(file_src, 'z');

%--------------------------------------------------------------------------
% Estimate transfo between source and GW template
%--------------------------------------------------------------------------

for level = 1:length(levels)
    % affine
    cmd = ['isct_antsRegistration --metric MeanSquares[' files_ref{level} ',' files_src{level} ',1,32] --dimensionality 2 -o [reg_,filesrc_reg.nii.gz] --convergence 50 --smoothing-sigmas 0 --shrink-factors 1 --interpolation Linear -r [' files_ref{level} ',' files_src{level} ',0] --transform Affine[5]'];
    j_disp(log,['>> ',cmd]); [status result] = unix(cmd);
    
    % copy and rename matrix
    mat_folder{level} = ['mat_level' num2str(level)];
    if ~exist(mat_folder{level},'dir'), mkdir(mat_folder{level}); end
    unix(['mv reg_0GenericAffine.mat ' mat_folder{level} '/reg_1Affine.mat']);
    
    cmd = ['isct_antsRegistration --metric MeanSquares[' files_ref{level} ',filesrc_reg.nii.gz,1,32] --dimensionality 2 -o [reg_,test.nii] --convergence 10 --smoothing-sigmas 0 --shrink-factors 1 --interpolation Linear --transform BsplineSyN[0.2,2,0,3]'];
    j_disp(log,['>> ',cmd]); [status result] = unix(cmd);
	unix(['mv reg_0Warp.nii.gz ' mat_folder{level} '/reg_2Warp.nii.gz']);
    unix(['mv reg_0InverseWarp.nii.gz ' mat_folder{level} '/reg_2InverseWarp.nii.gz']);
    unix(['isct_ComposeMultiTransform 2 ' mat_folder{level} '/warp_final.nii.gz -R ' files_ref{level} ' ' mat_folder{level} '/reg_2Warp.nii.gz ' mat_folder{level} '/reg_1Affine.mat'])
    unix(['rm ' files_src{level}]);
end




%--------------------------------------------------------------------------
% apply transfo
%--------------------------------------------------------------------------

for i_file_reg = 1:length(file_reg)
    freg = load_nii(file_reg{i_file_reg});
    files_reg = sct_sliceandrename(file_reg{i_file_reg}, 'z');
    pause(0.5)
    for level = 1:freg.dims(3)
        if warp_transfo, warp_mat = [mat_folder{level} '/warp_final.nii.gz ']; else warp_mat = ' '; end
        % split
        files_tmp = sct_sliceandrename(files_reg{level}, 't');
        pause(0.5)
        mergelist='';
        for iT=1:freg.dims(4)
            % register reg file
            cmd = ['isct_antsApplyTransforms -d 2 -i ' files_tmp{iT} ' -o ' sct_tool_remove_extension(files_tmp{iT},1) '_reg.nii.gz -t ' warp_mat ' -r ' files_ref{level} ' -n BSpline[3]'];
            j_disp(log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
            mergelist=[mergelist ' ' sct_tool_remove_extension(files_tmp{iT},1) '_reg.nii.gz'];
        end
        pause(0.5)
        cmd = ['fslmerge -t ' sct_tool_remove_extension(files_reg{level},0) '_reg.nii.gz ' mergelist];
        j_disp(log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
        unix(['rm ' sct_tool_remove_extension(file_reg{i_file_reg},1) '*t*']);
        unix(['rm ' files_reg{level}]);
    end
    
    % merge files
    %reg
    mergelist='';
    for iZ=1:freg.dims(3)
        mergelist=[mergelist sct_tool_remove_extension(files_reg{iZ},0) '_reg '];
    end
    cmd = ['fslmerge -z ' sct_tool_remove_extension(file_reg{i_file_reg},1) '_reg ' mergelist];
    j_disp(log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    unix(['rm ' sct_tool_remove_extension(file_reg{i_file_reg},0) '_z*_reg*']);
    
    
end

% remove matrix
unix('rm -rf mat_level*');
% remove template
for level = 1:freg.dims(3), delete([files_ref{level} '*']); end
%delete([ref_fname '*']);
% display
if verbose
    unix(['fslview template_roi ' sct_tool_remove_extension(file_reg{1},1) '_reg /Volumes/taduv/data/Boston/2014-07/Connectome/template_roi/atlas/WMtract__16_roi.nii &']);
end