function sct_register2template(file_reg,file_src,levels,file_ref,verbose)
% sct_register2template(file_reg,file_src,levels [,file_ref,verbose])
% sct_register2template(file_reg,file_src,levels,[sct_dir '/dev/template/diffusion_template.nii'])
%
%-------------------------- FILES TO REGISTER -----------------------------------
% file_reg = {'data_highQ_mean_masked.nii','diameter.nii'}; % file to register
%------------------------------FILE SOURCE---------------------------------------
% file_src = 'data_highQ_mean_masked.nii';
%---------------------------Vertebral Levels----------------------------------
% levels = [5 4 3 2]; (Levels C5 to C1)
%--------------------------REFERENCE (DESTINATION)------------------------------------
% file_ref = [sct_dir '/dev/template/diffusion_template.nii'];
% use sct_template_extractlevels to warp atlas and templates
%--------------------------------------------------------------------------
dbstop if error

if ~exist('verbose','var')
    verbose=false;
end

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
if ~isempty(levels)
    levels_template=load_nii(levels_fname);
    z_lev=[];
    for i=levels
        [~,~,z]=find3d(levels_template.img==i); z_lev(end+1)=floor(mean(z));
    end


% choose only good slices of the template
template=load_nii(file_ref);
template_roi=template.img(:,:,z_lev);
src_nii=load_nii(file_src); % use slice thickness of the source
template_roi=make_nii(double(template_roi),[template.hdr.dime.pixdim(2:3) src_nii.hdr.dime.pixdim(4)],[],[]);
save_nii(template_roi,'template_roi.nii')
file_ref = 'template_roi.nii';
end

%--------------------------------------------------------------------------
% Estimate transfo between source and GW template
%--------------------------------------------------------------------------

sct_register_SbS(file_src,file_ref);



%--------------------------------------------------------------------------
% apply transfo
%--------------------------------------------------------------------------

for i_file_reg = file_reg
    if ~isempty(i_file_reg{1}), sct_unix(['sct_apply_transfo -i ' i_file_reg{1} ' -d ' file_ref ' -w warp_forward.nii'])    ; end
end

%delete([ref_fname '*']);
% display
if verbose
    unix(['fslview template_roi ' sct_tool_remove_extension(file_src,1) '_reg /Volumes/taduv/data/Boston/2014-07/Connectome/template_roi/atlas/WMtract__16_roi.nii &']);
end

%--------------------------------------------------------------------------
% Warp template
%--------------------------------------------------------------------------

% sct_unix(['sct_warp_template -d ' file_src ' -w warp_inverse.nii -t /Volumes/taduv/data/Boston/2014-07/Connectome/template_roi']);