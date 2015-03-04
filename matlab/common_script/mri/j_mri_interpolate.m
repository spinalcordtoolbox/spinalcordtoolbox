% =========================================================================
% FUNCTION
% j_mri_interpolate
%
% 3d interpolation of MRI volume(s).
%
% INPUT
% (fname)           string. File name of the Analyze volume to interpolate.
%                   Put '' for manual selection if you want to use the field 'opt'.
%					Accepts the following formats:
%					- 3D/4D nifti (nii)
%					- 3D/4D compressed nifti (nii.gz)
%					- Analyze 7.5 (img)
%
% (opt)             structure
%   scale               1x3 integer. interpolation factor (default=[2 2 2]).
%   disp_text           binary. Display text of processing. Default = 1
%   norm_scale          binary. Default=1
%   method              string. 'cubic','linear'. Default = linear.
%	adjust_voxel		binary. Adjust scaling on voxel size instead of matrix size (Default = 1).
%
% OUTPUT
% (-)
%
% COMMENTS
% Generatated files have prefixe: 'interp_'
% julien cohen-adad 2009-10-01
% =========================================================================
function varargout = j_mri_interpolate(fname_read,opt)


% default initialization
scale       = [1 1 1.33];
disp_text   = 1;
norm_scale  = 1;
method      = 'linear'; % 'cubic','linear'
adjust_voxel= 1;
prefixe			= 'interp_';

% user initialization
if ~exist('fname_read'), fname_read = ''; end
if ~exist('opt'), opt = []; end
if isfield(opt,'scale'), scale = opt.scale; end
if isfield(opt,'disp_text'), disp_text = opt.disp_text; end
if isfield(opt,'norm_scale'), norm_scale = opt.norm_scale; end
if isfield(opt,'method'), method = opt.method; end
if isfield(opt,'adjust_voxel'), adjust_voxel = opt.adjust_voxel; end

% disp('***')
% disp('j_analyze_interpolate')
% disp('***')

% load data
if disp_text, j_progress('load data ..............'); end
if isempty(fname_read)
    opt_getfiles.ext_filter = 'img';
    opt_getfiles.windows_title = 'Select file(s) to interpolate';
    opt_getfiles.file_selection = 'spm';
    opt_getfiles.output = 'cell';
    fname_read_cell = j_getfiles(opt_getfiles);
    if ~iscell(fname_read_cell), return, end
else
    fname_read_cell{1} = fname_read;
end
[data hdr] = j_mri_read(fname_read_cell{1},opt);
[nx ny nz nt] = size(data);
j_progress(1)

% temp numbering prefix in case of 4d file
num = j_numbering(nt,3,0);

% loop on each file
if (disp_text), j_progress([method,' interpolation ...']); end
[XI,YI,ZI] = meshgrid(1/scale(2):1/scale(2):ny,1/scale(1):1/scale(1):nx,1/scale(3):1/scale(3):nz);
data_interp = double(zeros(size(XI,1),size(XI,2),size(XI,3)));
opt_write.disp_text = 0;
opt_write.norm_scale = norm_scale;
for i=1:nt
%     % load data
%     [data hdr] = j_analyze_read(fname_read_cell{i},opt);
    % 3d interpolation
%     extrapval = min(data(:));
	data3d = double(squeeze(data(:,:,:,i)));
    data_interp = interp3(data3d,XI,YI,ZI,method);
    % write interpolated b0 volume
	hdr_interp = hdr;
% 	hdr_interp.dim(1:3) = size(XI);
	hdr_interp.dims(1:3) = size(XI);
	scale_real = size(XI)./size(data3d);
% 	hdr_interp.mat(1:3,1:3) = hdr(i).mat(1:3,1:3)/diag(scale_real);
% 	hdr_interp.hdr.dime.pixdim(2:4) = hdr_interp.hdr.dime.pixdim(2:4)/diag(scale_real);
	if nt>1
		fname_write = ['temp',num{i},fname_read];
	else
		fname_write = [prefixe,fname_read];
	end
	pix_dim = hdr_interp.scales(1:3)./scale_real';
	save_avw(data_interp,fname_write,'s',pix_dim);
% 	j_mri_write(hdr_interp,prefixe_tmp,opt_write);
	if (disp_text), j_progress(i/nt); end
end

% merge file using FSL if originally from 4d
if nt>1
	j_progress('merge 3d data ..........')
	fname_write = [prefixe,fname_read];
	cmd = ['fslmerge -t ',fname_write,' temp*.*'];
	unix(cmd);
	j_progress(1)
	% delete temp files
	delete('temp*.*')
end
		
% display output
% if (disp_text)
%     for i=1:nt
%         fprintf('-> generated file: %s\n',strcat(file_write(i,:)));
%     end
% end

% disp('***')
