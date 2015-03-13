% =========================================================================
% FUNCTION
% j_analyze_interpolate
%
% 3d interpolation of MRI volume(s).
%
% INPUT
% (fname)           string. File name of the Analyze volume to interpolate.
%                   Put '' for manual selection if you want to use the field 'opt'.
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
% julien cohen-adad 2007-04-10
% =========================================================================
function varargout = j_analyze_interpolate(fname_read,opt)


% default initialization
scale       = [1 1 1.33];
disp_text   = 1;
norm_scale  = 1;
method      = 'linear'; % 'cubic','linear'
adjust_voxel= 1;

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
if disp_text, disp('Load data...'); end
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
[data hdr] = j_analyze_read(fname_read_cell{1},opt);
[nx ny nz] = size(data);
nt = size(fname_read_cell,2);

% loop on each file
if (disp_text), j_progress([method,' interpolation...']); end
[XI,YI,ZI] = meshgrid(1/scale(2):1/scale(2):ny,1/scale(1):1/scale(1):nx,1/scale(3):1/scale(3):nz);
data_interp = zeros(size(XI,1),size(XI,2),size(XI,3));
opt_write.disp_text = 0;
opt_write.norm_scale = norm_scale;
for i=1:nt
    % load data
    [data hdr] = j_analyze_read(fname_read_cell{i},opt);
    % 3d interpolation
%     extrapval = min(data(:));
    data_interp = interp3(data,XI,YI,ZI,method);
    % write interpolated b0 volume
    hdr_interp = hdr;
    hdr_interp.dim(1:3) = size(XI);
    scale_real = size(XI)./size(data);
    hdr_interp.mat(1:3,1:3) = hdr.mat(1:3,1:3)/diag(scale_real);
    [path_write file_write(i,:)] = j_analyze_write(data_interp,hdr_interp,'interp_',opt_write);
    
    if (disp_text), j_progress(i/nt); end
end

% display output
if (disp_text)
    for i=1:nt
        fprintf('-> generated file: %s\n',strcat(file_write(i,:)));
    end
end

% disp('***')
