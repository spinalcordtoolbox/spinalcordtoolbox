% =========================================================================
% FUNCTION
% j_mri_read.m
%
% Read MRI volumes.
% 
% Accepted format	extension
% ----------------------------
% Analyze 7.5			img
% Nifti					nii
% Compressed Nifti		nii.gz
%
%
% INPUTS
% (fname)           may be:
%                   - a path name on the form '.\pouf\'
%                   - a list of n files on the form <n,m>
%                   - a cell with n file names
%                   - '' for interactive selection
% (opt)             structure
%   precision       'single', 'double'*, 'logical'
%   windows_title    prompt text to display
%   warning         'on', 'off'*
%   check_numbering true, false. Algo that checks numbering in case of
%                       blank for some files. Default=false
%   file_selection  'spm'
%                   'java'* (for multiple file selection, also faster)
%                   'matlab' (can display text prompt and filter extensions)
%   output			'nifti','img'* (default)
%
% OUTPUTS
% data				est un tableau qui contient les donnees (3D ou 4D).
% header			est une structure qui contient une description du volume.
%
% COMMENTS
% Julien Cohen-Adad 2009-10-01
% =========================================================================
function varargout = j_mri_read(fname_read,opt)


% default initialization
windows_title           = 'Please select volume(s)';
% warning_status          = 'off';
check_numbering         = false;
file_selection          = 'spm';
path_current			= pwd;
% warning(warning_status)
prefixe_read            = '';
extension               = '*';
precision               = 'double';
nb_files				= 2;
output					= 'img';

% user initialization
if ~exist('fname_read'), fname_read = ''; end
if ~exist('opt'), opt = []; end
if isfield(opt,'windows_title'), windows_title = opt.windows_title; end
if isfield(opt,'warning'), warning_status = opt.warning; end
if isfield(opt,'check_numbering'), check_numbering = opt.check_numbering; end
if isfield(opt,'file_selection'), file_selection = opt.file_selection; end
if isfield(opt,'prefixe_read'), prefixe_read = opt.prefixe_read; end
if isfield(opt,'precision'), precision = opt.precision; end
if isfield(opt,'header_only'), header_only = opt.header_only; end
if isfield(opt,'output'), output = opt.output; end

% fname_read is empty
if isempty(fname_read)
	opt_getfile.output = 'array';
    opt_getfile.ext_filter = extension;
    opt_getfile.windows_title = windows_title;
    opt_getfile.file_selection = file_selection;
	opt_getfile.nb_files = nb_files;
    fname_read = j_getfiles(opt_getfile);
    if ~fname_read
        for i_argout=1:nargout
            varargout{i_argout}=0;
        end
        return;
    end

% fname_read is a cell
elseif iscell(fname_read)    
    fname_read_tmp = fname_read;
    clear fname_read;
    fname_read = j_cell2array(fname_read_tmp);
    clear fname_read_tmp;
%     for iFile=1:size(fname_read,1)
%         [path_read(iFile,:) file_read(iFile,:)] = fileparts(fname_read(iFile,:));
%     end

% fname_read is a directory
elseif exist(fname_read)==7    
    path_read = fname_read;
    file_read = dir(strcat(path_read,prefixe_read,'*.',extension));
	path_read = fname_read;
	fname_read = '';
	for i=1:size(file_read,1);
	    fname_read(i,:) = strcat(path_read,file_read(i).name);
	end

% fname_read is a file
elseif exist(fname_read)==2
	
% fname_read is part of a file
elseif ~isempty(dir([fname_read(1,:),'*.*']))
	fname_tmp = ls([fname_read(1,:),'*.*']);
	fname_read = fname_tmp(1,:);
	clear fname_tmp
	
% else, exit
else
	disp('This file doesn''t exist');
    varargout{1} = 0;
    varargout{2} = 0;
    varargout{3} = 0;
    varargout{4} = 0;
    return
end
nb_files = size(fname_read,1);

% check if all rows have same lenght. If not, there might be a problem of ordering
if (check_numbering)
    j = 1;
    k = 1;
    correct_numbering = false;
    for i=1:nb_files
        if strcmp(fname_read(i,end),' ')
            first_files(j) = i;
            j = j+1;
            correct_numbering = true;
        else
            last_files(k) = i;
            k = k+1;
        end
    end
    if (correct_numbering)
        fname_read = cat(1,fname_read(first_files,:),fname_read(last_files,:));
    end
end

% % get absolute file_read
% clear file_read
% for iFile=1:nb_files
%     [path_read(iFile,:) file_read(iFile,:)] = fileparts(fname_read(iFile,:));
% end
% if isempty(path_read)
%     path_read = '.';
% end
% path_read = strcat(path_read,filesep);
% cd(path_read(1,:));
% path_readAbs = strcat(pwd,filesep);
% cd(path_current);
% fname_readAbs = strcat(path_readAbs,file_read,extension);

% read MRI volume(s)
% header = zeros(size(fname_read,1),1);

% if extension is 'nii.gz', gunzip file
[path_read file_read extension] = fileparts(fname_read(1,:));
if ~isempty(findstr(extension,'.gz'))
	fname_read_gz = fname_read;
	clear fname_read;
	for i_file=1:size(fname_read_gz,1)
		[path_read file_read extension] = fileparts(fname_read_gz(i_file,:));
% 		unix(['gunzip ',fname_read_gz(i_file,:)]);
		gunzip(fname_read_gz(i_file,:));
		fname_read(i_file,:) = [path_read,filesep,file_read];
	end
end

% only return header
% if header_only
%     header = spm_vol(fname_read);
%     data = 0;
%     varargout{1} = data;
%     varargout{2} = header;
%     return;
% end

% if fname_read is a file
if exist(fname_read(1,:))==2
	
% FSL
% 	[data dims scales bpp endian] = read_avw(fname_read);
% 	hdr.dims = dims;
% 	hdr.scales = scales;
% 	hdr.bpp = bpp;
% 	hdr.endian = endian;

	% load_nii
	hdr = j_load_untouch_nii(fname_read);


% SPM
% 	warning off
%     header = spm_vol(fname_read);
% 	warning on
    % check if all .mat are the same, otherwise SPM won't open it...
%     for i_hdr=2:nb_files
%         if ~isempty(find((header(1).mat==header(i_hdr).mat)==0))
%             header(i_hdr).mat=header(1).mat;
%         end
%     end
%     data = j_spm_read_vols(header,0,precision);
%     data = spm_read_vols(header,0);

% else
%     data = 0;
end

% if strcmp(precision,'logical')
%     data = logical(data);
% end

% output
% dim = size(data);
switch (output)
	case 'nifti'
		varargout{1} = hdr;
	case 'img'
		varargout{1} = double(hdr.img);
end

% varargout{2} = hdr;
% varargout{3} = dim;
% varargout{4} = fname_read;

% if extension is 'nii.gz', delete generated nii file
if ~isempty(findstr(extension,'.gz'))
	for i_file = 1:size(fname_read,1)
		delete(fname_read(i_file,:));
	end
end

% % put warning on back
% if strcmp(warning_status,'off'), warning on; end

