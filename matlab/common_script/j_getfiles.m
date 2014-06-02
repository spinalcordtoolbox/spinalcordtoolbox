% =========================================================================
% FUNCTION
% j_getfiles.m
%
% Retrieve Path and file name and remember it for next use.
% Allows multiple files selection
%
% INPUT
% (opt)             structure
%   ext_filter      string. Example: 'txt' (default='*')
%                   !!! if used with java option, then field are 'img'
%                   or 'image'
%   windows_title   string
%   file_selection  'spm' (can do multiple path selection)
%                   'java'* (for multiple file selection, also faster)
%                   'matlab' (can display text prompt and filter extensions)
%   output          'cell'*
%                   'array' (only one variable containing path,file and ext)
% 
% OUTPUT
% fname             cell or array (depending on 'opt.output')
% path_name         cell or array
% file_name         cell or array
% ext_read          cell or array
%
% DEPENDANCES
% uigetfile, j_uigetfiles, j_cell2array
% 
% COMMENTS
% * = default
% julien cohen-adad 2007-01-25
% =========================================================================
function [varargout] = j_getfiles(opt)

% load default parameters
j_parameters

% default initialization
windows_title	= 'Please select files(s)';
ext_filter		= '*';
file_selection	= 'spm';
output			= 'cell';
nb_files		= 1;

% user initialization
if ~exist('opt'), opt = []; end
if isfield(opt,'windows_title'), windows_title = opt.windows_title; end
if isfield(opt,'ext_filter'), ext_filter = opt.ext_filter; end
if isfield(opt,'file_selection'), file_selection = opt.file_selection; end
if isfield(opt,'output'), output = opt.output; end
if isfield(opt,'nb_files'), nb_files = opt.nb_files; end

% if SPM5 is not installed, choose Matlab file selection
if strcmp(file_selection,'spm')
	if ~exist('spm_select')
		file_selection = 'matlab';
	end
end

% retrieve actual path
path_curr = pwd;

% define the path where the file 'lastfolder.mat' should be
path_lastfolder = fileparts(which('j_getfiles.m'));
fname_lastfolder = strcat(path_lastfolder,filesep,'lastfolder.mat');

% retrieve path from last time
if exist(fname_lastfolder,'file')
	load(fname_lastfolder);
	if ~isdir(path_last)
		path_last = path_curr;
	end
else
	path_last = path_curr;
end
cd(path_last)

% interactive selection using different GUI

% SPM selection
if strcmp(file_selection,'spm')
	% SPM2
	if strcmp(param.spm.version,'spm2')
		fname_read = j_spm_get(Inf,ext_filter,sprintf(windows_title));
	% SPM5
	elseif strcmp(param.spm.version,'spm5')
		
		if strcmp(ext_filter,'*')
			ext_filter = 'any';
		end
		fname_read = spm_select(Inf,ext_filter,sprintf(windows_title),'',path_last);
	end
    % if user closed the selection window
    if isempty(fname_read)
        varargout{1} = 0;
        return;
    end
    for i=1:size(fname_read,1)
        [path_read{i} file_read{i} ext_read{i}] = fileparts(fname_read(i,:));
        path_read{i} = strcat(path_read{i},filesep);
    end
    clear fname_read
    
% JAVA selection
elseif strcmp(file_selection,'java')
    opt_getfiles.file_filter = ext_filter;
    opt_getfiles.windows_title = windows_title;
    [file_read_char path_read_char] = j_uigetfiles(opt_getfiles);
    % if user pressed cancel
    if ~file_read_char
        varargout{1} = 0;
        return;
    end
    for i=1:size(file_read_char,1)
        [tmp_path tmp_file tmp_ext] = fileparts(file_read_char(i,:));
        file_read{i} = tmp_file;
        path_read{i} = path_read_char;
        ext_read{i} = tmp_ext;
        clear tmp_file tmp_ext
    end
    clear file_read_char path_read_char
    
% Matlab selection
elseif strcmp(file_selection,'matlab')
	ext_filter = strcat('*.',ext_filter);
    [file_read path_read_arr] = uigetfile(ext_filter,windows_title,'MultiSelect','on');
    % if user pressed cancel
    if ~iscell(file_read)
		if ~file_read
			varargout{1} = 0;
			return;
		end
	end
	if ~iscell(file_read)
        tmp1 = file_read;
        clear file_read
        file_read{1} = tmp1;
        clear tmp1
    end
    for i=1:size(file_read,2)
        [path_read{i} file_read{i} ext_read{i}] = fileparts(strcat(path_read_arr,file_read{i}));
    end
    path_read = strcat(path_read,filesep);
end

fname = strcat(path_read,file_read,ext_read);

% save path name
path_last = path_read{1};
save(fname_lastfolder,'path_last');

% go back to previous path
cd(path_curr)

% output
if strcmp(output,'cell')
    varargout{1} = fname;
    varargout{2} = path_read;
    varargout{3} = file_read;
    varargout{4} = ext_read;
elseif strcmp(output,'array')
    % convert fname to array
    fname_arr = j_cell2array(fname);
    path_arr = j_cell2array(path_read);
    file_arr = j_cell2array(file_read);
    ext_arr = j_cell2array(ext_read);
    varargout{1} = fname_arr;
    varargout{2} = path_read;
    varargout{3} = file_arr;
    varargout{4} = ext_arr;
end



%==========================================================================
% OLD CODE

% CONVERT CELL INTO ARRAY
%-------------------------------------
% % find file max chars
% max_file_chars=0;
% for i=1:nb_files
%     if (max_file_chars<size(file_read{i},2))
%         max_file_chars = size(file_read{i},2);
%     end
% end
% 
% % find path max chars
% max_path_chars=0;
% for i=1:nb_files
%     if (max_path_chars<size(path_read{i},2))
%         max_path_chars = size(path_read{i},2);
%     end
% end
% 
% % transform cell into char array (fill with blanks when necessary)
% file_read_char = char(zeros(nb_files,max_file_chars));
% path_read_char = char(zeros(nb_files,max_path_chars));
% for i=1:nb_files
%     file_read_char(i,:) = [file_read{i},blanks(max_file_chars-size(file_read{i},2))];
%     path_read_char(i,:) = [path_read{i},blanks(max_path_chars-size(path_read{i},2))];
% end
% clear file_read path_read
% file_read = file_read_char;
% path_read = path_read_char;
% clear file_read_char path_read_char

