% =========================================================================
% FUNCTION
% j_getdir.m
%
% Retrieve Path name and remember it for next use.
%
% INPUT
% (opt)             structure
%   windows_title   string
%   file_selection	'pickfile'	-> multiple file selection (Default)
%					'matlab'	-> default matlab selection
%   output			'cell'		-> cell array (Default)
%					'array'		-> array
%
% OUTPUT
% path_name         array
%
% COMMENTS
% julien cohen-adad 2008-10-27
% =========================================================================
function [path_name] = j_getdir(opt)


% default initialization
path_selection = pwd;
windows_title	= 'Please select folder';
file_selection	= 'pickfile';
output			= 'cell';
if ~exist('opt'), opt = []; end

% user initialization
if isfield(opt,'windows_title'), windows_title = opt.windows_title; end
if isfield(opt,'file_selection'), file_selection = opt.file_selection; end
if isfield(opt,'output'), output = opt.output; end

% retrieve last path
fname_func = which('j_getdir');
[path_func file_func] = fileparts(fname_func);
fname_lastfolder = strcat(path_func,filesep,'lastfolder.mat');

% check if this file already exists
if exist(fname_lastfolder,'file')
    load(fname_lastfolder);
    if path_last
        if isdir(path_last)
            path_selection = path_last;
        end
    end
end

% Choose selection type
switch(file_selection)
	case 'matlab'
		% Matlab selection
		path_name{1} = uigetdir(path_selection,windows_title);

	case 'pickfile'
		path_name = j_uipickfiles('FilterSpec',path_selection,'Prompt',windows_title);

end

% if user pressed cancel
if ~iscell(path_name), return; end

% save path name for next time
path_last = strcat(fileparts(path_name{1}),filesep);
save(fname_lastfolder,'path_last');

% output
switch(output)
	case 'array'
		path_name = j_cell2array(path_name);
end