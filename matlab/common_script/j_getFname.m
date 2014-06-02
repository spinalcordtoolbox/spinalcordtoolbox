% =========================================================================
% FUNCTION
% j_getFname
%
% Useful stuff
%
% INPUT
% path_read			string (e.g., d:\pouf\*.img')
% (output_type)		string. 'array', cell' (Default = 'cell').
%
% OUTPUT
% fname_read		
%
% COMMENTS
% Julien Cohen-Adad 2008-10-23
% =========================================================================
function fname_read = j_getFname(path_read,output_type)


if nargin<2, output_type = 'cell'; end

% get all files
file_read = dir(strcat(path_read));

% retrieve path
path_read = fileparts(path_read);

% if no file
if ~size(file_read,1)
    fname_read = 0;
	return
end

% if UNIX OS, remove '.' and '..'
if strcmp(file_read(1).name,'.')
	i_start = 3;
	if strcmp(file_read(i_start).name,'.DS_Store')
		i_start = 4;
	end
else
	i_start=1;
end

% build fname
for i=i_start:size(file_read,1)
	switch(output_type)
		
		case 'cell'
		fname_read{i-i_start+1} = strcat(path_read,filesep,file_read(i).name);
	
		case 'array'
		fname_read(i-i_start+1,:) = strcat(path_read,filesep,file_read(i).name);
	end
end


