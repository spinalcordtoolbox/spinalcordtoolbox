% =========================================================================
% FUNCTION
% j_readFile.m
%
% Read an ASCII file where numbers are put in lines
%
% INPUTS
% fname         string. Name of the file
% opt               structure
%   drop_line       integer. drop a specified number of first lines (in case of a header or something...). Default = 0.
%   data_type       'float'*
%                   'string'
%                   'dicom' special reading
%	read_method		'linePerLine' (read line by line - could be very long)
%					'allLines'*
%
% OUTPUTS
% data              matrix if numeric, cell if string
% fname         sring. File name.
%
% COMMENTS
% * = default
% Julien Cohen-Adad 2006-10-10
% =========================================================================
function varargout = j_readFile(fname,opt)


% default initialization

drop_line   = 0;
data_type   = 'float';
read_method = 'allLines';

% user initialization
if ~exist(fname), fname = ''; end
if ~exist('opt'), opt = []; end
if isfield(opt,'data_type'), data_type = opt.data_type; end
if isfield(opt,'drop_line'), drop_line = opt.drop_line; end
if isfield(opt,'read_method'), read_method = opt.read_method; end

% load file
if isempty(fname)
	opt.ext_filter = 'txt';
	opt.output = 'array';
    fname = j_getfiles(opt);
    if isempty(fname)
        varargout{1} = 0;
        return;
    end
end

% read line per line
if strcmp(read_method,'linePerLine')

	% Check if DICOM file
	if strcmp(data_type,'dicom')
		fid = fopen(fname,'r','b');
	else
		fid = fopen(fname, 'rt');
	end

	if (fid==-1) return; end;

	% drop first lines
	for i_line=1:drop_line
		line_read = fgetl(fid);
	end

	y = 1;
	loop = true;
	while loop
		line_read = fgetl(fid);
		if strcmp(data_type,'float')
			data(y,:) = sscanf(line_read,'%hg');
			if feof(fid) ~= 0
				loop = false;
			end
		elseif strcmp(data_type,'string')
			data{y} = line_read;
		elseif strcpm(data_type,'dicom')
			line_read = fgetl(fid);
			strfind(line_read,'pouf')
		end
		y=y+1;
	end
	fclose(fid);

% read the file at once
elseif strcmp(read_method,'allLines')

	% load file containing code
	txt = textread(fname,'%s','delimiter','\n');
	nb_lines = size(txt,1);

	% read each line as a string containing float
% 	j_progress('Read text file ...')
	for iLine = 1:nb_lines
		data(iLine,:) = sscanf(txt{iLine},'%hg');
% 		j_progress(iLine/nb_lines)
	end
end

varargout{1} = data;
varargout{2} = fname;
