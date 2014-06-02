% =========================================================================
% FUNCTION
% j_numberLinesOfCode.m
%
% Find number of lines od code in a folder (include subfolders).
%
% INPUT
% (-)			
% 
% OUTPUT
% (-)
%
% COMMENTS
% julien cohen-adad 2008-06-29
% =========================================================================
function j_numberLinesOfCode()


% initialization
prefixe			= 'j_';
suffixe			= '.m';
path_script		= 'D:\mes_documents\matlab\script';

% retrieve all files
dir_script = j_dir(path_script,'.m');
nb_files = length(dir_script);

% loop on each file to count the number of lines
nb_lines = 0;
for iFile=1:nb_files
	fid = fopen(dir_script{iFile},'r');
	end_of_file = 0;
	while ~end_of_file
		get_line = fgetl(fid);
		nb_lines = nb_lines + 1;
		if ~isstr(get_line)
			end_of_file = 1;
		end
	end
	fclose(fid);
end

% display results
fprintf('\nPATH: %s\nNUMBER OF FILES: %i\nNUMBER OF LINES: %i\n',path_script,nb_files,nb_lines)
