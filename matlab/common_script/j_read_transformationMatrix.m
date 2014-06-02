% =========================================================================
% FUNCTION
% j_read_transformationMatrix
%
% Read transformation file in ASCII format. The matrix should contain 4
% lines and 4 rows.
% 
% INPUTS
% fname					string. Name of the txt file.
% 
% OUTPUTS
% transfo				4x4 float.
%
% COMMENTS
% Julien Cohen-Adad 2009-04-29
% =========================================================================
function transfo = j_read_transformationMatrix(fname)


% default initialization
if ~exist('fname'), help j_read_transformationMatrix; return; end

% read transformation matrix
fid = fopen(fname,'r');

% check for errors
if fid<0
	transfo = 0;
	return; 
end

% read lines
transfo = fscanf(fid,'%f',[4 4]);
transfo = transfo';

% close file
fclose(fid);


