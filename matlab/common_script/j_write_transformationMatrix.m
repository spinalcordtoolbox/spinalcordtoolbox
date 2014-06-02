% =========================================================================
% FUNCTION
% j_write_transformationMatrix
%
% Write transformation file in ASCII format. The matrix should contain 4
% lines and 4 rows.
% 
% INPUTS
% fname					string. Name of the txt file.
% transfo				4x4 float.
% 
% OUTPUTS
% (-)
% 
% COMMENTS
% Julien Cohen-Adad 2009-04-29
% =========================================================================
function j_write_transformationMatrix(fname,transfo)


% default initialization
if ~exist('fname'), help j_write_transformationMatrix; return; end
if ~exist('transfo'), help j_write_transformationMatrix; return; end

% read transformation matrix
fid = fopen(fname,'w');

% write transfo into file
transfo = transfo';
fprintf(fid,'%f %f %f %f\n',transfo);

% close file
fclose(fid);

